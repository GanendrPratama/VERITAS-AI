"""Keeps the orchestrator alive: restarts it whenever it exits, with
exponential backoff so a crash loop doesn't spin. Stops (and takes the
orchestrator down with it) on SIGTERM/SIGINT, or when killed by
scripts/stop.sh / stop.ps1.

It also watches the orchestrator while it runs:
- restarts it if it stops answering (GET /ping silent for PING_DEADLINE s:
  the process is frozen even though it hasn't exited), or if GET /state has
  been blocked for STATE_DEADLINE s (a request stuck holding the session
  lock -- longer than any legit LLM/STT call);
- logs dashboard-visible availability (GET /state answering within the
  dashboard's 5s timeout) to logs/uptime.jsonl, one line per minute plus a
  line per restart. `python scripts/uptime.py` reports the percentage.

Usage: python scripts/supervise.py [command ...]   (default: orchestrator.py)
"""
import json
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORCH_URL = "http://127.0.0.1:8000"
UPTIME_LOG = ROOT / "logs" / "uptime.jsonl"
MAX_BACKOFF = 30
HEALTHY_AFTER = 60  # a run this long resets the backoff
PROBE_EVERY = 5
PING_TIMEOUT = 2
STATE_TIMEOUT = 5  # same as the dashboard's GET timeout, so "up" means up as the operator sees it
STARTUP_GRACE = 120  # imports (torch, py-feat) can take a while before the port opens
PING_DEADLINE = 30
STATE_DEADLINE = 300  # STT + two 60s LLM tries, with margin
SUMMARY_EVERY = 60

stopping = False
child = None


def log(msg):
    print(f"[supervisor] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def record(**fields):
    try:
        UPTIME_LOG.parent.mkdir(parents=True, exist_ok=True)
        with UPTIME_LOG.open("a") as f:
            f.write(json.dumps({"ts": round(time.time()), **fields}) + "\n")
    except OSError as e:
        log(f"can't write {UPTIME_LOG}: {e}")


def answers(path, timeout):
    try:
        with urllib.request.urlopen(f"{ORCH_URL}{path}", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def shutdown(signum, _frame):
    global stopping
    log(f"got signal {signum}, shutting down")
    stopping = True
    if child and child.poll() is None:
        child.terminate()


def kill_child():
    child.terminate()
    try:
        child.wait(10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


class Availability:
    """Per-minute probe counts for logs/uptime.jsonl."""

    def __init__(self):
        self.probes = self.ok = 0
        self.since = time.time()

    def add(self, ok):
        self.probes += 1
        self.ok += ok
        if time.time() - self.since >= SUMMARY_EVERY:
            self.flush()

    def flush(self):
        if self.probes:
            record(probes=self.probes, ok=self.ok)
        self.probes = self.ok = 0
        self.since = time.time()


def watch(avail):
    """Probe the running child until it exits or must be restarted. Returns why it was killed, or None."""
    started = time.time()
    last_ping = last_state = None  # None until the API first answers
    next_probe = started
    while not stopping and child.poll() is None:
        now = time.time()
        if now < next_probe:
            time.sleep(0.2)
            continue
        next_probe = now + PROBE_EVERY
        ping_ok = answers("/ping", PING_TIMEOUT)
        state_ok = ping_ok and answers("/state", STATE_TIMEOUT)
        now = time.time()
        if ping_ok:
            last_ping = now
        if state_ok:
            last_state = now
        avail.add(state_ok)  # startup counts as downtime too: the operator can't use it yet
        if last_ping is None:
            if now - started > STARTUP_GRACE:
                return f"API never came up within {STARTUP_GRACE}s"
            continue
        if now - last_ping > PING_DEADLINE:
            return f"no answer on /ping for {now - last_ping:.0f}s"
        if last_state is None:
            last_state = last_ping  # API just came up: start the /state clock now, not at the latest ping
        if now - last_state > STATE_DEADLINE:
            return f"/state blocked for {now - last_state:.0f}s"
    return None


def main(cmd):
    global child
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    avail = Availability()
    record(event="supervisor_start")
    backoff = 1
    while not stopping:
        started = time.time()
        child = subprocess.Popen(cmd, cwd=ROOT)
        log(f"started pid {child.pid}")
        reason = watch(avail)
        if reason:
            log(f"pid {child.pid} {reason} -- killing it")
            kill_child()
        code = child.wait()
        if stopping:
            break
        backoff = 1 if time.time() - started > HEALTHY_AFTER else min(backoff * 2, MAX_BACKOFF)
        record(event="restart", code=code, reason=reason or "exited")
        log(f"exited with code {code}; restarting in {backoff}s")
        deadline = time.time() + backoff
        while not stopping and time.time() < deadline:
            avail.add(False)  # down while waiting out the backoff
            time.sleep(min(PROBE_EVERY, max(0, deadline - time.time())))
    avail.flush()
    record(event="supervisor_stop")
    log("stopped")


if __name__ == "__main__":
    main(sys.argv[1:] or [sys.executable, "-u", "orchestrator.py"])
