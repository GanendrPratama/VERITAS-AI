"""Keeps the orchestrator alive: restarts it whenever it exits, with
exponential backoff so a crash loop doesn't spin. Stops (and takes the
orchestrator down with it) on SIGTERM/SIGINT, or when killed by
scripts/stop.sh / stop.ps1.

Usage: python scripts/supervise.py [command ...]   (default: orchestrator.py)
"""
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BACKOFF = 30
HEALTHY_AFTER = 60  # a run this long resets the backoff

stopping = False
child = None


def log(msg):
    print(f"[supervisor] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def shutdown(signum, _frame):
    global stopping
    log(f"got signal {signum}, shutting down")
    stopping = True
    if child and child.poll() is None:
        child.terminate()


def main(cmd):
    global child
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    backoff = 1
    while not stopping:
        started = time.time()
        child = subprocess.Popen(cmd, cwd=ROOT)
        log(f"started pid {child.pid}")
        code = child.wait()
        if stopping:
            break
        backoff = 1 if time.time() - started > HEALTHY_AFTER else min(backoff * 2, MAX_BACKOFF)
        log(f"exited with code {code}; restarting in {backoff}s")
        deadline = time.time() + backoff
        while not stopping and time.time() < deadline:
            time.sleep(0.2)
    log("stopped")


if __name__ == "__main__":
    main(sys.argv[1:] or [sys.executable, "-u", "orchestrator.py"])
