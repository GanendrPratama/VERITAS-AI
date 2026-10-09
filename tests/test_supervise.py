"""Self-check for scripts/supervise.py's watchdog and uptime log.

Runs the real supervisor (timings shrunk) over a stand-in orchestrator: a
tiny HTTP server whose /ping and /state can be told, via a mode file, to
answer, hang, or exit -- so a frozen process, a /state stuck behind the
session lock, and a crash are each checked for a restart, and the uptime
log for what scripts/uptime.py reports.
"""
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import uptime

FAKE_ORCH = """
import sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
port, mode_file = int(sys.argv[1]), Path(sys.argv[2])

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        mode = mode_file.read_text().strip()
        if mode == "exit":
            mode_file.write_text("ok")
            import os; os._exit(3)
        if mode == "hang" or (mode == "state" and self.path == "/state"):
            time.sleep(30)
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"{}")
    def log_message(self, *a):
        pass

ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""

RUN_SUPERVISOR = """
import sys
from pathlib import Path
sys.path.insert(0, {scripts!r})
import supervise
supervise.ORCH_URL = "http://127.0.0.1:{port}"
supervise.UPTIME_LOG = Path({log!r})
supervise.PROBE_EVERY, supervise.PING_TIMEOUT, supervise.STATE_TIMEOUT = 0.2, 0.5, 0.5
supervise.PING_DEADLINE, supervise.STATE_DEADLINE, supervise.STARTUP_GRACE = 1.5, 2.5, 5
supervise.SUMMARY_EVERY = 0.5
supervise.main([sys.executable, {fake!r}, "{port}", {mode!r}])
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def events(log):
    rows = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return [r for r in rows if "event" in r]


def wait_for(cond, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.1)
    return False


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fake, mode, log = tmp / "fake_orch.py", tmp / "mode", tmp / "uptime.jsonl"
        fake.write_text(FAKE_ORCH)
        mode.write_text("ok")
        port = free_port()
        runner = tmp / "run_supervisor.py"
        runner.write_text(RUN_SUPERVISOR.format(scripts=str(ROOT / "scripts"), port=port, log=str(log),
                                                fake=str(fake), mode=str(mode)))
        sup = subprocess.Popen([sys.executable, str(runner)], stdout=None if "-v" in sys.argv else subprocess.DEVNULL, stderr=None if "-v" in sys.argv else subprocess.DEVNULL)
        try:
            def restarts():
                return [e for e in events(log) if e["event"] == "restart"]

            assert wait_for(lambda: any(e["event"] == "supervisor_start" for e in events(log))), "supervisor didn't start"
            time.sleep(1.5)  # some healthy probes

            # frozen process: /ping stops answering -> killed and restarted
            mode.write_text("hang")
            assert wait_for(lambda: len(restarts()) == 1), events(log)
            assert "/ping" in restarts()[0]["reason"], restarts()
            mode.write_text("ok")
            time.sleep(1.5)

            # /state stuck (e.g. a request holding the session lock) while /ping still answers
            mode.write_text("state")
            assert wait_for(lambda: len(restarts()) == 2), events(log)
            assert "/state blocked" in restarts()[1]["reason"], restarts()
            mode.write_text("ok")
            time.sleep(1.5)

            # plain crash
            mode.write_text("exit")
            assert wait_for(lambda: len(restarts()) == 3), events(log)
            assert restarts()[2] == {**restarts()[2], "code": 3, "reason": "exited"}, restarts()
            assert wait_for(lambda: mode.read_text() == "ok")
            time.sleep(1.5)

            days, by_reason = uptime.summarize(log.read_text().splitlines())
            probes, ok = sum(d[0] for d in days.values()), sum(d[1] for d in days.values())
            assert probes > 0 and 0 < ok < probes, (probes, ok)  # outages above were counted, healthy time too
            assert sum(by_reason.values()) == 3, by_reason
        finally:
            sup.terminate()
            sup.wait(15)

        if sys.platform != "win32":  # terminate() is a hard kill on Windows; no clean shutdown to check
            assert events(log)[-1]["event"] == "supervisor_stop", events(log)
            assert wait_for(lambda: not _port_open(port), 5), "orchestrator left running after supervisor stopped"
    print("test_supervise.py passed")


def _port_open(port):
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


if __name__ == "__main__":
    main()
