"""Self-check for orchestrator.py's HTTP API: session lifecycle, claims,
and the full turn loop -- state-machine transitions plus the Analyst/
Interviewer/fusion/stoplogic wiring.

The Analyst/Interviewer calls are monkeypatched (no live Ollama here,
same spirit as design doc Section 5's "Analyst prompt fixtures" -- assert
the wiring reacts correctly to canned agent output, not that a live model
happens to say the right thing). Mic capture is monkeypatched too (no real
microphone in this environment); services/stt.py's own self-check already
covers the WAV-writing logic in isolation.
"""
import io
import json
import re
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orchestrator
from pypdf import PdfWriter


def make_test_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def fake_call_agent(host, model, prompt, validate=None, keep_alive=None, max_tokens=None):
    if "AROUSAL:" in prompt:
        data = {"state": "consistent", "plausibility": "adequate", "reasoning": "matches report", "new_claim_text": None}
    else:
        m = re.search(r"^(c\d+):", prompt, re.MULTILINE)
        data = {"type": "deepen", "claim_id": m.group(1) if m else None, "question": "Ceritakan lebih lanjut?"}
    if validate:
        validate(data)
    return data


def start_server(logs_dir):
    orchestrator.llm.call_agent = fake_call_agent
    orchestrator.Orchestrator._capture_answer = lambda self, wav_path: ("saya ada di rumah", False)

    server = orchestrator.ThreadingHTTPServer(
        ("localhost", 0), orchestrator.make_handler(orchestrator.Orchestrator(logs_dir=logs_dir))
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def call(port, method, path, body=None):
    if body is not None and not isinstance(body, (bytes, bytearray)):
        body = json.dumps(body).encode()
    req = urllib.request.Request(f"http://localhost:{port}{path}", data=body, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def main():
    with tempfile.TemporaryDirectory() as tmp:
        server, port = start_server(tmp)
        try:
            status, state = call(port, "GET", "/state")
            assert status == 200 and state == {"active": False}, state

            status, health = call(port, "GET", "/health")
            assert status == 200 and {h["status"] for h in health} <= {"ok", "idle", "down"} and len(health) == 5, health

            status, logs = call(port, "GET", "/logs")
            assert status == 200 and "text" in logs, logs
            assert call(port, "GET", "/logs/dashboard")[0] == 200
            assert call(port, "GET", "/logs/nope")[0] == 404

            status, _ = call(port, "POST", "/record/start")
            assert status == 409, "should refuse actions before any session exists"

            status, session_a = call(port, "POST", "/sessions", make_test_pdf())
            assert status == 200 and session_a["phase"] == "idle", session_a
            session_a_id = session_a["session_id"]

            status, sessions = call(port, "GET", "/sessions")
            assert status == 200 and len(sessions) == 1 and sessions[0]["session_id"] == session_a_id, sessions

            status, state = call(port, "POST", "/claims", ["was at home at 10pm"])
            assert status == 200 and list(state["ledger"].values())[0]["state"] == "unverified", state
            claim_id = list(state["ledger"].keys())[0]

            status, state = call(port, "POST", "/calibrate/start")
            assert status == 200 and state["phase"] == "calibration", state

            status, _ = call(port, "POST", "/claims", ["too late"])
            assert status == 409, "claims can't be added once calibration has started"

            # Session A is now mid-calibration (unfinished). Create session B on top of it.
            status, session_b = call(port, "POST", "/sessions", make_test_pdf())
            assert status == 200 and session_b["session_id"] != session_a_id, session_b

            status, sessions = call(port, "GET", "/sessions")
            ids = {s["session_id"] for s in sessions}
            assert ids == {session_a_id, session_b["session_id"]}, sessions

            # Reopen session A -- should resume exactly where it was left (mid-calibration).
            status, state = call(port, "POST", f"/sessions/{session_a_id}/open")
            assert status == 200 and state["phase"] == "calibration" and state["session_id"] == session_a_id, state

            status, _ = call(port, "POST", "/sessions/does-not-exist/open")
            assert status == 404

            status, _ = call(port, "DELETE", f"/sessions/{session_b['session_id']}")
            assert status == 200
            status, _ = call(port, "DELETE", f"/sessions/{session_b['session_id']}")
            assert status == 404
            status, _ = call(port, "DELETE", "/sessions/..")
            assert status == 404
            status, sessions = call(port, "GET", "/sessions")
            assert {s["session_id"] for s in sessions} == {session_a_id}, sessions

            status, state = call(port, "POST", "/calibrate/stop")
            assert status == 200 and state["phase"] == "interview", state

            status, state = call(port, "POST", "/generate-question")
            assert status == 200 and state["current_question"] and state["current_claim_id"] == claim_id, state

            status, state = call(port, "POST", "/record/start")
            assert status == 200 and state["recording"] is True, state

            status, _ = call(port, "POST", "/generate-question")
            assert status == 409, "should refuse a question while recording"

            status, state = call(port, "POST", "/record/stop")
            assert status == 200 and state["recording"] is False, state
            assert state["ledger"][claim_id]["state"] == "consistent", state["ledger"]
            assert state["cleared_claims"] == [claim_id], state
            assert state["confidence"] > 0, state
            # only claim already resolved -> all-claims-probed stop condition fires
            assert state["stop_recommended"] is True and state["stop_reason"] == "all_claims_probed", state

            status, _ = call(port, "POST", "/generate-question")
            assert status == 409, "orchestrator should refuse a further question once stop fired"

            status, state = call(port, "POST", "/stop-interview")
            assert status == 200 and state["phase"] == "stopped", state

            status, sessions = call(port, "GET", "/sessions")
            assert session_a_id not in {s["session_id"] for s in sessions}, "stopped session shouldn't be listed"

            # No mic/STT: stop is refused (still recording), then a typed answer completes the turn.
            call(port, "POST", "/sessions", make_test_pdf())
            call(port, "POST", "/claims", ["was at home at 10pm"])
            call(port, "POST", "/calibrate/start")
            call(port, "POST", "/calibrate/stop")
            call(port, "POST", "/generate-question")
            call(port, "POST", "/record/start")

            def no_mic(self, wav_path):
                raise orchestrator.OrchestratorError("no audio captured")

            orchestrator.Orchestrator._capture_answer = no_mic
            status, _ = call(port, "POST", "/record/stop")
            assert status == 409, "no audio and no typed answer should be refused"
            status, state = call(port, "POST", "/record/stop", b"saya di rumah")
            assert status == 200 and state["recording"] is False, state
            assert state["transcript"][-1]["answer"] == "saya di rumah", state["transcript"]

            print("test_orchestrator_api.py passed")
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
