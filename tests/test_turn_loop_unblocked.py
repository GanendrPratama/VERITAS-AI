"""Self-check that slow LLM/STT calls don't block the API, and that failures come
back as JSON errors instead of dropped connections.

The Interviewer/Analyst calls are replaced by fakes that wait on an Event, so the
test controls exactly when the "model" answers and can poll /state, /health and
the turn actions in between.
"""
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orchestrator
from services import sensors, stt
from test_orchestrator_api import call, fake_call_agent, make_test_pdf, start_server

release = threading.Event()  # set -> the fake model answers
entered = threading.Event()  # set by the fake once a call is in flight
fail_next = []  # truthy -> the next agent call raises


def slow_call_agent(*args, **kwargs):
    entered.set()
    assert release.wait(10), "test never released the fake model"
    if fail_next:
        fail_next.pop()
        raise ConnectionError("Ollama went away")
    return fake_call_agent(*args, **kwargs)


def in_background(port, method, path, body=None):
    result = {}
    entered.clear()  # earlier unblocked calls also set it
    t = threading.Thread(target=lambda: result.update(r=call(port, method, path, body)), daemon=True)
    t.start()
    return t, result


def wait_entered():
    assert entered.wait(5), "agent call never started"


def timed(port, method, path):
    start = time.time()
    status, body = call(port, method, path)
    return status, body, time.time() - start


def interview_ready(port):
    call(port, "POST", "/sessions", make_test_pdf())
    call(port, "POST", "/claims", ["was at home at 10pm", "met Budi"])
    call(port, "POST", "/calibrate/start")
    status, state = call(port, "POST", "/calibrate/stop")  # used to crash without bleak installed
    assert status == 200 and state["phase"] == "interview", state


def main():
    # Pure baseline math stays importable without bleak; the BLE service itself refuses.
    assert sensors.baseline_from_window([(0, "hr", 60), (1, "hr", 62)])["hr"][0] == 61
    real_client = sensors.BleakClient
    sensors.BleakClient = None
    try:
        sensors.SensorService("x", "a", "b")
        assert False, "should refuse to build without bleak"
    except ImportError:
        pass
    finally:
        sensors.BleakClient = real_client

    # A recording with no audio is an error, not a silent empty WAV fed to Whisper.
    try:
        stt.Recorder().stop("/nonexistent/never-written.wav")
        assert False, "should raise with no audio"
    except RuntimeError:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        server, port = start_server(tmp)
        orchestrator.llm.call_agent = slow_call_agent
        try:
            interview_ready(port)

            # Generate Question: state/health stay fast while the LLM call runs; a second
            # turn action is refused instead of queueing behind it.
            release.clear()
            t, result = in_background(port, "POST", "/generate-question")
            wait_entered()
            for path in ("/state", "/health"):
                status, _, took = timed(port, "GET", path)
                assert status == 200 and took < 2, (path, took)
            status, body = call(port, "POST", "/generate-question")
            assert status == 409 and "still working" in body["error"], body
            status, body = call(port, "POST", "/record/start")
            assert status == 409 and "still working" in body["error"], body
            release.set()
            t.join(5)
            status, state = result["r"]
            assert status == 200 and state["current_question"] and state["question_count"] == 1, state

            # Record/Stop with a typed answer: the Analyst call doesn't block /state either.
            call(port, "POST", "/record/start")
            release.clear()
            t, result = in_background(port, "POST", "/record/stop", b"saya di rumah")
            wait_entered()
            status, state, took = timed(port, "GET", "/state")
            assert status == 200 and took < 2 and state["recording"] is True, (took, state)
            release.set()
            t.join(5)
            status, state = result["r"]
            assert status == 200 and state["recording"] is False, state
            assert state["transcript"][-1]["answer"] == "saya di rumah", state["transcript"]

            # Analyst failure: still recording, and a bare Stop retries with the same answer.
            call(port, "POST", "/generate-question")
            call(port, "POST", "/record/start")
            fail_next.append(True)
            status, body = call(port, "POST", "/record/stop", b"saya tidak ingat")
            assert status == 502 and "press Stop again" in body["error"], body
            status, state = call(port, "GET", "/state")
            assert state["recording"] is True, state
            status, state = call(port, "POST", "/record/stop")
            assert status == 200 and state["transcript"][-1]["answer"] == "saya tidak ingat", state

            # Stopping the interview mid-call discards the late answer instead of reviving the session.
            interview_ready(port)
            release.clear()
            t, result = in_background(port, "POST", "/generate-question")
            wait_entered()
            status, state = call(port, "POST", "/stop-interview")
            assert status == 200 and state["phase"] == "stopped", state
            release.set()
            t.join(5)
            status, body = result["r"]
            assert status == 409 and "discarded" in body["error"], body
            status, state = call(port, "GET", "/state")
            assert state["phase"] == "stopped" and state["current_question"] is None, state

            # Bad input and unexpected exceptions come back as JSON errors, not a dropped connection.
            status, body = call(port, "POST", "/claims", 123)
            assert status == 400, body
            real = orchestrator.Orchestrator.list_sessions
            orchestrator.Orchestrator.list_sessions = lambda self: 1 / 0
            try:
                status, body = call(port, "GET", "/sessions")
                assert status == 500 and "ZeroDivisionError" in body["error"], body
            finally:
                orchestrator.Orchestrator.list_sessions = real

            print("test_turn_loop_unblocked.py passed")
        finally:
            release.set()
            server.shutdown()


if __name__ == "__main__":
    main()
