"""The next auto-picked question is generated in the background once calibration finishes and
after every assessed answer, so "Generate Question" with "Let the interviewer choose" uses it
instead of a fresh LLM call. A focus claim or a typed question must not use a stale prefetch."""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orchestrator
from test_orchestrator_api import call, fake_call_agent, make_test_pdf, start_server

calls = []


def counting_call_agent(host, model, prompt, validate=None, **options):
    kind = "analyst" if "AROUSAL:" in prompt else "interviewer"
    calls.append((kind, options.get("cancel") is not None))
    assert options.get("num_ctx"), "every call carries the session's num_ctx"
    return fake_call_agent(host, model, prompt, validate)


def wait_for(cond, timeout=3):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def main():
    orchestrator.CONFIG["llm_prefetch_question"] = True
    orchestrator.CONFIG["llm_warmup"] = False
    with tempfile.TemporaryDirectory() as tmp:
        server, port = start_server(tmp)
        orchestrator.llm.call_agent = counting_call_agent
        try:
            call(port, "POST", "/sessions", make_test_pdf())
            call(port, "POST", "/claims", ["was at home at 10pm", "met Budi", "left at 11pm", "drove alone"])
            call(port, "POST", "/calibrate/start")
            call(port, "POST", "/calibrate/stop")
            assert wait_for(lambda: calls == [("interviewer", True)]), calls  # prefetched in the background

            status, state = call(port, "POST", "/generate-question", {"claim_id": None})
            assert status == 200 and state["current_question"], state
            assert calls == [("interviewer", True)], calls  # served from the prefetch, no second call

            call(port, "POST", "/record/start")
            status, state = call(port, "POST", "/record/stop", b"saya di rumah")
            assert status == 200, state
            assert wait_for(lambda: len(calls) == 3), calls
            assert calls[1:] == [("analyst", False), ("interviewer", True)], calls

            # A chosen claim doesn't match the auto-pick prefetch: fresh call about that claim.
            status, state = call(port, "POST", "/generate-question", {"claim_id": "c2"})
            assert status == 200 and state["current_claim_id"] == "c2", state
            assert calls[-1] == ("interviewer", False) and len(calls) == 4, calls

            # Typed question: answered, prefetch for the next one, then a typed question drops it.
            call(port, "POST", "/record/start")
            call(port, "POST", "/record/stop", b"bertemu Budi")
            assert wait_for(lambda: len(calls) == 6), calls
            status, state = call(port, "POST", "/ask-question", {"question": "Jam berapa?", "claim_id": "c1"})
            assert status == 200 and state["current_question"] == "Jam berapa?", state
            call(port, "POST", "/record/start")
            call(port, "POST", "/record/stop", b"jam sepuluh")
            assert wait_for(lambda: len(calls) == 8), calls
            status, state = call(port, "POST", "/generate-question")
            assert status == 200 and len(calls) == 8, calls  # the fresh prefetch, not the dropped one
        finally:
            server.shutdown()
    print("test_question_prefetch.py passed")


if __name__ == "__main__":
    main()
