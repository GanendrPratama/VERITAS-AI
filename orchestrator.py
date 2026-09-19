"""Orchestrator: session lifecycle, turn-loop state, agent calls, and a local
HTTP API for the Operator UI.

Stdlib-only HTTP server (no Flask/FastAPI) -- the API is a handful of
small JSON endpoints, not enough surface to justify a dependency. The UI
(ui/app.py) is a separate process that only ever talks to this API;
nothing here imports Streamlit and nothing in ui/app.py imports this
module.

Each session is a full state snapshot written to
`logs/<session_id>/state.json` after every state-changing action -- that
snapshot IS the resume mechanism for "open old session" (see design doc
Section 1/6). It's separate from the per-turn JSONL transcript log the
design also calls for, which is an append-only audit artifact, not a
resume point -- that log isn't wired up yet.

Hardware/model dependencies (bleak, py-feat/cv2, faster-whisper, Ollama)
are all best-effort: if a service can't be reached or a library isn't
installed, that channel's data just stays unavailable rather than crashing
the interview (Section 4: degrade, never crash).
"""
import json
import re
import threading
import time
import tomllib
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import fusion
import stoplogic
from services import llm, stt
from services.docconvert import pdf_to_markdown

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
CONFIG = tomllib.loads((ROOT / "config.toml").read_text())
ANALYST_TEMPLATE = (ROOT / "agents" / "analyst_prompt.md").read_text()
INTERVIEWER_TEMPLATE = (ROOT / "agents" / "interviewer_prompt.md").read_text()


class _NullHardwareService:
    """Stand-in when a hardware service's dependency isn't installed --
    channels it would provide just stay unavailable (Section 4)."""

    baseline = {}

    def start(self):
        pass

    def stop(self):
        pass

    def window(self, _start_ts, _end_ts):
        return []

    def delta_sd(self, _start_ts, _end_ts):
        return {}


def _build_sensors():
    try:
        from services.sensors import SensorService

        return SensorService(
            CONFIG["ble_device_name"], CONFIG["ble_hr_char_uuid"], CONFIG["ble_gsr_char_uuid"]
        )
    except ImportError:
        return _NullHardwareService()


def _build_vision():
    try:
        from services.vision import VisionService

        return VisionService(CONFIG["camera_index"], CONFIG["vision_device"])
    except ImportError:
        return _NullHardwareService()


def new_session_state(session_id, report_text):
    return {
        "session_id": session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "report_text": report_text,
        "phase": "idle",  # idle -> calibration -> interview -> stopped
        "recording": False,
        "current_question": None,
        "current_claim_id": None,
        "question_count": 0,
        "confidence": 0.0,
        "confidence_history": [],
        "flagged_claims": [],
        "cleared_claims": [],
        "ledger": {},
        "transcript": [],
        "stop_recommended": False,
        "stop_reason": None,
        "calibration_start_ts": None,
        "interview_start_ts": None,
        "record_start_ts": None,
    }


class OrchestratorError(Exception):
    def __init__(self, message, status=409):
        super().__init__(message)
        self.status = status


class Orchestrator:
    def __init__(self, logs_dir=LOGS_DIR, config=CONFIG):
        self.logs_dir = Path(logs_dir)
        self.config = config
        self.active = None
        self.lock = threading.Lock()
        self.sensors = _build_sensors()
        self.vision = _build_vision()
        self.recorder = stt.Recorder(samplerate=config["mic_samplerate"])
        self._whisper_model = None

    # -- persistence --

    def _session_path(self, session_id):
        return self.logs_dir / session_id / "state.json"

    def _save(self):
        path = self._session_path(self.active["session_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.active))

    def _new_session_id(self):
        base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session_id, n = base, 2
        while self._session_path(session_id).exists():
            session_id = f"{base}-{n}"
            n += 1
        return session_id

    def _next_claim_id(self):
        n = len(self.active["ledger"]) + 1
        while f"c{n}" in self.active["ledger"]:
            n += 1
        return f"c{n}"

    def _require_active(self):
        if self.active is None:
            raise OrchestratorError("no active session -- create or open one first")

    def _whisper(self):
        if self._whisper_model is None:
            self._whisper_model = stt.load_model(self.config["stt_model"], device=self.config["stt_device"])
        return self._whisper_model

    # -- session lifecycle --

    def create_session(self, pdf_bytes):
        with self.lock:
            try:
                report_text = pdf_to_markdown(pdf_bytes)
            except Exception as e:
                raise OrchestratorError(f"couldn't read PDF: {e}", status=400)
            self.active = new_session_state(self._new_session_id(), report_text)
            self._save()
            return dict(self.active)

    def list_sessions(self):
        summaries = []
        for path in sorted(self.logs_dir.glob("*/state.json")):
            try:
                state = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if state.get("phase") == "stopped":
                continue
            summaries.append({
                "session_id": state["session_id"],
                "created_at": state["created_at"],
                "phase": state["phase"],
                "label": (state.get("report_text") or "")[:80],
            })
        summaries.sort(key=lambda s: s["created_at"], reverse=True)
        return summaries

    def open_session(self, session_id):
        with self.lock:
            path = self._session_path(session_id)
            if not path.exists():
                raise OrchestratorError(f"no such session: {session_id}", status=404)
            self.active = json.loads(path.read_text())
            return dict(self.active)

    def get_state(self):
        with self.lock:
            if self.active is None:
                return {"active": False}
            return dict(self.active)

    # -- claims --

    def add_claims(self, texts):
        with self.lock:
            self._require_active()
            if self.active["phase"] != "idle":
                raise OrchestratorError("claims can only be added before calibration starts")
            for text in texts:
                text = text.strip()
                if text:
                    self.active["ledger"][self._next_claim_id()] = {
                        "text": text, "state": "unverified", "turns": [],
                    }
            self._save()
            return dict(self.active)

    # -- turn loop actions, scoped to self.active --

    def calibrate_start(self):
        with self.lock:
            self._require_active()
            if self.active["phase"] != "idle":
                raise OrchestratorError(f"cannot start calibration from phase {self.active['phase']}")
            self.sensors.start()
            self.vision.start()
            self.active["phase"] = "calibration"
            self.active["calibration_start_ts"] = time.time()
            self._save()

    def calibrate_stop(self):
        with self.lock:
            self._require_active()
            if self.active["phase"] != "calibration":
                raise OrchestratorError(f"cannot finish calibration from phase {self.active['phase']}")
            start = self.active["calibration_start_ts"]
            now = time.time()
            self.sensors.baseline = _baseline_from(self.sensors, start, now)
            self.vision.baseline = _baseline_from(self.vision, start, now)
            self.active["phase"] = "interview"
            self.active["interview_start_ts"] = now
            self._save()

    def record_start(self):
        with self.lock:
            self._require_active()
            if self.active["phase"] not in ("calibration", "interview"):
                raise OrchestratorError(f"cannot record during phase {self.active['phase']}")
            if self.active["recording"]:
                raise OrchestratorError("already recording")
            try:
                self.recorder.start()
            except Exception as e:
                print(f"[orchestrator] mic recording unavailable: {e}")
            self.active["recording"] = True
            self.active["record_start_ts"] = time.time()
            self._save()

    def record_stop(self):
        with self.lock:
            self._require_active()
            if not self.active["recording"]:
                raise OrchestratorError("not recording")
            self.active["recording"] = False
            record_start_ts = self.active["record_start_ts"]
            record_end_ts = time.time()

            wav_path = self._session_path(self.active["session_id"]).parent / f"turn_{self.active['question_count']}.wav"
            answer_text, unclear = self._capture_answer(wav_path)

            if self.active["phase"] == "interview":
                self._assess_answer(answer_text, unclear, record_start_ts, record_end_ts)
            self._save()

    def _capture_answer(self, wav_path):
        try:
            self.recorder.stop(str(wav_path))
        except Exception as e:
            print(f"[orchestrator] mic recording unavailable: {e}")
            return "", True
        try:
            result = stt.transcribe(self._whisper(), str(wav_path), self.config["stt_min_logprob"])
            return result["text"], result["unclear"]
        except Exception as e:
            print(f"[orchestrator] STT failed: {e}")
            return "", True

    def _assess_answer(self, answer_text, unclear, start_ts, end_ts):
        deltas = {}
        deltas.update(self.sensors.delta_sd(start_ts, end_ts))
        deltas.update(self.vision.delta_sd(start_ts, end_ts))
        arousal = fusion.bucket_arousal(deltas, self.config)

        claim_id = self.active["current_claim_id"]
        claim_text = self.active["ledger"][claim_id]["text"] if claim_id else None
        history = llm.trim_history(self.active["transcript"], self.config["llm_context_turns"])

        if unclear:
            response = {"state": "evasive", "plausibility": "non_answer",
                        "reasoning": "answer unclear (STT)", "new_claim_text": None}
        else:
            prompt = llm.build_analyst_prompt(
                ANALYST_TEMPLATE, self.active["report_text"], claim_text, history, answer_text, arousal
            )
            try:
                response = llm.call_agent(
                    self.config["ollama_host"], self.config["model"], prompt,
                    validate=llm.validate_analyst_response,
                    keep_alive=self.config["llm_keep_alive"], max_tokens=self.config["llm_max_tokens"],
                )
            except Exception as e:
                raise OrchestratorError(f"Analyst call failed: {e}", status=502)

        if claim_id:
            claim = self.active["ledger"][claim_id]
            claim["state"] = response["state"]
            claim["turns"].append({"arousal": arousal})
        elif response.get("new_claim_text"):
            new_id = self._next_claim_id()
            self.active["ledger"][new_id] = {
                "text": response["new_claim_text"], "state": response["state"],
                "turns": [{"arousal": arousal}],
            }

        self.active["transcript"].append({"question": self.active["current_question"], "answer": answer_text})
        self.active["current_question"] = None
        self.active["current_claim_id"] = None

        confidence, flagged, cleared = fusion.update_confidence(self.active["ledger"], self.config)
        self.active["confidence"] = confidence
        self.active["flagged_claims"] = flagged
        self.active["cleared_claims"] = cleared
        self.active["confidence_history"].append(confidence)

        elapsed_minutes = (time.time() - self.active["interview_start_ts"]) / 60
        stop, reason = stoplogic.should_stop(
            self.active["ledger"], confidence, self.active["confidence_history"],
            self.active["question_count"], elapsed_minutes, operator_override=False, config=self.config,
        )
        self.active["stop_recommended"] = stop
        self.active["stop_reason"] = reason

    def generate_question(self):
        with self.lock:
            self._require_active()
            if self.active["phase"] != "interview":
                raise OrchestratorError(f"cannot generate a question during phase {self.active['phase']}")
            if self.active["recording"]:
                raise OrchestratorError("stop the current recording first")
            if self.active["stop_recommended"]:
                # Section 4: orchestrator is authority -- a stop condition already
                # fired, so no further question is generated (Interviewer output
                # would just be discarded anyway).
                raise OrchestratorError(
                    f"stop condition met ({self.active['stop_reason']}) -- use Stop Interview", status=409
                )

            history = llm.trim_history(self.active["transcript"], self.config["llm_context_turns"])
            prompt = llm.build_interviewer_prompt(
                INTERVIEWER_TEMPLATE, self.active["report_text"], self.active["ledger"], history
            )
            valid_ids = set(self.active["ledger"].keys())
            try:
                response = llm.call_agent(
                    self.config["ollama_host"], self.config["model"], prompt,
                    validate=lambda d: llm.validate_interviewer_response(d, valid_ids),
                    keep_alive=self.config["llm_keep_alive"], max_tokens=self.config["llm_max_tokens"],
                )
            except Exception as e:
                raise OrchestratorError(f"Interviewer call failed: {e}", status=502)

            self.active["question_count"] += 1
            self.active["current_question"] = response["question"]
            self.active["current_claim_id"] = response["claim_id"]
            self._save()

    def stop_interview(self):
        with self.lock:
            self._require_active()
            self.sensors.stop()
            self.vision.stop()
            self.active["phase"] = "stopped"
            self.active["recording"] = False
            self.active["current_question"] = None
            self._save()


def _baseline_from(service, start_ts, end_ts):
    from services.sensors import baseline_from_window  # pure fn, safe even if hardware unavailable

    return baseline_from_window(service.window(start_ts, end_ts))


SIMPLE_ROUTES = {
    "/calibrate/start": Orchestrator.calibrate_start,
    "/calibrate/stop": Orchestrator.calibrate_stop,
    "/record/start": Orchestrator.record_start,
    "/record/stop": Orchestrator.record_stop,
    "/generate-question": Orchestrator.generate_question,
    "/stop-interview": Orchestrator.stop_interview,
}

OPEN_SESSION_RE = re.compile(r"^/sessions/([^/]+)/open$")


def make_handler(orch):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def do_GET(self):
            if self.path == "/state":
                self._send_json(200, orch.get_state())
            elif self.path == "/sessions":
                self._send_json(200, orch.list_sessions())
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):
            try:
                if self.path == "/sessions":
                    pdf_bytes = self._read_body()
                    if not pdf_bytes:
                        self._send_json(400, {"error": "empty request body, expected PDF bytes"})
                        return
                    self._send_json(200, orch.create_session(pdf_bytes))
                    return

                if self.path == "/claims":
                    body = self._read_body()
                    try:
                        texts = json.loads(body)
                    except json.JSONDecodeError:
                        self._send_json(400, {"error": "expected a JSON list of claim strings"})
                        return
                    self._send_json(200, orch.add_claims(texts))
                    return

                match = OPEN_SESSION_RE.match(self.path)
                if match:
                    self._send_json(200, orch.open_session(match.group(1)))
                    return

                action = SIMPLE_ROUTES.get(self.path)
                if action is None:
                    self._send_json(404, {"error": "not found"})
                    return
                action(orch)
                self._send_json(200, orch.get_state())
            except OrchestratorError as e:
                self._send_json(e.status, {"error": str(e)})

        def log_message(self, fmt, *args):
            pass  # state changes already print in the Orchestrator methods above

    return Handler


def serve(host="localhost", port=8000):
    orch = Orchestrator()
    server = ThreadingHTTPServer((host, port), make_handler(orch))
    print(f"[orchestrator] API listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
