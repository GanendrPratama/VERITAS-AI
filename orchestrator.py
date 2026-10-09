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
import faulthandler
import json
import os
import re
import shutil
import sys
import threading
import time
import tomllib
import traceback
os.environ.setdefault("TQDM_DISABLE", "1")  # py-feat's per-frame progress bars flood the log and bury real errors
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import fusion
import stoplogic
from services import llm, stt
from services.docconvert import pdf_to_markdown

faulthandler.enable()  # a native crash (torch/xgboost/CUDA) now dumps a Python traceback to the .err log

ROOT = Path(__file__).resolve().parent
LOGS_DIR = ROOT / "logs"
CONFIG = tomllib.loads((ROOT / "config.toml").read_text())
CLAIMS_TEMPLATE = (ROOT / "agents" / "claims_prompt.md").read_text()
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
        self.lock = threading.Lock()  # guards self.active; never held across an LLM or STT call
        self._busy = None  # the slow turn action (LLM/STT) running outside the lock, if any
        self._pending_answer = None  # (session_id, answer, unclear) kept when the Analyst fails, for a retry
        self.sensors = _build_sensors()
        self.vision = _build_vision()
        self._camera_list = []
        self.recorder = stt.Recorder(samplerate=config["mic_samplerate"])
        self._whisper_model = None
        self._stt_device = config["stt_device"]
        self._stt_lock = threading.Lock()  # one transcription at a time: live captions vs the final pass
        self._caption_stop = threading.Event()
        self.stt_status = {"state": "idle", "device": None, "last_text": None, "partial": "", "error": None}
        self._resume()

    # -- persistence --

    def _session_path(self, session_id):
        return self.logs_dir / session_id / "state.json"

    def _save(self):
        path = self._session_path(self.active["session_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.active))
        self._remember_active()

    def _remember_active(self):
        # Pointer so a restarted orchestrator (see scripts/supervise.py) picks the session back up.
        (self.logs_dir / "active_session").write_text(self.active["session_id"])

    def _resume(self):
        try:
            session_id = (self.logs_dir / "active_session").read_text().strip()
            state = json.loads(self._session_path(session_id).read_text())
        except (OSError, json.JSONDecodeError):
            return
        if state.get("phase") in (None, "stopped"):
            return
        state["recording"] = False  # the mic capture died with the old process
        self.active = state
        if state["phase"] != "idle":
            self.sensors.start()
            self.vision.start()
        # ponytail: calibration baselines live in memory only, so an interview resumed
        # after a crash has no arousal signal; persist them if that matters.
        print(f"[orchestrator] resumed session {session_id} in phase {state['phase']}")

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

    def _release_capture(self):
        """Drop any in-progress mic capture and live captions (session ending or switching)."""
        self._caption_stop.set()
        if self.recorder.is_recording():
            self.recorder.cancel()

    def _switch_session(self, state):
        """Make `state` the active session, moving capture hardware over from the old one --
        otherwise the old session's mic recording and camera/BLE capture keep running."""
        self._release_capture()
        old_live = self.active is not None and self.active["phase"] in ("calibration", "interview")
        new_live = state["phase"] in ("calibration", "interview")
        if new_live:
            self.sensors.start()
            self.vision.start()
        elif old_live:
            self.sensors.stop()
            self.vision.stop()
        state["recording"] = False  # whatever capture it had belonged to another session or process
        self.active = state

    def _require_active(self):
        if self.active is None:
            raise OrchestratorError("no active session -- create or open one first")

    def _whisper(self):
        if self._whisper_model is None:
            self.stt_status.update(state="loading model", device=self._stt_device)
            self._whisper_model = stt.load_model(self.config["stt_model"], device=self._stt_device)
        return self._whisper_model

    # -- session lifecycle --

    def create_session(self, pdf_bytes):
        with self.lock:
            try:
                report_text = pdf_to_markdown(pdf_bytes)
            except Exception as e:
                raise OrchestratorError(f"couldn't read PDF: {e}", status=400)
            self._switch_session(new_session_state(self._new_session_id(), report_text))
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
            self._switch_session(json.loads(path.read_text()))
            self._remember_active()
            return dict(self.active)

    def delete_session(self, session_id):
        with self.lock:
            path = self._session_path(session_id)
            # only ever delete a directory that really is a session under logs/
            if not re.fullmatch(r"[\w-]+", session_id) or not path.exists():
                raise OrchestratorError(f"no such session: {session_id}", status=404)
            if self.active and self.active["session_id"] == session_id:
                self._release_capture()
                self.sensors.stop()
                self.vision.stop()
                self.active = None
            shutil.rmtree(path.parent)

    def get_state(self):
        with self.lock:
            if self.active is None:
                return {"active": False}
            return dict(self.active)

    # -- component health (for the UI status panel) --

    def health(self):
        import importlib.util

        def lib(name):
            return importlib.util.find_spec(name) is not None

        with self.lock:
            # a stopped session's capture is off too -- don't report its last errors as live
            live = self.active is not None and self.active["phase"] in ("calibration", "interview")
        drop = self.config["sensor_dropout_sec"]

        def stream(svc, missing, idle, extra=None):
            if isinstance(svc, _NullHardwareService):
                return "down", missing
            running = getattr(svc, "is_running", lambda: False)()
            if not live and not running:
                return "idle", idle
            if getattr(svc, "error", None):
                return "down", svc.error
            if extra == "no_ble":
                return "down", f"{self.config['ble_device_name']} not found over BLE"
            if running and getattr(svc, "frame_note", None):
                return "down", f"camera shows a {svc.frame_note}"
            samples = getattr(svc, "samples", [])
            if not samples:
                if running and getattr(svc, "device_used", None):  # face model loaded, frames analysed
                    return "down", "camera on, but no face detected -- check the preview shows the subject"
                return ("idle", "starting camera / face model...") if running else ("down", "no data received yet")
            age = time.time() - samples[-1][0]
            return ("ok", "streaming") if age <= drop else ("down", f"no data for {age:.0f}s")

        ble_ok = getattr(self.sensors, "is_connected", lambda: True)()
        esp = stream(self.sensors, "bleak not installed", "connects when calibration starts",
                     None if ble_ok else "no_ble")
        cam = stream(self.vision, "py-feat / opencv not installed", "starts with calibration")

        try:
            import sounddevice as sd

            mic = ("ok", sd.query_devices(self.recorder.device, kind="input")["name"])  # the selected mic
        except Exception as e:
            mic = ("down", f"no input device ({e})")

        try:
            import requests

            requests.get(f"{self.config['ollama_host']}/api/tags", timeout=1).raise_for_status()
            llm_ = ("ok", self.config["model"])
        except Exception:
            llm_ = ("down", f"Ollama unreachable at {self.config['ollama_host']}")

        stt_ = ("ok", self.config["stt_model"]) if lib("faster_whisper") else ("down", "faster-whisper not installed")
        return [
            {"name": n, "status": st, "detail": d}
            for n, (st, d) in [("ESP32 sensor (HR/GSR)", esp), ("Camera / face", cam),
                               ("Microphone", mic), ("Speech-to-text", stt_), ("LLM (Ollama)", llm_)]
        ]

    def cameras(self):
        # Probing opens and releases every camera (the webcam light blinks), and the dashboard asks
        # on every rerun -- so probe once and reuse the list; set_camera/calibration re-probe as needed.
        # Probing a device the capture thread holds would fail anyway.
        if not self._camera_list and not getattr(self.vision, "is_running", lambda: False)():
            self._pick_camera()
        try:
            from services.vision import CAMERA_NOTES
        except ImportError:
            CAMERA_NOTES = {}
        return {
            "cameras": [c["index"] for c in self._camera_list],
            "notes": {str(c["index"]): CAMERA_NOTES.get(c["kind"]) for c in self._camera_list},
            "selected": getattr(self.vision, "camera_index", None),
        }

    def _pick_camera(self):
        """Re-probe the cameras and, while capture is off, move off a configured/stale index that
        doesn't open (or is a frozen virtual camera) onto the best real one."""
        try:
            from services.vision import list_cameras, pick_camera
        except ImportError:
            return
        self._camera_list = list_cameras()
        if hasattr(self.vision, "camera_index") and not self.vision.is_running():
            best = pick_camera(self._camera_list, self.vision.camera_index)
            if best is not None and best != self.vision.camera_index:
                print(f"[orchestrator] camera {self.vision.camera_index} unusable; using camera {best}")
                self.vision.camera_index = best

    def live(self):
        """Latest readings from every stream. Lock-free: it only reads the services'
        sample buffers, never self.active."""

        def latest(svc):
            out = {}
            for t, ch, v in list(getattr(svc, "samples", []))[-90:]:
                out[ch] = {"value": v, "age": round(time.time() - t, 1)}
            return out

        try:
            self.recorder.open()
        except Exception:
            pass  # recorder.error carries the reason
        r = self.recorder
        return {
            "mic": {"level": r.level, "device": r.device_name, "error": r.error, "recording": r.is_recording()},
            "stt": dict(self.stt_status),
            "sensors": latest(self.sensors),
            "vision": {"channels": latest(self.vision), "device": getattr(self.vision, "device_used", None),
                       "running": getattr(self.vision, "is_running", lambda: False)(),
                       "error": getattr(self.vision, "error", None)},
        }

    def camera_frame(self):
        return getattr(self.vision, "latest_jpeg", None)

    def set_camera(self, index):
        with self.lock:
            if self.active is not None and self.active["phase"] in ("calibration", "interview"):
                raise OrchestratorError("camera can't be changed during an interview")
            if hasattr(self.vision, "camera_index"):
                from services.vision import probe_camera

                self.vision.stop()  # release the device so it can be probed
                if probe_camera(index) is None:
                    self.vision.start()  # keep the old camera running
                    raise OrchestratorError(f"camera {index} won't open", status=400)
                self.vision.camera_index = index
                self.vision.start()  # open the webcam now so problems show before calibration

    def tail_logs(self, which="orchestrator", lines=200):
        if which not in ("orchestrator", "dashboard"):
            raise OrchestratorError(f"unknown log: {which}", status=404)
        out = []
        for name in (f"{which}.log", f"{which}.err.log"):
            path = self.logs_dir / name
            if path.is_file():
                out.append(f"== {name} ==")
                out += path.read_text(errors="replace").splitlines()[-lines:]
        return {"text": "\n".join(out) or "(no log output yet)"}

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

    def extract_claims(self):
        """Suggest claims from the report via the LLM. Returns them without saving, so the operator can edit first."""
        with self.lock:
            self._require_active()
            report, existing = self.active["report_text"], {c["text"] for c in self.active["ledger"].values()}
        try:
            data = llm.call_agent(
                self.config["ollama_host"], self.config["model"], CLAIMS_TEMPLATE.replace("<<REPORT>>", report),
                validate=llm.validate_claims_response, keep_alive=self.config["llm_keep_alive"], max_tokens=768,
            )
        except (llm.InvalidAgentJSON, OSError) as e:  # OSError covers requests' connection errors
            raise OrchestratorError(f"couldn't extract claims ({type(e).__name__}: {e}) -- is Ollama running?", status=502)
        return {"claims": [c.strip() for c in data["claims"] if c.strip() and c.strip() not in existing]}

    # -- microphones and compute devices --

    def devices(self):
        try:
            mics = [{"index": i, "name": n} for i, n in stt.list_inputs()]
            import sounddevice as sd

            selected = self.recorder.device if self.recorder.device is not None else sd.query_devices(kind="input")["index"]
        except Exception:
            mics, selected = [], None
        return {
            "mics": mics, "mic_selected": selected,
            "stt_device": self._stt_device, "vision_device": getattr(self.vision, "requested_device", None),
        }

    def set_devices(self, changes):
        with self.lock:
            if self.active and self.active["recording"]:
                raise OrchestratorError("can't change devices while recording")
            if "mic" in changes:
                try:
                    self.recorder.set_device(changes["mic"])
                except Exception as e:
                    raise OrchestratorError(f"couldn't switch microphone: {e}")
            if "stt_device" in changes:
                self._set_compute("stt", changes["stt_device"])
            if "vision_device" in changes:
                self._set_compute("vision", changes["vision_device"])

    def _set_compute(self, which, device):
        if device not in ("cuda", "cpu"):
            raise OrchestratorError(f"unknown device: {device!r}", status=400)
        if which == "stt":
            with self._stt_lock:  # don't pull the model out from under a running transcription
                self._stt_device, self._whisper_model = device, None
        elif hasattr(self.vision, "requested_device"):
            self.vision.stop()
            self.vision.requested_device = device
            self.vision.start()

    # -- turn loop actions, scoped to self.active --

    def calibrate_start(self):
        with self.lock:
            self._require_active()
            if self.active["phase"] != "idle":
                raise OrchestratorError(f"cannot start calibration from phase {self.active['phase']}")
            self.sensors.start()
            if not getattr(self.vision, "is_running", lambda: True)():
                self._pick_camera()  # don't calibrate on a camera index that won't open
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
            self._require_idle()
            self._pending_answer = None
            try:
                self.recorder.start()
                self._start_captions()
            except Exception as e:
                print(f"[orchestrator] mic recording unavailable: {e}")
            self.active["recording"] = True
            self.active["record_start_ts"] = time.time()
            self._save()

    def _require_idle(self):
        if self._busy:
            raise OrchestratorError(f"still working on {self._busy} -- wait for it to finish")

    def _begin_slow(self, name):
        """Call with self.lock held. Marks a slow action as running so the lock can be
        released for its LLM/STT call while other turn actions are refused."""
        self._require_idle()
        self._busy = name
        return self.active

    def _still_current(self, session):
        """Call with self.lock held, after a slow call: the session it started on is still
        the active one (not replaced by create/open/delete, or stopped, meanwhile)."""
        return self.active is session and session["phase"] != "stopped"

    def record_stop(self, typed_answer=None):
        """typed_answer: operator-typed text used instead of mic+STT (partial setups).

        self.lock is only held to read and to apply state -- STT and the Analyst call run
        without it, so /state, /health and the dashboard keep answering meanwhile."""
        with self.lock:
            self._require_active()
            if not self.active["recording"]:
                raise OrchestratorError("not recording")
            session = self._begin_slow("the answer")
            self._caption_stop.set()
            wav_path = self._session_path(session["session_id"]).parent / f"turn_{session['question_count']}.wav"
            pending = self._pending_answer if self._pending_answer and self._pending_answer[0] == session["session_id"] else None
            assess = session["phase"] == "interview"
            ctx = self._analyst_context(session) if assess else None
        try:
            if typed_answer:
                try:
                    self.recorder.stop(str(wav_path))  # release the mic if it did start
                except Exception:
                    pass
                answer_text, unclear = typed_answer, False
            elif pending:
                _, answer_text, unclear = pending  # the Analyst failed last time; the mic already stopped
            else:
                # raises (staying in "recording") if there's no audio, so the operator can retry typed
                answer_text, unclear = self._capture_answer(wav_path)
            if assess:
                try:
                    response = self._call_analyst(ctx, answer_text, unclear)
                except OrchestratorError:
                    self._pending_answer = (session["session_id"], answer_text, unclear)
                    raise
            with self.lock:
                if not self._still_current(session):
                    raise OrchestratorError("the session changed while the answer was processed -- answer discarded")
                session["recording"] = False
                self._pending_answer = None
                if assess:
                    self._apply_assessment(ctx, response, answer_text)
                self._save()
        finally:
            self._busy = None

    def _capture_answer(self, wav_path):
        try:
            self.recorder.stop(str(wav_path))
        except Exception as e:
            print(f"[orchestrator] mic recording unavailable: {e}")
            raise OrchestratorError("no audio captured (mic unavailable) -- type the answer and press Stop again")
        self.stt_status.update(state="transcribing", error=None)
        with self._stt_lock:
            result = self._transcribe(str(wav_path))
        self.stt_status.update(state="ready", last_text=result["text"], partial="")
        return result["text"], result["unclear"]

    def _transcribe(self, audio, **options):
        """Whisper call that drops from GPU to CPU once if CUDA libs (e.g. cublas64_12.dll) fail at first use."""
        for attempt in range(2):
            try:
                result = stt.transcribe(self._whisper(), audio, self.config["stt_min_logprob"],
                                        language=self.config.get("stt_language") or None, **options)
                self.stt_status["device"] = self._stt_device
                return result
            except Exception as e:
                print(f"[orchestrator] STT failed on {self._stt_device}: {e}")
                self.stt_status.update(state="failed", error=str(e))
                if attempt == 0 and self._stt_device != "cpu":
                    self._stt_device, self._whisper_model = "cpu", None
                    self.stt_status.update(state="retrying on CPU")
                    continue
                raise OrchestratorError("speech-to-text unavailable -- type the answer and press Stop again")

    # -- live captions: re-transcribe the audio so far while the operator records --

    def _start_captions(self):
        if self.recorder.samplerate != 16000:  # the model takes raw arrays only at 16 kHz
            return
        self._caption_stop = threading.Event()
        self.stt_status["partial"] = ""
        threading.Thread(target=self._caption_loop, args=(self._caption_stop,), daemon=True).start()

    def _caption_loop(self, stop):
        # ponytail: re-decodes the last 30s each tick (fine for short answers); real streaming
        # decode only if CPU can't keep up.
        while not stop.wait(1.0):
            audio = self.recorder.snapshot()
            if audio is None or len(audio) < 16000:
                continue
            try:
                with self._stt_lock:
                    if stop.is_set():
                        return
                    text = self._transcribe(audio[-30 * 16000:], beam_size=1, condition_on_previous_text=False)["text"]
            except OrchestratorError:
                return  # STT is broken; the final pass will report it
            if not stop.is_set():
                self.stt_status["partial"] = text

    def _analyst_context(self, session):
        """Everything the Analyst call needs, snapshotted while self.lock is held."""
        start_ts, end_ts = session["record_start_ts"], time.time()
        deltas = {}
        deltas.update(self.sensors.delta_sd(start_ts, end_ts))
        deltas.update(self.vision.delta_sd(start_ts, end_ts))
        claim_id = session["current_claim_id"]
        return {
            "session": session,
            "arousal": fusion.bucket_arousal(deltas, self.config),
            "claim_id": claim_id,
            "claim_text": session["ledger"][claim_id]["text"] if claim_id in session["ledger"] else None,
            "history": llm.trim_history(list(session["transcript"]), self.config["llm_context_turns"]),
            "report_text": session["report_text"],
            "question": session["current_question"],
        }

    def _call_analyst(self, ctx, answer_text, unclear):
        if unclear:
            return {"state": "evasive", "plausibility": "non_answer",
                    "reasoning": "answer unclear (STT)", "new_claim_text": None}
        prompt = llm.build_analyst_prompt(
            ANALYST_TEMPLATE, ctx["report_text"], ctx["claim_text"], ctx["history"], answer_text, ctx["arousal"],
            question=ctx["question"],
        )
        try:
            return llm.call_agent(
                self.config["ollama_host"], self.config["model"], prompt,
                validate=llm.validate_analyst_response,
                keep_alive=self.config["llm_keep_alive"], max_tokens=self.config["llm_max_tokens"],
            )
        except Exception as e:
            raise OrchestratorError(f"Analyst call failed: {e} -- press Stop again to retry", status=502)

    def _apply_assessment(self, ctx, response, answer_text):
        """Call with self.lock held."""
        active, arousal, claim_id = ctx["session"], ctx["arousal"], ctx["claim_id"]
        if claim_id in active["ledger"]:
            claim = active["ledger"][claim_id]
            claim["state"] = response["state"]
            claim["turns"].append({"arousal": arousal})
        elif response.get("new_claim_text"):
            new_id = self._next_claim_id()
            active["ledger"][new_id] = {
                "text": response["new_claim_text"], "state": response["state"],
                "turns": [{"arousal": arousal}],
            }

        active["transcript"].append({"question": ctx["question"], "answer": answer_text})
        active["current_question"] = None
        active["current_claim_id"] = None

        confidence, flagged, cleared = fusion.update_confidence(active["ledger"], self.config)
        active["confidence"] = confidence
        active["flagged_claims"] = flagged
        active["cleared_claims"] = cleared
        active["confidence_history"].append(confidence)

        elapsed_minutes = (time.time() - active["interview_start_ts"]) / 60
        stop, reason = stoplogic.should_stop(
            active["ledger"], confidence, active["confidence_history"],
            active["question_count"], elapsed_minutes, operator_override=False, config=self.config,
        )
        active["stop_recommended"] = stop
        active["stop_reason"] = reason

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
            session = self._begin_slow("the next question")
            history = llm.trim_history(list(session["transcript"]), self.config["llm_context_turns"])
            prompt = llm.build_interviewer_prompt(
                INTERVIEWER_TEMPLATE, session["report_text"], session["ledger"], history
            )
            valid_ids = set(session["ledger"].keys())
        try:  # self.lock is released for the LLM call, which can take many seconds
            try:
                response = llm.call_agent(
                    self.config["ollama_host"], self.config["model"], prompt,
                    validate=lambda d: llm.validate_interviewer_response(d, valid_ids),
                    keep_alive=self.config["llm_keep_alive"], max_tokens=self.config["llm_max_tokens"],
                )
            except Exception as e:
                raise OrchestratorError(f"Interviewer call failed: {e}", status=502)
            with self.lock:
                if not self._still_current(session):
                    raise OrchestratorError("the session changed while the question was generated -- question discarded")
                session["question_count"] += 1
                session["current_question"] = response["question"]
                session["current_claim_id"] = response["claim_id"]
                self._save()
        finally:
            self._busy = None

    def stop_interview(self):
        with self.lock:
            self._require_active()
            self._release_capture()  # stopping mid-answer must not leave the mic recording
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
            try:
                self.wfile.write(body)
            except ConnectionError:
                pass  # client gave up waiting (e.g. UI 5s timeout during a long STT call)

        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            return self.rfile.read(length) if length else b""

        def _get(self):
            if self.path == "/ping":  # lock-free liveness check for scripts/supervise.py
                self._send_json(200, {"ok": True})
            elif self.path == "/state":
                self._send_json(200, orch.get_state())
            elif self.path == "/health":
                self._send_json(200, orch.health())
            elif self.path.startswith("/logs"):
                try:
                    self._send_json(200, orch.tail_logs(self.path[len("/logs/"):] or "orchestrator"))
                except OrchestratorError as e:
                    self._send_json(e.status, {"error": str(e)})
            elif self.path == "/devices":
                self._send_json(200, orch.devices())
            elif self.path == "/live":
                self._send_json(200, orch.live())
            elif self.path == "/camera/frame":
                jpeg = orch.camera_frame()
                if jpeg is None:
                    self._send_json(404, {"error": "no camera frame"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            elif self.path == "/cameras":
                self._send_json(200, orch.cameras())
            elif self.path == "/sessions":
                self._send_json(200, orch.list_sessions())
            else:
                self._send_json(404, {"error": "not found"})

        def _delete(self):
            match = re.fullmatch(r"/sessions/([^/]+)", self.path)
            if not match:
                self._send_json(404, {"error": "not found"})
                return
            try:
                orch.delete_session(match.group(1))
                self._send_json(200, {"deleted": match.group(1)})
            except OrchestratorError as e:
                self._send_json(e.status, {"error": str(e)})

        def _post(self):
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
                        texts = None
                    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                        self._send_json(400, {"error": "expected a JSON list of claim strings"})
                        return
                    self._send_json(200, orch.add_claims(texts))
                    return

                if self.path == "/claims/extract":
                    self._send_json(200, orch.extract_claims())
                    return

                if self.path == "/devices":
                    try:
                        changes = json.loads(self._read_body())
                    except json.JSONDecodeError:
                        self._send_json(400, {"error": "expected a JSON object"})
                        return
                    orch.set_devices(changes)
                    self._send_json(200, orch.devices())
                    return

                if self.path == "/camera":
                    try:
                        index = int(self._read_body())
                    except ValueError:
                        self._send_json(400, {"error": "expected a camera index"})
                        return
                    orch.set_camera(index)
                    self._send_json(200, orch.cameras())
                    return

                if self.path == "/record/stop":
                    orch.record_stop(self._read_body().decode("utf-8").strip() or None)
                    self._send_json(200, orch.get_state())
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

        def _guarded(self, handler):
            """An unexpected exception is logged and answered as a JSON 500 instead of
            dropping the connection, so the UI can show the reason."""
            try:
                handler()
            except Exception as e:
                traceback.print_exc()
                self._send_json(500, {"error": f"internal error: {type(e).__name__}: {e}"})

        def do_GET(self):
            self._guarded(self._get)

        def do_POST(self):
            self._guarded(self._post)

        def do_DELETE(self):
            self._guarded(self._delete)

        def log_message(self, fmt, *args):
            pass  # state changes already print in the Orchestrator methods above

    return Handler


class Server(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a second orchestrator bind the same port alongside a running
    # one (requests then split between two processes with different state); fail to bind instead.
    allow_reuse_address = sys.platform != "win32"


def _preload_native_libs():
    """Import torch/py-feat and sounddevice on the main thread before serving. Imported lazily
    from the vision thread while a /health request imported sounddevice, torch's DLL init failed
    (WinError 1114) and the retry crashed the process with an access violation."""
    for name in ("sounddevice", "feat"):
        try:
            __import__(name)
        except Exception as e:
            print(f"[orchestrator] preloading {name} failed: {type(e).__name__}: {e}")


def serve(host="127.0.0.1", port=8000):
    _preload_native_libs()
    orch = Orchestrator()
    server = Server((host, port), make_handler(orch))
    print(f"[orchestrator] API listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    serve()
