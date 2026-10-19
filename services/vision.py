"""Vision service: facial Action Unit intensity (microexpression signal) from the
webcam.

Two face models (`vision_model` in config.toml):
- "mediapipe" (default): MediaPipe Face Landmarker blendshapes, ~10 ms a frame on a
  laptop CPU, so every camera frame (~30 fps) is analysed. Micro-expressions last
  1/25 to 1/2 s; at the old ~5 fps (less on CPU, once py-feat's own time per frame
  is added) most of them fell between samples.
- "pyfeat": the original py-feat Detector. GPU by default (`vision_device`), falls
  back to CPU on failure -- same pattern as services/stt.py's `stt_device`.

Runs its own background thread doing a capture+detect loop (py-feat and
OpenCV are sync APIs), so the rest of the orchestrator just calls
start()/stop()/calibrate()/delta_sd() -- same shape as services/sensors.py,
and reuses its baseline/delta math directly since it's identical statistics.

Tracks a few tension-associated Action Units rather than the full FACS set
(design doc Section 9: coarse AUs, not full OpenFace-grade output). Gaze
aversion isn't included -- py-feat gives AU/emotion, not gaze/eye-tracking.

The baseline/delta-SD math below is identical to services/sensors.py's --
duplicated rather than imported cross-module (this repo has no
services/__init__.py, and each service file is meant to stand alone and run
standalone, same as stt.py/llm.py).
"""
import math
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2

AU_CHANNELS = ["AU04", "AU07", "AU23"]  # brow lowerer, lid tightener, lip pressor
MICRO_CHANNEL = "micro"  # delta_sd() key: SDs above baseline of the strongest micro-expression

# MediaPipe blendshapes closest to each AU (left/right averaged, which also halves the noise).
BLENDSHAPE_AUS = {
    "AU04": ("browDownLeft", "browDownRight"),
    "AU07": ("eyeSquintLeft", "eyeSquintRight"),
    "AU23": ("mouthPressLeft", "mouthPressRight"),
}
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _group_by_channel(window):
    grouped = {}
    for _t, ch, v in window:
        grouped.setdefault(ch, []).append(v)
    return grouped


def baseline_from_window(window):
    baseline = {}
    for ch, values in _group_by_channel(window).items():
        if len(values) >= 2:
            baseline[ch] = (statistics.mean(values), statistics.stdev(values))
    return baseline


def delta_sd_from_window(window, baseline):
    grouped = _group_by_channel(window)
    deltas = {}
    for ch, (mean, sd) in baseline.items():
        values = grouped.get(ch)
        deltas[ch] = None if not values or sd == 0 else (statistics.mean(values) - mean) / sd
    return deltas


def micro_events(window, baseline, blinks=(), z_on=4.0, z_off=1.0, max_sec=0.5, min_samples=2,
                 sd_floor=0.02, max_gap=0.25, blink_level=0.4, blink_pad=0.15):
    """Brief AU spikes in `window` [(t, channel, value)] against the calibration `baseline`
    {channel: (mean, sd)}: per channel, a run of samples at >= z_off SDs above baseline that
    peaks at >= z_on, lasts at most max_sec (longer is a macro expression, which the windowed
    mean in delta_sd_from_window already sees), spans >= min_samples frames (one-frame jitter
    isn't an expression), and both rises from (onset) and returns to (offset) baseline inside
    the window and without the face dropping out.

    sd_floor: a very still calibration gives a near-zero SD, and then landmark jitter alone
    reads as many SDs. blinks: [(t, eye-closure 0..1)]; a blink squeezes the lids, so an
    AU07 (lid tightener) run near one is dropped. Returns
    [{"channel", "onset", "apex", "offset", "peak_z"}] in time order."""
    blink_times = [t for t, v in blinks if v >= blink_level]
    events = []
    for ch, (mean, sd) in baseline.items():
        sd = max(sd, sd_floor)
        series = sorted((t, v) for t, c, v in window if c == ch)
        run = []

        def close(complete):
            if not run:
                return
            onset, offset = run[0][0], run[-1][0]
            apex_t, peak = max(run, key=lambda r: r[1])
            blinked = ch == "AU07" and any(onset - blink_pad <= b <= offset + blink_pad for b in blink_times)
            if complete and peak >= z_on and len(run) >= min_samples and offset - onset <= max_sec and not blinked:
                events.append({"channel": ch, "onset": onset, "apex": apex_t, "offset": offset, "peak_z": peak})
            run.clear()

        prev_t, onset_seen = None, False
        for t, v in series:
            if prev_t is not None and t - prev_t > max_gap:
                close(False)  # face lost mid-run: no offset seen
                onset_seen = False  # and the tracker resettles when it finds the face again
            z = (v - mean) / sd
            if z >= z_off:
                run.append((t, z))
            else:
                close(onset_seen)
                onset_seen = True
            prev_t = t
        close(False)  # still above baseline when the window ends
    return sorted(events, key=lambda e: e["onset"])


def micro_score(window, baseline, blinks=()):
    """delta_sd()'s "micro" channel: the strongest micro-expression's peak, in SDs above
    baseline (always >= z_on, so it reads as high arousal), 0.0 if none, None if no AU data."""
    if not baseline or not any(ch in baseline for _t, ch, _v in window):
        return None
    return max((e["peak_z"] for e in micro_events(window, baseline, blinks)), default=0.0)


# MSMF (OpenCV's Windows default) often fails to open webcams; DirectShow is reliable.
CAP_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def open_camera(index):
    return cv2.VideoCapture(index, CAP_BACKEND)


CAMERA_NOTES = {
    "black": "black image -- lens covered or privacy shutter closed?",
    "static": "frozen image -- a virtual camera (e.g. OBS) that isn't streaming?",
}


def frame_kind(frames):
    """frames: consecutive BGR frames from one camera -> "ok" | "black" | "static".
    A real sensor always has some pixel noise; a virtual camera's placeholder doesn't."""
    import numpy as np

    if max(int(f.max()) for f in frames) < 16:
        return "black"
    if len(frames) > 1 and all(np.array_equal(a, b) for a, b in zip(frames, frames[1:])):
        return "static"
    return "ok"


def probe_camera(index, n_frames=6):
    """None if the camera won't open or deliver frames, else its frame_kind()."""
    cap = open_camera(index)
    try:
        if not cap.isOpened():
            return None
        frames = []
        for _ in range(n_frames):
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
            time.sleep(0.05)
        return frame_kind(frames) if frames else None
    finally:
        cap.release()


def list_cameras(max_index=5):
    """[{"index": i, "kind": "ok" | "black" | "static"}] for every camera that opens."""
    found = []
    for i in range(max_index):
        kind = probe_camera(i)
        if kind is not None:
            found.append({"index": i, "kind": kind})
    return found


def pick_camera(cameras, preferred):
    """Keep `preferred` if it opens and isn't a frozen virtual camera; otherwise the first
    working camera, then a real-but-dark one, then anything. None if nothing opens."""
    by_index = {c["index"]: c["kind"] for c in cameras}
    if by_index.get(preferred) in ("ok", "black"):
        return preferred
    for kind in ("ok", "black", "static"):
        for c in cameras:
            if c["kind"] == kind:
                return c["index"]
    return None


def _make_detector(device):
    import feat

    if hasattr(feat, "Detector"):  # py-feat < 1.0
        return feat.Detector(device=device)
    # py-feat 2.x renamed it. AUs need only face + landmarks, so skip the heavy emotion/identity/gaze models.
    return feat.Detectorv1(device=device, emotion_model=None, identity_model=None, gaze_model=None)


def _load_on(device):
    try:
        return _make_detector(device), device
    except Exception:
        if device == "cuda":
            return _make_detector("cpu"), "cpu"
        raise


def load_detector(device="cuda"):
    """Cached weights first: py-feat otherwise asks the HF Hub for the newest revision on every
    start, and when upstream publishes new weights over a slow link the camera hangs on
    "starting face model" for as long as that download takes. Online only if nothing is cached."""
    try:
        import huggingface_hub.constants as hf
    except ImportError:
        return _load_on(device)
    was_offline, hf.HF_HUB_OFFLINE = hf.HF_HUB_OFFLINE, True
    try:
        return _load_on(device)
    except Exception as e:
        print(f"[vision] no cached face model ({type(e).__name__}); downloading", file=sys.stderr, flush=True)
    finally:
        hf.HF_HUB_OFFLINE = was_offline
    return _load_on(device)


class PyFeatFace:
    """frame -> {AU: intensity}, or None when no face is found."""

    def __init__(self, device):
        self.detector, self.device_used = load_detector(device)
        self._tmp = tempfile.TemporaryDirectory()
        self._frame_path = str(Path(self._tmp.name) / "frame.jpg")

    def analyse(self, frame, _ts):
        cv2.imwrite(self._frame_path, frame)
        if hasattr(self.detector, "detect_image"):
            result = self.detector.detect_image(self._frame_path)
        else:
            result = self.detector.detect([self._frame_path], data_type="image", progress_bar=False)
        aus = {au: float(result[au].iloc[0]) for au in AU_CHANNELS if au in result.columns}
        aus = {au: v for au, v in aus.items() if math.isfinite(v)}  # NaN when no face is found
        return aus or None


def face_landmarker_path():
    """The MediaPipe model file (3.6 MB), downloaded once into the user cache."""
    path = Path.home() / ".cache" / "veritas-ai" / "face_landmarker.task"
    if not path.exists():
        import urllib.request

        print("[vision] downloading MediaPipe face model", file=sys.stderr, flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)
        tmp.replace(path)
    return path


class MediaPipeFace:
    """frame -> {AU: intensity, "blink": eye closure}, or None when no face is found.
    VIDEO mode tracks the face between frames instead of re-detecting it each time."""

    device_used = "cpu"

    def __init__(self, model_path=None):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path or face_landmarker_path())),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def analyse(self, frame, ts):
        ms = max(int(ts * 1000), self._last_ms + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ms = ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(image, ms)
        if not result.face_blendshapes:
            return None
        scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
        out = {au: (scores[left] + scores[right]) / 2 for au, (left, right) in BLENDSHAPE_AUS.items()}
        out["blink"] = (scores["eyeBlinkLeft"] + scores["eyeBlinkRight"]) / 2
        return out


def load_face_model(model, device):
    """`vision_model` -> a loaded face model. MediaPipe falls back to py-feat if it isn't installed."""
    if model == "mediapipe":
        try:
            return MediaPipeFace()
        except ImportError:
            print("[vision] mediapipe not installed; using py-feat", file=sys.stderr, flush=True)
    return PyFeatFace(device)


class VisionService:
    def __init__(self, camera_index=0, device="cuda", model="mediapipe"):
        self.camera_index = camera_index
        self.requested_device = device
        self.model = model
        self.device_used = None
        self.error = None  # last failure reason, shown in the UI status panel
        self.samples = []  # [(timestamp, channel, value)]
        self.blinks = []  # [(timestamp, eye closure)] -- only to discount AU07 runs caused by blinks
        self.baseline = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._last_detect_error = None
        self.latest_jpeg = None  # newest camera frame, for the UI preview
        self.frame_note = None  # why the live frames are useless (black / frozen), shown in health

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_running():
            return  # already capturing (e.g. started at camera selection)
        self.error = None
        self.frame_note = None
        self._stop_event = threading.Event()  # fresh event so a lingering old thread can't be revived
        self._thread = threading.Thread(target=self._run, args=(self._stop_event,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)  # let it release the camera before anyone reopens it

    def _run(self, stop_event):
        cap = open_camera(self.camera_index)
        if not cap.isOpened():
            self.error = f"camera {self.camera_index} won't open -- pick another in the camera list"
            cap.release()
            return
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # analyse the newest frame, not a queued stale one

        # Load the (slow) face model in the background so the camera preview works immediately.
        loaded = {}

        def load():
            try:
                loaded["face"] = load_face_model(self.model, self.requested_device)
                self.device_used = loaded["face"].device_used
            except Exception as e:
                self.error = f"face detector failed to load: {type(e).__name__}: {e}"

        threading.Thread(target=load, daemon=True).start()
        try:
            recent, last_preview = [], 0.0
            while not stop_event.is_set():
                ok, frame = cap.read()  # blocks until the next frame, so this paces at the camera's rate
                ts = time.time()
                if not ok:
                    time.sleep(0.05)
                    continue
                if ts - last_preview >= 0.1:  # the preview and frozen-camera check don't need every frame
                    last_preview = ts
                    self.latest_jpeg = cv2.imencode(".jpg", frame)[1].tobytes()
                    recent = (recent + [frame])[-5:]
                    self.frame_note = CAMERA_NOTES.get(frame_kind(recent)) if len(recent) == 5 else None
                if "face" in loaded:
                    self._detect_frame(loaded["face"], frame, ts)
        finally:
            cap.release()
            self.latest_jpeg = None

    def _detect_frame(self, face, frame, ts):
        """ts: when the frame was captured, not when analysis finished."""
        try:
            values = face.analyse(frame, ts)
            if values is None:  # no face; recording NaN/zeros would poison the baseline stats
                return
            with self._lock:
                for au in AU_CHANNELS:
                    if au in values:
                        self.samples.append((ts, au, values[au]))
                if "blink" in values:
                    self.blinks.append((ts, values["blink"]))
        except Exception as e:
            key = f"{type(e).__name__}: {e}"
            if key != self._last_detect_error:  # log each distinct failure once, not every frame
                self._last_detect_error = key
                print(f"[vision] frame detection failed: {key}", file=sys.stderr, flush=True)
            # skip the frame, keep going

    def window(self, start_ts, end_ts):
        with self._lock:
            return [(t, ch, v) for t, ch, v in self.samples if start_ts <= t <= end_ts]

    def calibrate(self, duration_sec):
        start = time.time()
        time.sleep(duration_sec)
        self.baseline = baseline_from_window(self.window(start, time.time()))
        return self.baseline

    def delta_sd(self, start_ts, end_ts):
        """Per-AU shift of the mean over the answer (sustained tension) plus "micro": the
        strongest brief spike, which a mean over a whole answer averages away."""
        window = self.window(start_ts, end_ts)
        with self._lock:
            blinks = [(t, v) for t, v in self.blinks if start_ts <= t <= end_ts]
        deltas = delta_sd_from_window(window, self.baseline)
        deltas[MICRO_CHANNEL] = micro_score(window, self.baseline, blinks)
        return deltas


def _selfcheck():
    baseline_window = [(i, "AU04", 0.5 + 0.01 * (i % 3)) for i in range(10)]
    baseline = baseline_from_window(baseline_window)
    mean, sd = baseline["AU04"]
    assert sd > 0

    spiked = [(i, "AU04", mean + 3 * sd) for i in range(10, 15)]
    deltas = delta_sd_from_window(spiked, baseline)
    assert abs(deltas["AU04"] - 3.0) < 1e-9, deltas

    assert delta_sd_from_window([], baseline) == {"AU04": None}

    import numpy as np

    rng = np.random.default_rng(0)
    noisy = [rng.integers(0, 255, (4, 4, 3), dtype="uint8") for _ in range(3)]
    assert frame_kind(noisy) == "ok"
    assert frame_kind([np.full((4, 4, 3), 3, dtype="uint8")] * 3) == "black"
    assert frame_kind([np.full((4, 4, 3), 120, dtype="uint8")] * 3) == "static"

    cams = [{"index": 0, "kind": "black"}, {"index": 1, "kind": "static"}, {"index": 2, "kind": "ok"}]
    assert pick_camera(cams, 7) == 2  # stale/unknown index -> first working camera
    assert pick_camera(cams, 0) == 0  # a real camera with its shutter closed is kept
    assert pick_camera(cams, 1) == 2  # frozen virtual camera is skipped
    assert pick_camera(cams[:2], 7) == 0
    assert pick_camera([], 0) is None

    # micro-expressions: a 100 ms spike at 30 fps is caught; a long (macro) rise, a one-frame
    # glitch and a run still open at the end of the window are not
    fps, calm = 30, {"AU04": (0.1, 0.05)}
    flat = [(i / fps, "AU04", 0.1) for i in range(90)]
    spike = [(t, ch, 0.4 if 1.0 <= t < 1.1 else v) for t, ch, v in flat]
    (event,) = micro_events(spike, calm)
    assert event["channel"] == "AU04" and abs(event["peak_z"] - 6.0) < 1e-9 and 0.99 < event["onset"] < 1.01, event
    assert micro_score(spike, calm) == event["peak_z"]
    assert micro_score(flat, calm) == 0.0 and micro_score([], calm) is None
    macro = [(t, ch, 0.4 if 1.0 <= t < 2.0 else v) for t, ch, v in flat]
    assert micro_events(macro, calm) == []
    glitch = [(t, ch, 0.4 if i == 30 else v) for i, (t, ch, v) in enumerate(flat)]
    assert micro_events(glitch, calm) == []
    assert micro_events([(t, ch, 0.4 if t >= 2.9 else v) for t, ch, v in flat], calm) == []  # no offset
    assert micro_events([(t, ch, 0.4 if t < 0.1 else v) for t, ch, v in flat], calm) == []  # no onset
    regained = [s for s in spike if not 0.5 < s[0] < 1.0]  # face lost just before the spike
    assert micro_events(regained, calm) == []
    squint = [(t, "AU07", v) for t, _ch, v in spike]
    assert micro_events(squint, {"AU07": (0.1, 0.05)}, blinks=[(1.05, 0.9)]) == []  # a blink, not AU07
    assert len(micro_events(squint, {"AU07": (0.1, 0.05)}, blinks=[(2.5, 0.9)])) == 1

    # a face-less frame (NaN AUs) is skipped, not recorded
    class FakeResult:
        columns = AU_CHANNELS

        def __getitem__(self, _au):
            import pandas as pd

            return pd.Series([float("nan")])

    class FakeDetector:
        def detect(self, *_a, **_k):
            return FakeResult()

    face = PyFeatFace.__new__(PyFeatFace)
    face.detector, face._tmp = FakeDetector(), tempfile.TemporaryDirectory()
    face._frame_path = str(Path(face._tmp.name) / "f.jpg")
    svc = VisionService.__new__(VisionService)
    svc.samples, svc.blinks, svc._lock, svc._last_detect_error = [], [], threading.Lock(), None
    svc._detect_frame(face, np.zeros((8, 8, 3), dtype="uint8"), 0.0)
    assert svc.samples == [], svc.samples
    print("vision.py self-check passed (baseline/delta/micro-expression math only -- no camera or face model)")


if __name__ == "__main__":
    _selfcheck()
