"""Vision service: py-feat AU intensity (microexpression signal) from the
webcam. GPU by default (`vision_device`), falls back to CPU on failure --
same pattern as services/stt.py's `stt_device`.

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
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2

AU_CHANNELS = ["AU04", "AU07", "AU23"]  # brow lowerer, lid tightener, lip pressor


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


# MSMF (OpenCV's Windows default) often fails to open webcams; DirectShow is reliable.
CAP_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def open_camera(index):
    return cv2.VideoCapture(index, CAP_BACKEND)


def list_cameras(max_index=5):
    found = []
    for i in range(max_index):
        cap = open_camera(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


def load_detector(device="cuda"):
    from feat import Detector

    try:
        return Detector(device=device), device
    except Exception:
        if device == "cuda":
            return Detector(device="cpu"), "cpu"
        raise


class VisionService:
    def __init__(self, camera_index=0, device="cuda"):
        self.camera_index = camera_index
        self.requested_device = device
        self.device_used = None
        self.error = None  # last failure reason, shown in the UI status panel
        self.samples = []  # [(timestamp, channel, value)]
        self.baseline = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_running():
            return  # already capturing (e.g. started at camera selection)
        self.error = None
        self._stop_event = threading.Event()  # fresh event so a lingering old thread can't be revived
        self._thread = threading.Thread(target=self._run, args=(self._stop_event,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)  # let it release the camera before anyone reopens it

    def _run(self, stop_event):
        try:
            detector, self.device_used = load_detector(self.requested_device)
        except Exception as e:
            self.error = f"face detector failed to load: {type(e).__name__}: {e}"
            return  # no vision this session -- channels stay unavailable

        cap = open_camera(self.camera_index)
        if not cap.isOpened():
            self.error = f"camera index {self.camera_index} won't open"
            cap.release()
            return
        try:
            with tempfile.TemporaryDirectory() as tmp:
                frame_path = str(Path(tmp) / "frame.jpg")
                while not stop_event.is_set():
                    ok, frame = cap.read()
                    if ok:
                        self._detect_frame(detector, frame, frame_path)
                    time.sleep(0.2)  # ~5 fps -- AU intensity doesn't need more
        finally:
            cap.release()

    def _detect_frame(self, detector, frame, frame_path):
        try:
            cv2.imwrite(frame_path, frame)
            result = detector.detect_image(frame_path)
            for au in AU_CHANNELS:
                if au in result.columns:
                    self._record(au, float(result[au].iloc[0]))
        except Exception:
            pass  # skip a bad/no-face frame, keep going

    def _record(self, channel, value):
        with self._lock:
            self.samples.append((time.time(), channel, value))

    def window(self, start_ts, end_ts):
        with self._lock:
            return [(t, ch, v) for t, ch, v in self.samples if start_ts <= t <= end_ts]

    def calibrate(self, duration_sec):
        start = time.time()
        time.sleep(duration_sec)
        self.baseline = baseline_from_window(self.window(start, time.time()))
        return self.baseline

    def delta_sd(self, start_ts, end_ts):
        return delta_sd_from_window(self.window(start_ts, end_ts), self.baseline)


def _selfcheck():
    baseline_window = [(i, "AU04", 0.5 + 0.01 * (i % 3)) for i in range(10)]
    baseline = baseline_from_window(baseline_window)
    mean, sd = baseline["AU04"]
    assert sd > 0

    spiked = [(i, "AU04", mean + 3 * sd) for i in range(10, 15)]
    deltas = delta_sd_from_window(spiked, baseline)
    assert abs(deltas["AU04"] - 3.0) < 1e-9, deltas

    assert delta_sd_from_window([], baseline) == {"AU04": None}
    print("vision.py self-check passed (baseline/delta math only -- no camera/py-feat here)")


if __name__ == "__main__":
    _selfcheck()
