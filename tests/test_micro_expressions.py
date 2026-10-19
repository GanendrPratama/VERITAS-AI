"""Micro-expressions end to end through VisionService, with a scripted face model instead of a
webcam: a 100 ms brow-lowerer spike during a 20 s answer is reported in the "micro" channel and
makes the turn's arousal high, while the answer-long mean (all the old loop had) barely moves."""
import math
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fusion
from services import vision

CFG = {"arousal_elevated_sd": 1.0, "arousal_high_sd": 2.0}
FPS = 30


class ScriptedFace:
    """AU04 follows `brow(t)`; small aperiodic jitter on every channel, eyes open."""

    def __init__(self, brow):
        self.brow = brow

    def analyse(self, _frame, ts):
        jitter = 0.03 * math.sin(round(ts * FPS) * 2.4)
        return {"AU04": self.brow(ts) + jitter, "AU07": 0.3 + jitter, "AU23": 0.1 + jitter, "blink": 0.05}


def capture(svc, face, start, seconds):
    for i in range(int(seconds * FPS)):
        svc._detect_frame(face, None, start + i / FPS)


def new_service():
    svc = vision.VisionService.__new__(vision.VisionService)
    svc.samples, svc.blinks, svc.baseline = [], [], {}
    svc._lock, svc._last_detect_error = threading.Lock(), None
    return svc


def main():
    svc = new_service()
    capture(svc, ScriptedFace(lambda t: 0.1), 0, 30)  # calibration
    svc.baseline = vision.baseline_from_window(svc.window(0, 30))

    answer_start = 100.0
    spike = lambda t: 0.5 if answer_start + 7.0 <= t < answer_start + 7.1 else 0.1  # 100 ms frown
    capture(svc, ScriptedFace(spike), answer_start, 20)
    deltas = svc.delta_sd(answer_start, answer_start + 20)
    assert deltas["AU04"] < 0.5, deltas  # averaged over 600 frames, the spike all but vanishes
    assert deltas[vision.MICRO_CHANNEL] >= 4.0, deltas
    assert fusion.bucket_arousal(deltas, CFG) == "high"
    (event,) = vision.micro_events(svc.window(answer_start, answer_start + 20), svc.baseline)
    assert event["channel"] == "AU04" and abs(event["onset"] - (answer_start + 7.0)) < 0.05, event

    calm = svc.delta_sd(200, 220)  # nothing recorded yet for this span
    assert calm[vision.MICRO_CHANNEL] is None and fusion.bucket_arousal(calm, CFG) == "unavailable"
    capture(svc, ScriptedFace(lambda t: 0.1), 200, 20)
    calm = svc.delta_sd(200, 220)
    assert calm[vision.MICRO_CHANNEL] == 0.0 and fusion.bucket_arousal(calm, CFG) == "low", calm

    # the old loop (one sample per 0.2 s, mean over the answer) misses the same spike
    old_samples = [s for s in svc.window(answer_start, answer_start + 20)
                   if round((s[0] - answer_start) * FPS) % 6 == 3]
    old = vision.delta_sd_from_window(old_samples, svc.baseline)
    assert fusion.bucket_arousal(old, CFG) == "low", old

    vision._selfcheck()
    print("micro-expression capture test passed")


if __name__ == "__main__":
    main()
