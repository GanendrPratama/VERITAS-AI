"""Benchmark micro-expression capture on a real face video, no webcam needed.

    python scripts/micro_bench.py path/to/face.mp4 [--trials 10] [--old-latency 0 0.5]

1. Speed: the MediaPipe face model's time per frame on this machine's CPU.
2. Capture: builds test clips from the video (a short clip is ping-ponged so it loops without
   a jump), warps the subject's brows down and together (AU04, with the lids narrowing as a
   real frown does) for a known number of frames, and checks which pipeline sees it:
     old: one sample every 0.2 s + the face model's own time per frame (the previous capture
          loop), then the mean over the whole answer against the calibration baseline
     new: every frame (30 fps) through services.vision.micro_events
   Both use the same MediaPipe signal, so this compares sampling and aggregation only.
   Clips with no injected expression count false alarms.

The video should show one mostly still, neutral face (a webcam recording of someone
listening works well).
"""
import argparse
import random
import statistics
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import vision  # noqa: E402

FPS = 30
CLIP_SEC = 10
DURATIONS_MS = [40, 67, 100, 200, 333, 500]
L_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
R_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
INNER_BROW = {107, 55, 65, 336, 285, 295, 66, 296}


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            return frames
        frames.append(frame)


def landmarks(frames, model_path):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as mpv

    lm = mpv.FaceLandmarker.create_from_options(mpv.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)), num_faces=1))
    out = []
    for f in frames:
        r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        h, w = f.shape[:2]
        out.append(np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]]) if r.face_landmarks else None)
    return out


def frown(frame, pts, amount):
    """Brows down and together by `amount` (0..1 of a clear frown), as a smooth image warp."""
    if amount <= 0 or pts is None:
        return frame
    io = np.linalg.norm(pts[33] - pts[263])  # interocular distance sets the scale
    mid = (pts[33][0] + pts[263][0]) / 2
    h, w = frame.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy, wsum = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    sigma = 0.09 * io
    for i in L_BROW + R_BROW:
        d = np.array([0.0, 0.10 * io])
        if i in INNER_BROW:
            d += [0.05 * io * np.sign(mid - pts[i][0]), 0.04 * io]
        g = np.exp(-((xs - pts[i][0]) ** 2 + (ys - pts[i][1]) ** 2) / (2 * sigma ** 2))
        dx += g * d[0] * amount
        dy += g * d[1] * amount
        wsum += g
    norm = np.maximum(wsum, 1.0)
    return cv2.remap(frame, xs - dx / norm, ys - dy / norm, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def envelope(n):
    """onset -> apex -> offset over n frames (a raised cosine; flat for 1-2 frames)."""
    if n <= 2:
        return [1.0] * n
    return list(0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n))


def run_clip(job):
    """Analyse one synthetic clip frame by frame, exactly as the live capture loop does."""
    model_path, base, pts, start, n_frames, inject_at, n_event, amount = job
    face = vision.MediaPipeFace(model_path)
    seq = list(range(len(base))) + list(range(len(base) - 2, 0, -1))  # ping-pong: loops without a jump
    env = envelope(n_event)
    samples, blinks, lat = [], [], []
    for k in range(n_frames):
        i = seq[(start + k) % len(seq)]
        frame = base[i]
        if inject_at is not None and inject_at <= k < inject_at + n_event:
            frame = frown(frame, pts[i], amount * env[k - inject_at])
        ts = k / FPS
        t0 = time.perf_counter()
        values = face.analyse(frame, ts)
        lat.append(time.perf_counter() - t0)
        if values:
            samples += [(ts, au, values[au]) for au in vision.AU_CHANNELS]
            blinks.append((ts, values["blink"]))
    return samples, blinks, lat


def old_pipeline_hit(samples, baseline, period, phase, elevated_sd=1.0):
    """The previous loop: one sample per `period` seconds, mean over the answer, any AU >= 1 SD."""
    times = sorted({t for t, _c, _v in samples})
    picks, nxt = set(), phase
    for t in times:
        if t >= nxt:
            picks.add(t)
            nxt += period
    sub = [s for s in samples if s[0] in picks]
    deltas = vision.delta_sd_from_window(sub, baseline)
    return any(d is not None and d >= elevated_sd for d in deltas.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--trials", type=int, default=10, help="clips per duration/strength")
    ap.add_argument("--old-latency", type=float, nargs="+", default=[0.0, 0.5],
                    help="face-model seconds per frame to assume for the old 5 fps loop")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--z-on", type=float, nargs="+", default=[4.0], help="micro_events peak thresholds to compare")
    args = ap.parse_args()

    model_path = vision.face_landmarker_path()
    base = read_frames(args.video)
    base = [cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1]))) for f in base]
    pts = landmarks(base, model_path)
    rng = random.Random(0)
    period_frames = 2 * len(base) - 2
    n = CLIP_SEC * FPS

    jobs = [(model_path, base, pts, rng.randrange(period_frames), 30 * FPS, None, 0, 0)]  # calibration
    labels = [("calibration", 0, 0)]
    for amount in (0.5, 1.0):
        for ms in DURATIONS_MS:
            for _ in range(args.trials):
                n_event = max(1, round(ms * FPS / 1000))
                jobs.append((model_path, base, pts, rng.randrange(period_frames), n,
                             rng.randrange(FPS, n - 2 * FPS), n_event, amount))
                labels.append(("event", ms, amount))
    for _ in range(3 * args.trials):
        jobs.append((model_path, base, pts, rng.randrange(period_frames), n, None, 0, 0))
        labels.append(("neutral", 0, 0))

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(run_clip, jobs)
    print(f"analysed {sum(len(r[2]) for r in results)} frames in {time.time() - t0:.0f}s")

    lat = np.array([x for r in results for x in r[2][5:]]) * 1000
    print(f"\nMediaPipe face model, CPU: median {np.median(lat):.1f} ms/frame, p95 {np.percentile(lat, 95):.1f} ms"
          f" -> up to {1000 / np.median(lat):.0f} fps (camera-limited to {FPS})")

    try:  # the old face model's time per frame, when py-feat and its weights are installed here
        face = vision.PyFeatFace("cpu")
        times = []
        for frame in base[:10]:
            t1 = time.perf_counter()
            face.analyse(frame, 0)
            times.append(time.perf_counter() - t1)
        print(f"py-feat face model, CPU: median {1000 * statistics.median(times[2:]):.0f} ms/frame"
              f" -> old loop samples every {0.2 + statistics.median(times[2:]):.2f}s")
    except Exception as e:
        print(f"py-feat not timed ({type(e).__name__}); pass its seconds per frame with --old-latency")

    cal_samples = results[0][0]
    baseline = vision.baseline_from_window(cal_samples)
    print("calibration baseline (mean, sd):", {k: (round(m, 3), round(s, 3)) for k, (m, s) in baseline.items()})

    def new_hit(res, job, z_on):
        events = vision.micro_events(res[0], baseline, res[1], z_on=z_on)
        if job[5] is None:
            return bool(events)
        lo, hi = job[5] / FPS - 0.2, (job[5] + job[6]) / FPS + 0.2
        return any(lo <= e["apex"] <= hi for e in events)

    rows = []
    header = ["expression"] + [f"new (every frame, onset/apex, z>={z:g})" for z in args.z_on] + [
        f"old (5 fps, +{lat_s:g}s model, mean)" for lat_s in args.old_latency]
    for amount in (0.5, 1.0):
        for ms in DURATIONS_MS:
            idx = [i for i, lab in enumerate(labels) if lab == ("event", ms, amount)]
            row = [f"{ms} ms, {'subtle' if amount < 1 else 'clear'}"] + [
                statistics.mean(new_hit(results[i], jobs[i], z) for i in idx) for z in args.z_on]
            for lat_s in args.old_latency:
                period = 0.2 + lat_s
                row.append(statistics.mean(
                    old_pipeline_hit(results[i][0], baseline, period, rng.uniform(0, period)) for i in idx))
            rows.append(row)
    idx = [i for i, lab in enumerate(labels) if lab[0] == "neutral"]
    fa = ["no expression (false alarm)"] + [statistics.mean(new_hit(results[i], jobs[i], z) for i in idx)
                                            for z in args.z_on]
    for i in idx:
        for e in vision.micro_events(results[i][0], baseline, results[i][1]):
            print(f"false alarm: {e['channel']} at {e['onset']:.2f}s, {1000 * (e['offset'] - e['onset']):.0f} ms,"
                  f" peak {e['peak_z']:.1f} SD")
    for lat_s in args.old_latency:
        period = 0.2 + lat_s
        fa.append(statistics.mean(
            old_pipeline_hit(results[i][0], baseline, period, rng.uniform(0, period)) for i in idx))
    rows.append(fa)

    print("\n| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join([row[0]] + [f"{v:.0%}" for v in row[1:]]) + " |")


if __name__ == "__main__":
    main()
