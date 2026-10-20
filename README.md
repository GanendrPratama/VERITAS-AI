# VERITAS-AI

Design: [`veritas-ai-design.md`](docs/veritas-ai-design.md) (untracked — see `.gitignore`).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Start it up

Linux/macOS:

```bash
./scripts/start.sh
```

Windows (PowerShell):

```powershell
.\scripts\start.ps1
```

(If scripts are disabled: `Set-ExecutionPolicy -Scope Process RemoteSigned` first.)

Either way, both processes start detached — they keep running after the
terminal closes. Orchestrator API on http://localhost:8000, dashboard on
http://localhost:8501. Logs go to `logs/orchestrator.log` and
`logs/dashboard.log` (Windows: `logs\orchestrator.err.log` /
`logs\dashboard.err.log` also capture stderr).

Also needs, separately: an [Ollama](https://ollama.com) server running the
model in `config.toml` (`ollama pull qwen2.5:7b-instruct-q4_K_M`) for the
Analyst/Interviewer agents — without it, Record/Generate Question return a
clean error instead of crashing, but nothing actually assesses answers.

Open http://localhost:8501 in a browser. Start with a new session (upload the
subject's report as a PDF), type in the claims to track, then work through
the turn loop with the on-screen buttons.

### Dashboard

The UI follows the SlideFlow mockup (dark theme, 4-step tracker in the
sidebar: Document → Claims → Interview → Results). The mockup itself
(`VERITAS-AI_Dashboard_SlideFlow.html`) is a local design reference and is
gitignored.

**Component Status** (sidebar, refreshes every 3s) shows whether the ESP32
sensor, camera, microphone, speech-to-text and Ollama are detected, with the
reason when not (e.g. "VERITAS-SENSOR not found over BLE"). A red
"Not detected: …" banner repeats any `down` component on every screen. The
ESP32 and camera only start at calibration, so they show `idle` until then.
The data comes from the orchestrator's `GET /health`.

The **Logs** expander at the bottom of every screen live-tails the
orchestrator and dashboard logs in two tabs (`GET /logs/orchestrator`,
`GET /logs/dashboard`; last 200 lines of `logs/<name>.log`, plus `.err.log`
on Windows).

The dashboard and orchestrator are separate processes that only talk over
that local HTTP API — either can be restarted without killing the other.

`start.sh` / `start.ps1` run the orchestrator under `scripts/supervise.py`, which restarts it
if it ever exits (backoff 2s → 30s; each restart is logged to `logs/orchestrator.log`). It
also restarts it if it hangs: `GET /ping` silent for 30s (process frozen), or `GET /state`
blocked for 5 minutes (a request stuck holding the session lock). A
session that was open resumes automatically (calibration baselines are not saved, so an
interview resumed mid-way has no arousal signal).

The supervisor probes `GET /state` every 5s with the dashboard's 5s timeout and logs the
result per minute to `logs/uptime.jsonl`. `python scripts/uptime.py [--days N]` prints the
availability per day and overall against the 98% target, plus restarts by cause. Time
with the system stopped via `stop.sh` / `stop.ps1` is not counted.

Stop both with `scripts/stop.sh` (Linux/macOS) or `scripts\stop.ps1`
(Windows). They kill the whole process trees, escalate to a force-kill, and
exit non-zero if anything is still running or holding ports 8000/8501.

## Status

The full turn loop is wired: Analyst/Interviewer agents, `fusion.py`,
`stoplogic.py`, manual claim entry, mic recording, and best-effort
BLE (`services/sensors.py`) + webcam (`services/vision.py`) capture. Any
missing piece — no Ollama, no mic, no ESP32, no camera — degrades to
"unavailable" for that channel rather than crashing (design doc Section 4).

Webcam micro-expressions: MediaPipe Face Landmarker (`vision_model = "mediapipe"`,
about 10 ms a frame on a laptop CPU) analyses every camera frame, and each answer
gets a `micro` channel: the strongest brief (at most 0.5 s) brow-lowerer /
lid-tightener / lip-press spike against the calibration baseline. The model file
(3.6 MB) downloads into `~/.cache/veritas-ai/` on first start. Set
`vision_model = "pyfeat"` for the old py-feat model. To check capture on a recorded
face video instead of a webcam:

```bash
python scripts/micro_bench.py path/to/face.mp4
```

ESP32 firmware: `esp32/VERITAS/` (Section 9, Open decision #8) — implements
the BLE profile `services/sensors.py` expects, but untested on real hardware
from this machine.

## ESP32 sensor node

`esp32/VERITAS/` (PlatformIO project, `src/main.cpp`) — BLE GATT server
publishing heart rate (GSR was dropped in the ESP32-C3 build; the
dashboard still uses a GSR characteristic if a device exposes one), built against the same board this project
assumed from the start (Section 1: RTX 4050 laptop, GPU-bound; the sensor
node is a separate cheap microcontroller, not the laptop).

**Hardware**

| Part | Notes |
|---|---|
| Board | ESP32-C3 SuperMini (single-core RISC-V, BLE 5) |
| Heart rate | MAX30102 pulse oximeter breakout, I2C |

**Wiring**

| Signal | ESP32-C3 pin |
|---|---|
| MAX30102 SDA | GPIO5 |
| MAX30102 SCL | GPIO6 |
| MAX30102 VIN | 3.3V |
| MAX30102 GND | GND |

**BLE profile** (must match `config.toml`'s `ble_*` keys — `services/sensors.py`
connects by device name and reads by characteristic UUID):

| Key | Value | Notes |
|---|---|---|
| Device name | `VERITAS-SENSOR` | |
| Heart Rate service | `0x180D` | standard BLE SIG service |
| Heart Rate characteristic | `0x2A37` | standard BLE SIG char, notify, uint8 format (flags byte `0x00` + BPM) |
| GSR service | `bfe582f0-d884-43d4-aa9d-7648f8536e40` | optional, not in the ESP32-C3 firmware — custom, no BLE SIG service fits GSR |
| GSR characteristic | `6b3a2c00-1e3d-4f6a-9c1e-2a1a2f3b4c5d` | optional, custom, notify, raw 12-bit ADC as uint16 little-endian |

**Firmware architecture** — two FreeRTOS tasks joined by a queue, not one
`loop()` doing both jobs:

- `SensorTask` — polls the MAX30102 every ~2ms (beat detection needs
  frequent sampling) and pushes each new BPM onto `sampleQueue` while a
  finger is on the sensor. Never touches BLE.
- `BLETask` — blocks on `sampleQueue`, calls `notify()` the moment a reading
  arrives. Never touches sensor hardware.

Splitting sensing from BLE I/O means a slow BLE stack call can't stall
sensor polling, and vice versa — the two tasks only communicate through the
queue, never shared state. (The C3 is single-core, so they're interleaved by
the scheduler rather than pinned to separate cores.)

**Flashing** — a [PlatformIO](https://platformio.org) project
(`esp32/VERITAS/`), not an Arduino IDE sketch: `platformio.ini` pins the
board (`esp32-c3-devkitc-02`, with USB CDC on boot so the serial log
shows over the native USB port) and
declares the SparkFun MAX3010x library in `lib_deps`, so PlatformIO fetches
it and the ESP32 toolchain itself, no manual Library/Boards Manager steps.

```bash
cd esp32/VERITAS
pio run -t upload     # build + flash
pio device monitor    # serial log at 115200 baud
```

(VS Code + the PlatformIO IDE extension works the same way — open
`esp32/VERITAS/` as the project folder and use its Upload/Monitor buttons.)

Untested on real hardware from this machine (no ESP32/MAX30102 here) — in particular, verify the `irValue > 50000` finger-presence
threshold in `sensorTask` against your actual sensor.

## Tests

Pure logic and self-contained services can be checked without any hardware,
Ollama, or GPU:

```bash
python tests/test_fusion.py
python tests/test_stoplogic.py
python tests/test_orchestrator_api.py   # monkeypatches the Analyst/Interviewer/mic calls
python tests/test_supervise.py          # watchdog restarts + uptime log, with a stand-in orchestrator
python tests/test_micro_expressions.py  # brief AU spikes reach the arousal bucket, scripted face model
python tests/test_sensors.py            # heart-rate-only ESP32-C3 still streams over BLE, faked bleak
python tests/test_mic_select.py         # the picked microphone survives a restart, matched by name
python services/llm.py
python services/stt.py
python services/sensors.py
python services/vision.py               # baseline/delta/micro-expression math only, no camera needed
python services/docconvert.py           # PDF->Markdown, falls back to pypdf if markitdown is missing
```
