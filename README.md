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

The dashboard and orchestrator are separate processes that only talk over
that local HTTP API — either can be restarted without killing the other.

Stop both with:

```bash
# Linux/macOS
pkill -f orchestrator.py; pkill -f 'streamlit run'
```

```powershell
# Windows
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'orchestrator\.py|streamlit run' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## Status

The full turn loop is wired: Analyst/Interviewer agents, `fusion.py`,
`stoplogic.py`, manual claim entry, mic recording, and best-effort
BLE (`services/sensors.py`) + webcam (`services/vision.py`) capture. Any
missing piece — no Ollama, no mic, no ESP32, no camera — degrades to
"unavailable" for that channel rather than crashing (design doc Section 4).

Not built: the ESP32 firmware itself (Section 9, Open decision #8) — only
the BLE profile `services/sensors.py` expects.

## Tests

Pure logic and self-contained services can be checked without any hardware,
Ollama, or GPU:

```bash
python tests/test_fusion.py
python tests/test_stoplogic.py
python tests/test_orchestrator_api.py   # monkeypatches the Analyst/Interviewer/mic calls
python services/llm.py
python services/stt.py
python services/sensors.py
python services/vision.py               # baseline/delta math only, no camera needed
```
