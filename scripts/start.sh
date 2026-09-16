#!/usr/bin/env bash
# Starts the orchestrator API and the operator dashboard, detached from this
# shell (nohup + disown) so they keep running after the terminal closes.
set -e
cd "$(dirname "$0")/.."
VENV="${VENV:-venv}"
mkdir -p logs

nohup "$VENV/bin/python" orchestrator.py > logs/orchestrator.log 2>&1 &
disown
nohup "$VENV/bin/streamlit" run ui/app.py --server.port 8501 --server.headless true > logs/dashboard.log 2>&1 &
disown

echo "orchestrator: http://localhost:8000  (log: logs/orchestrator.log)"
echo "dashboard:    http://localhost:8501  (log: logs/dashboard.log)"
echo "stop with:    pkill -f orchestrator.py; pkill -f 'streamlit run'"
