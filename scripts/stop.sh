#!/usr/bin/env bash
# Stops the orchestrator and dashboard started by start.sh, escalating to
# SIGKILL, then verifies nothing is left running or listening on 8000/8501.
PATTERNS=('supervise\.py' 'orchestrator\.py' 'streamlit run ui/app\.py')
PORTS=(8000 8501)

alive() { for p in "${PATTERNS[@]}"; do pgrep -f "$p" >/dev/null && return 0; done; return 1; }
signal() { for p in "${PATTERNS[@]}"; do pkill "-$1" -f "$p"; done; }
listening() { ss -ltn 2>/dev/null | grep -qE ":($(IFS='|'; echo "${PORTS[*]}"))\s"; }

signal TERM
for _ in 1 2 3 4 5 6 7 8 9 10; do alive || break; sleep 0.5; done
alive && { echo "still running after SIGTERM -- sending SIGKILL"; signal KILL; sleep 1; }

# a stray process (not matched by name) may still hold a port
if listening; then
    for port in "${PORTS[@]}"; do fuser -k -KILL "$port"/tcp >/dev/null 2>&1; done
    sleep 1
fi

if alive || listening; then
    echo "FAILED: processes or ports still in use" >&2
    pgrep -af 'supervise\.py|orchestrator\.py|streamlit run' >&2
    ss -ltnp 2>/dev/null | grep -E ":(8000|8501)\s" >&2
    exit 1
fi
echo "orchestrator and dashboard stopped; ports 8000/8501 free"
