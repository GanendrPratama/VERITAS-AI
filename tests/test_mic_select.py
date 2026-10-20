"""Microphone selection is remembered: a new orchestrator (restart, or the dashboard reloading
against a restarted one) reopens the mic picked last, matched by name since device indices
shift when mics are plugged in or out. Device listing is faked -- no microphone here."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orchestrator


def main():
    inputs = [(1, "Built-in Mic"), (3, "USB Headset")]
    orchestrator.stt.list_inputs = lambda: inputs

    with tempfile.TemporaryDirectory() as tmp:
        orch = orchestrator.Orchestrator(logs_dir=tmp)
        assert orch.recorder.device is None  # nothing saved yet: system default
        orch.set_devices({"mic": 3})
        assert orch.recorder.device == 3

        assert orchestrator.Orchestrator(logs_dir=tmp).recorder.device == 3

        inputs[:] = [(0, "USB Headset"), (2, "Built-in Mic")]  # re-plugged: same mic, new index
        assert orchestrator.Orchestrator(logs_dir=tmp).recorder.device == 0

        inputs[:] = [(2, "Built-in Mic")]  # unplugged: fall back to the system default
        assert orchestrator.Orchestrator(logs_dir=tmp).recorder.device is None
    print("test_mic_select passed")


if __name__ == "__main__":
    main()
