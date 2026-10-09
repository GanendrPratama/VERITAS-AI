"""Camera selection: a stopped session doesn't lock the camera, an index that won't open is
refused (old camera kept), and calibration moves off a stale index onto one that works.
Camera probing is faked -- no webcam here."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import orchestrator
from services import vision


class FakeVision:
    def __init__(self, index):
        self.camera_index, self.running, self.baseline = index, False, {}

    def is_running(self):
        return self.running

    def start(self):
        self.running = True

    def stop(self):
        self.running = False


def main():
    working = {0: "black", 2: "ok"}
    vision.probe_camera = lambda i, n_frames=6: working.get(i)

    with tempfile.TemporaryDirectory() as tmp:
        orch = orchestrator.Orchestrator(logs_dir=tmp)
        orch.vision = FakeVision(7)

        orch.set_camera(2)
        assert orch.vision.camera_index == 2 and orch.vision.running

        try:
            orch.set_camera(5)
            assert False, "should have refused a camera that won't open"
        except orchestrator.OrchestratorError as e:
            assert e.status == 400, e
        assert orch.vision.camera_index == 2 and orch.vision.running  # old camera kept

        orch.active = orchestrator.new_session_state("s1", "report")
        orch.active["phase"] = "interview"
        try:
            orch.set_camera(0)
            assert False, "should have refused mid-interview"
        except orchestrator.OrchestratorError:
            pass

        orch.active["phase"] = "stopped"  # used to be refused too, leaving a bad index stuck
        orch.set_camera(0)
        assert orch.vision.camera_index == 0

        # calibration on a stale index falls back to the working camera
        orch.vision = FakeVision(7)
        orch.sensors = orchestrator._NullHardwareService()
        orch.active = orchestrator.new_session_state("s2", "report")
        orch.calibrate_start()
        assert orch.vision.camera_index == 2 and orch.vision.running, orch.vision.camera_index
        assert orch.cameras()["notes"] == {"0": vision.CAMERA_NOTES["black"], "2": None}
    print("test_camera_select passed")


if __name__ == "__main__":
    main()
