"""BLE sensor link against the ESP32-C3 firmware, which publishes heart rate only (no GSR
characteristic): HR still streams, GSR is simply absent. bleak is faked -- no BLE radio here."""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import sensors

HR_UUID = "2A37"
GSR_UUID = "6b3a2c00-1e3d-4f6a-9c1e-2a1a2f3b4c5d"


class FakeServices:
    def __init__(self, uuids):
        self.uuids = uuids

    def get_characteristic(self, uuid):
        return uuid if uuid in self.uuids else None


def fake_bleak(char_uuids, packets):
    """A device exposing char_uuids; once subscribed, HR notifies `packets` then disconnects."""
    subscribed = []

    class Scanner:
        @staticmethod
        async def find_device_by_name(name, timeout):
            return name

    class Client:
        def __init__(self, device):
            self.services = FakeServices(char_uuids)
            self.is_connected = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.is_connected = False

        async def start_notify(self, uuid, callback):
            if uuid not in char_uuids:  # what bleak does for a characteristic the device lacks
                raise ValueError(f"Characteristic {uuid} was not found!")
            subscribed.append(uuid)
            if uuid == HR_UUID:
                for p in packets:
                    callback(None, bytearray(p))
                self.is_connected = False

        async def disconnect(self):
            self.is_connected = False

    return Scanner, Client, subscribed


def run(char_uuids, packets):
    sensors.BleakScanner, sensors.BleakClient, subscribed = fake_bleak(char_uuids, packets)
    svc = sensors.SensorService("VERITAS-SENSOR", HR_UUID, GSR_UUID)
    asyncio.run(svc._connect_and_listen())
    return svc, subscribed


def main():
    # ESP32-C3 firmware: HR service only, flags 0x00 + uint8 BPM
    svc, subscribed = run({HR_UUID}, [[0x00, 72], [0x00, 75]])
    assert svc.error is None, svc.error
    assert subscribed == [HR_UUID]
    assert [(ch, v) for _t, ch, v in svc.samples] == [("hr", 72), ("hr", 75)], svc.samples
    deltas = svc.delta_sd(0, time.time() + 1)
    assert "gsr" not in deltas  # no GSR baseline -> channel absent, not a crash

    # older firmware with GSR still gets both channels
    svc, subscribed = run({HR_UUID, GSR_UUID}, [[0x00, 70]])
    assert svc.error is None and subscribed == [HR_UUID, GSR_UUID], (svc.error, subscribed)

    sensors._selfcheck()
    print("test_sensors passed")


if __name__ == "__main__":
    main()
