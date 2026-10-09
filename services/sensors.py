"""BLE sensor service: heart rate (standard BLE Heart Rate Service) + GSR
(custom notify characteristic) from the ESP32. See docs/veritas-ai-design.md
Section 7 for the exact `ble_*` UUIDs the firmware must implement -- that
firmware isn't built yet (Open decision #8), so this can't be tested against
real hardware from this machine; the pure baseline/delta math below can be
and is (see _selfcheck).

Runs its own asyncio event loop in a background thread since bleak's API is
async-only, so the rest of the orchestrator (plain sync code) just calls
start()/stop()/calibrate()/delta_sd() -- same shape as services/vision.py.
Any BLE failure (no device, disconnect, missing firmware) is swallowed: the
channel just never gets samples, so delta_sd() reports it as unavailable
rather than crashing the interview (Section 4).
"""
import asyncio
import statistics
import threading
import time

try:
    from bleak import BleakClient, BleakScanner
except ImportError:  # keep the pure baseline/delta math importable without BLE support
    BleakClient = BleakScanner = None


def _group_by_channel(window):
    grouped = {}
    for _t, ch, v in window:
        grouped.setdefault(ch, []).append(v)
    return grouped


def baseline_from_window(window):
    """window: [(timestamp, channel, value), ...] -> {channel: (mean, sd)}."""
    baseline = {}
    for ch, values in _group_by_channel(window).items():
        if len(values) >= 2:
            baseline[ch] = (statistics.mean(values), statistics.stdev(values))
    return baseline


def delta_sd_from_window(window, baseline):
    """-> {channel: sd_above_baseline or None}, one entry per baselined channel."""
    grouped = _group_by_channel(window)
    deltas = {}
    for ch, (mean, sd) in baseline.items():
        values = grouped.get(ch)
        deltas[ch] = None if not values or sd == 0 else (statistics.mean(values) - mean) / sd
    return deltas


class SensorService:
    def __init__(self, device_name, hr_char_uuid, gsr_char_uuid):
        if BleakClient is None:
            raise ImportError("bleak not installed")
        self.device_name = device_name
        self.hr_char_uuid = hr_char_uuid
        self.gsr_char_uuid = gsr_char_uuid
        self.samples = []  # [(timestamp, channel, value)]
        self.baseline = {}
        self._lock = threading.Lock()
        self._loop = None
        self._client = None
        self._connected = threading.Event()
        self._thread = None
        self.error = None  # last failure reason, shown in the UI status panel

    def start(self):
        # Non-blocking: BLE scan/connect can take seconds, and calibrate_start
        # shouldn't stall on it -- the channel just has no samples until (and
        # unless) this connects. Check is_connected() if the UI wants status.
        if self._thread is not None and self._thread.is_alive():
            return  # already scanning/connected (e.g. a session reopened mid-interview)
        self.error = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def is_connected(self):
        return self._connected.is_set()

    def stop(self):
        # Only while the loop is alive -- after a failed scan it has already finished, and
        # scheduling onto it just leaks a never-awaited coroutine.
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._disconnect(), self._loop)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_listen())

    async def _connect_and_listen(self):
        try:
            device = await BleakScanner.find_device_by_name(self.device_name, timeout=10)
            if device is None:
                self.error = f"BLE scan (10s) found no device named {self.device_name!r}"
                return
            async with BleakClient(device) as client:
                self._client = client
                self._connected.set()
                await client.start_notify(self.hr_char_uuid, self._on_hr)
                await client.start_notify(self.gsr_char_uuid, self._on_gsr)
                while client.is_connected:
                    await asyncio.sleep(0.5)
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"  # degrade gracefully -- channel just stays unavailable
        finally:
            self._connected.clear()
            self._client = None

    async def _disconnect(self):
        if self._client:
            await self._client.disconnect()

    def _on_hr(self, _handle, data):
        # BLE Heart Rate Measurement, uint8 format: flags byte, then BPM.
        self._record("hr", data[1])

    def _on_gsr(self, _handle, data):
        self._record("gsr", int.from_bytes(data[:2], "little"))

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
    # Synthetic buffer -- exactly the "sensor-delta check" from design doc Section 5.
    baseline_window = [(i, "hr", 60 + (i % 3)) for i in range(10)]  # mean ~61, small sd
    baseline = baseline_from_window(baseline_window)
    assert "hr" in baseline
    mean, sd = baseline["hr"]
    assert 60 <= mean <= 62 and sd > 0

    spiked_window = [(i, "hr", mean + 4 * sd) for i in range(10, 15)]
    deltas = delta_sd_from_window(spiked_window, baseline)
    assert abs(deltas["hr"] - 4.0) < 1e-9, deltas

    # missing channel -> None, not a crash
    deltas = delta_sd_from_window([(0, "gsr", 100)], baseline)
    assert deltas == {"hr": None}, deltas

    # too few samples to baseline -> no entry at all
    assert baseline_from_window([(0, "hr", 60)]) == {}

    print("sensors.py self-check passed")


if __name__ == "__main__":
    _selfcheck()
