"""faster-whisper wrapper + mic recording.

Transcription runs on GPU by default per the design's latency pass
(Section 7, `stt_device`) — same model, same accuracy, lower latency than
CPU. On a Windows box the CUDA path needs cuDNN's DLLs discoverable on PATH
(they don't ship with the CUDA Toolkit installer by default); if that's not
set up, CTranslate2 fails to load the CUDA backend, so we fall back to CPU
rather than crash the interview (Section 4: degrade, never crash).

Recorder captures the mic between the operator's Record/Stop clicks (Section
1's "one turn") straight to a WAV file for transcribe() to consume.
"""
import wave

from faster_whisper import WhisperModel


def load_model(model_size="small", device="cuda", compute_type=None):
    compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception:
        if device == "cuda":
            return WhisperModel(model_size, device="cpu", compute_type="int8")
        raise


def write_wav(path, samples_int16, samplerate):
    """samples_int16: a mono int16 numpy array (or anything with .tobytes())."""
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(samples_int16.tobytes())


class Recorder:
    """Mic capture for one answer: start() on Record, stop() on Stop.

    The input stream stays open once opened (open() is idempotent) so the UI
    can show a live level meter; frames are only kept while recording."""

    def __init__(self, samplerate=16000):
        self.samplerate = samplerate
        self.level = 0.0  # 0..1, latest block loudness, for the live meter
        self.error = None
        self.device_name = None
        self._frames = []
        self._recording = False
        self._stream = None

    def is_recording(self):
        return self._recording

    def open(self):
        if self._stream is not None:
            return
        try:
            import numpy as np
            import sounddevice as sd

            def _callback(indata, _frames, _time_info, _status):
                self.level = min(1.0, float(np.sqrt(np.mean(indata.astype("float32") ** 2))) / 8000)
                if self._recording:
                    self._frames.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=self.samplerate, channels=1, dtype="int16", callback=_callback
            )
            self._stream.start()
            self.device_name = sd.query_devices(kind="input")["name"]
            self.error = None
        except Exception as e:
            self._stream = None
            self.error = f"{type(e).__name__}: {e}"
            raise

    def start(self):
        self._frames = []
        self.open()
        self._recording = True

    def snapshot(self):
        """Audio recorded so far as float32 mono in [-1, 1] (what faster-whisper takes), or None."""
        import numpy as np

        frames = list(self._frames)
        return np.concatenate(frames)[:, 0].astype("float32") / 32768 if frames else None

    def stop(self, out_path):
        import numpy as np

        self._recording = False
        audio = np.concatenate(self._frames, axis=0) if self._frames else np.zeros((0, 1), dtype="int16")
        self._frames = []
        write_wav(out_path, audio, self.samplerate)
        return out_path


def transcribe(model, audio, min_logprob, **options):
    """audio: a file path or a 16 kHz float32 array. options go to model.transcribe."""
    segments, _info = model.transcribe(audio, **options)
    segments = list(segments)
    text = " ".join(s.text.strip() for s in segments).strip()
    avg_logprob = sum(s.avg_logprob for s in segments) / len(segments) if segments else -999.0
    return {"text": text, "unclear": avg_logprob < min_logprob}


if __name__ == "__main__":
    class FakeSeg:
        def __init__(self, text, avg_logprob):
            self.text, self.avg_logprob = text, avg_logprob

    class FakeModel:
        def transcribe(self, path):
            return [FakeSeg(" halo", -0.2), FakeSeg(" dunia", -0.3)], None

    assert transcribe(FakeModel(), "fake.wav", min_logprob=-1.0) == {
        "text": "halo dunia",
        "unclear": False,
    }

    class FakeGarbledModel:
        def transcribe(self, path):
            return [FakeSeg(" ...", -3.0)], None

    assert transcribe(FakeGarbledModel(), "fake.wav", min_logprob=-1.0)["unclear"] is True

    import array
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "out.wav")
        samples = array.array("h", [0, 100, -100, 32767, -32768])  # int16 samples
        write_wav(path, samples, samplerate=16000)
        with wave.open(path, "rb") as f:
            assert f.getnchannels() == 1
            assert f.getsampwidth() == 2
            assert f.getframerate() == 16000
            assert f.getnframes() == len(samples)

    print("stt.py self-check passed")
