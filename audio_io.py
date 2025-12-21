import logging
import math
import time
from dataclasses import dataclass
from threading import Event
from typing import Iterable, List, Optional

import numpy as np

try:
    import scipy.signal
except ImportError:  # pragma: no cover - optional dep
    scipy = None  # type: ignore
    scipy_signal_resample_poly = None  # type: ignore
else:
    scipy_signal_resample_poly = scipy.signal.resample_poly  # type: ignore

import pyaudio

logger = logging.getLogger(__name__)


def create_pyaudio() -> pyaudio.PyAudio:
    pa = pyaudio.PyAudio()
    return pa


@dataclass
class InputDeviceInfo:
    index: int
    name: str
    rate: int
    channels: int


@dataclass
class RecordingResult:
    audio: bytes
    reason: str
    speech_detected: bool
    max_rms: float
    duration_ms: float
    voice_ms: float


def get_input_device(pa: pyaudio.PyAudio, device_index: Optional[int]) -> InputDeviceInfo:
    if device_index is None:
        device_index = int(pa.get_default_input_device_info()["index"])
    info = pa.get_device_info_by_index(device_index)
    rate = int(info.get("defaultSampleRate", 16000))
    channels = int(info.get("maxInputChannels", 1)) or 1
    logger.info("Selected input device %s: %s (rate=%s, channels=%s)", device_index, info.get("name"), rate, channels)
    return InputDeviceInfo(index=device_index, name=info.get("name", "unknown"), rate=rate, channels=channels)


class FrameAdapter:
    """Converts arbitrary input chunks to fixed-size frames at target rate."""

    def __init__(self, input_rate: int, input_channels: int, target_rate: int, target_frame_len: int):
        self.input_rate = input_rate
        self.input_channels = input_channels
        self.target_rate = target_rate
        self.target_frame_len = target_frame_len
        self.buffer = np.array([], dtype=np.int16)
        self.resampler = scipy_signal_resample_poly

    def _to_mono(self, data: np.ndarray) -> np.ndarray:
        if self.input_channels == 1:
            return data
        reshaped = data.reshape(-1, self.input_channels)
        mono = reshaped.mean(axis=1)
        return mono.astype(np.int16)

    def _resample(self, samples: np.ndarray) -> np.ndarray:
        if self.input_rate == self.target_rate:
            return samples
        if self.resampler:
            g = math.gcd(self.target_rate, self.input_rate)
            up = self.target_rate // g
            down = self.input_rate // g
            resampled = self.resampler(samples, up, down)
            return resampled.astype(np.int16)
        # Fallback: simple linear interpolation
        duration = len(samples) / self.input_rate
        target_len = int(duration * self.target_rate)
        target_idx = np.linspace(0, len(samples) - 1, target_len)
        resampled = np.interp(target_idx, np.arange(len(samples)), samples)
        return resampled.astype(np.int16)

    def process(self, chunk: bytes) -> List[bytes]:
        samples = np.frombuffer(chunk, dtype=np.int16)
        mono = self._to_mono(samples)
        resampled = self._resample(mono)
        self.buffer = np.concatenate([self.buffer, resampled])
        frames: List[bytes] = []
        while len(self.buffer) >= self.target_frame_len:
            frame = self.buffer[: self.target_frame_len]
            frames.append(frame.astype(np.int16).tobytes())
            self.buffer = self.buffer[self.target_frame_len :]
        return frames


class MicrophoneStream:
    def __init__(
        self,
        pa: pyaudio.PyAudio,
        device: InputDeviceInfo,
        frames_per_buffer: int = 1024,
    ):
        self.pa = pa
        self.device = device
        self.frames_per_buffer = frames_per_buffer
        self.stream: Optional[pyaudio.Stream] = None

    def __enter__(self) -> "MicrophoneStream":
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.device.channels,
            rate=self.device.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
            input_device_index=self.device.index,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def read_chunk(self) -> bytes:
        if not self.stream:
            raise RuntimeError("Microphone stream is not open")
        return self.stream.read(self.frames_per_buffer, exception_on_overflow=False)


class AudioPlayer:
    def __init__(self, pa: pyaudio.PyAudio, rate: int):
        self.pa = pa
        self.rate = rate
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            output=True,
        )

    def play_bytes(self, audio_bytes: bytes) -> None:
        self.stream.write(audio_bytes)

    def play_stream(self, chunks: Iterable[bytes]) -> None:
        for chunk in chunks:
            if not chunk:
                continue
            self.play_bytes(chunk)

    def close(self) -> None:
        self.stream.stop_stream()
        self.stream.close()


def record_utterance(
    mic: MicrophoneStream,
    adapter: FrameAdapter,
    max_seconds: int,
    vad,
    frame_ms: int,
    stop_event: Optional[Event] = None,
    start_timeout_ms: int = 2500,
    vad_grace_ms: int = 900,
    min_record_ms: int = 2500,
    max_record_ms: int = 12000,
    preroll_ms: int = 0,
    min_audio_ms: int = 300,
    rms_speech_start: float = 120.0,
    rms_speech_frames: int = 2,
    log_interval_ms: int = 250,
) -> RecordingResult:
    """Record until silence or limits hit, returning audio and finalize reason."""

    start = time.time()
    collected: List[bytes] = []
    silence_run_ms = 0
    heard_voice = False
    speech_frames = 0
    max_rms = 0.0
    grace_until = start + vad_grace_ms / 1000.0
    min_until = start + min_record_ms / 1000.0
    no_speech_deadline = start + start_timeout_ms / 1000.0
    hard_deadline = start + max_record_ms / 1000.0
    last_log = start
    last_rms = 0.0
    preroll_frames = max(0, int(preroll_ms / frame_ms))

    finalize_reason = "unknown"

    def log_energy(rms: float, tag: str) -> None:
        logger.info("energy_rms=%.1f tag=%s", rms, tag)

    def finish(reason: str, speech: bool, audio_bytes: bytes, voice_ms: float) -> RecordingResult:
        duration_ms = len(audio_bytes) / 2 / adapter.target_rate * 1000 if audio_bytes else 0.0
        logger.info(
            "recording_summary speech_detected=%s max_rms=%.1f duration_ms=%.0f voice_ms=%.0f finalize_reason=%s",
            speech,
            max_rms,
            duration_ms,
            voice_ms,
            reason,
        )
        return RecordingResult(audio_bytes, reason, speech, max_rms, duration_ms, voice_ms)

    # Optional preroll capture (best effort).
    for _ in range(preroll_frames):
        if stop_event and stop_event.is_set():
            return finish("stopped", False, b"".join(collected), 0.0)
        chunk = mic.read_chunk()
        frames = adapter.process(chunk)
        for frame in frames:
            collected.append(frame)

    voice_ms_acc = 0.0

    while True:
        now = time.time()
        if stop_event and stop_event.is_set():
            finalize_reason = "stopped"
            break
        if now - start >= max_seconds:
            finalize_reason = "max_utterance"
            break
        if now >= hard_deadline:
            finalize_reason = "max_record_ms"
            break
        chunk = mic.read_chunk()
        frames = adapter.process(chunk)
        for frame in frames:
            collected.append(frame)
            samples = np.frombuffer(frame, dtype=np.int16)
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            last_rms = rms
            if rms > max_rms:
                max_rms = rms
            is_silence = vad.is_silence(frame)
            if not heard_voice:
                if rms >= rms_speech_start:
                    speech_frames += 1
                    if speech_frames >= rms_speech_frames:
                        heard_voice = True
                        silence_run_ms = 0
                        voice_ms_acc += frame_ms
                        log_energy(rms, "speech_detected")
                else:
                    speech_frames = 0
                if not heard_voice and now >= no_speech_deadline and now >= grace_until:
                    finalize_reason = "no_speech_timeout"
                    joined = b"".join(collected)
                    return finish(finalize_reason, False, joined, voice_ms_acc)
            else:
                silence_run_ms = silence_run_ms + frame_ms if is_silence else 0
                if rms >= rms_speech_start:
                    voice_ms_acc += frame_ms
                elapsed_ms = (now - start) * 1000
                if elapsed_ms >= min_record_ms and silence_run_ms >= vad.silence_ms:
                    finalize_reason = "silence"
                    return finish(finalize_reason, True, b"".join(collected), voice_ms_acc)

            if (time.time() - last_log) * 1000 >= log_interval_ms:
                log_energy(last_rms, "recording")
                last_log = time.time()

    audio_bytes = b"".join(collected)
    total_ms = len(audio_bytes) / 2 / adapter.target_rate * 1000
    enough_audio = total_ms >= min_audio_ms
    if not enough_audio:
        return finish("too_short", heard_voice, b"", voice_ms_acc)
    return finish(finalize_reason, heard_voice, audio_bytes, voice_ms_acc)
