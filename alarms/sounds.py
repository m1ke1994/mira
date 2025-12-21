from __future__ import annotations

import logging
import math
import wave
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Optional

try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows fallback
    winsound = None  # type: ignore

try:  # Optional local TTS for spoken confirmations
    import pyttsx3
except ImportError:  # pragma: no cover - optional
    pyttsx3 = None  # type: ignore

logger = logging.getLogger(__name__)


def ensure_alarm_sound(path: Path, duration_seconds: float = 1.5) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 24000
    freq = 880.0
    amplitude = 0.4
    samples = int(duration_seconds * sample_rate)
    frames = bytearray()
    for i in range(samples):
        value = int(32767 * amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        frames.extend(value.to_bytes(2, byteorder="little", signed=True))
    with wave.open(path, "w") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    logger.info("Generated default alarm sound at %s", path)


class AlarmSoundPlayer:
    def __init__(self, sound_path: Path):
        self.sound_path = sound_path
        self._stop_event = Event()
        self._beep_thread: Optional[Thread] = None

    def start_loop(self) -> None:
        ensure_alarm_sound(self.sound_path)
        self._stop_event.clear()
        if winsound:
            try:
                winsound.PlaySound(
                    str(self.sound_path),
                    winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC,
                )
                return
            except RuntimeError:
                logger.warning("winsound.PlaySound failed, falling back to beep loop")

        # Generic fallback: simple beep loop in thread
        if self._beep_thread and self._beep_thread.is_alive():
            return
        self._beep_thread = Thread(target=self._beep_loop, name="alarm-beep", daemon=True)
        self._beep_thread.start()

    def stop_loop(self) -> None:
        self._stop_event.set()
        if winsound:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except RuntimeError:
                logger.debug("winsound.PlaySound purge failed")

    def _beep_loop(self) -> None:  # pragma: no cover - timing loop
        while not self._stop_event.is_set():
            if winsound:
                try:
                    winsound.Beep(880, 250)
                except RuntimeError:
                    logger.debug("winsound.Beep failed inside loop")
            else:
                logger.info("Alarm ringing...")
            self._stop_event.wait(0.75)


class LocalSpeaker:
    """Lightweight offline TTS wrapper (uses SAPI via pyttsx3 on Windows)."""

    def __init__(self, rate: int = 185):
        self._engine = pyttsx3.init() if pyttsx3 else None
        self._lock = Lock()
        if self._engine:
            try:
                self._engine.setProperty("rate", rate)
            except Exception:
                logger.debug("Failed to set pyttsx3 rate")

    @property
    def available(self) -> bool:
        return self._engine is not None

    def speak_async(self, text: str) -> bool:
        if not self._engine:
            return False
        Thread(target=self._speak, args=(text,), daemon=True).start()
        return True

    def _speak(self, text: str) -> None:
        if not self._engine:
            return
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:  # pragma: no cover - engine runtime errors
                logger.error("pyttsx3 failed to speak text", exc_info=True)
