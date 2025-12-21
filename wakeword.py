import logging
from pathlib import Path
from threading import Event
from typing import Optional

import numpy as np
import pvporcupine

from audio_io import FrameAdapter, MicrophoneStream

logger = logging.getLogger(__name__)


class PorcupineDetector:
    def __init__(
        self,
        access_key: str,
        keyword_path: Path,
        sensitivity: float,
    ):
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keyword_paths=[str(keyword_path)],
            sensitivities=[sensitivity],
        )
        self.sample_rate = self.porcupine.sample_rate
        self.frame_length = self.porcupine.frame_length
        logger.info(
            "Porcupine initialized (rate=%s, frame=%s, keyword=%s)",
            self.sample_rate,
            self.frame_length,
            keyword_path,
        )

    def close(self) -> None:
        self.porcupine.delete()

    def wait_for_wakeword(self, mic: MicrophoneStream, adapter: FrameAdapter) -> bool:
        return self.wait_for_wakeword_until(mic, adapter, None)

    def wait_for_wakeword_until(
        self,
        mic: MicrophoneStream,
        adapter: FrameAdapter,
        stop_event: Optional[Event],
    ) -> bool:
        logger.info("Listening for wakeword...")
        while True:
            if stop_event and stop_event.is_set():
                return False
            chunk = mic.read_chunk()
            frames = adapter.process(chunk)
            for frame in frames:
                pcm = np.frombuffer(frame, dtype=np.int16)
                result = self.porcupine.process(pcm)
                if result >= 0:
                    logger.info("Wakeword detected (index %s)", result)
                    return True


def init_wakeword_detector(
    access_key: Optional[str],
    keyword_path: Optional[Path],
    sensitivity: float,
) -> Optional[PorcupineDetector]:
    if not access_key or not keyword_path or not keyword_path.exists():
        logger.warning("Wakeword model not found, using push-to-talk")
        return None
    try:
        return PorcupineDetector(access_key, keyword_path, sensitivity)
    except Exception as exc:  # pragma: no cover - initialization failure
        logger.error("Failed to initialize Porcupine: %s", exc)
        return None
