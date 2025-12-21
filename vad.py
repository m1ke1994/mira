import numpy as np

SILENCE_THRESHOLD = 500  # simple energy threshold (int16 amplitude)


class EnergyVAD:
    def __init__(self, silence_ms: int, frame_ms: int = 30, threshold: float = SILENCE_THRESHOLD):
        self.silence_ms = silence_ms
        self.frame_ms = frame_ms
        self.threshold = threshold

    def is_silence(self, frame: bytes) -> bool:
        samples = np.frombuffer(frame, dtype=np.int16)
        energy = np.mean(np.abs(samples))
        return energy < self.threshold
