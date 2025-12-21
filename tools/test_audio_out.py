import math
import time

import numpy as np
import pyaudio


def main():
    pa = pyaudio.PyAudio()
    rate = 24000
    duration = 2.0
    tone_hz = 440.0
    samples = (
        0.2
        * np.sin(2 * math.pi * np.arange(int(rate * duration)) * tone_hz / rate)
    ).astype(np.float32)

    stream = pa.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)
    print("Playing test tone...")
    stream.write(samples.tobytes())
    stream.stop_stream()
    stream.close()
    pa.terminate()
    time.sleep(0.1)


if __name__ == "__main__":
    main()
