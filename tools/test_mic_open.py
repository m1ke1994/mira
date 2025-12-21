import pyaudio

IDX = 19  # попробуй потом 13, если нужно

pa = pyaudio.PyAudio()
info = pa.get_device_info_by_index(IDX)
print("Device:", info["name"])
print("Default rate:", info["defaultSampleRate"])
print("Max input channels:", info["maxInputChannels"])
print()

def try_open(rate, ch):
    try:
        s = pa.open(
            format=pyaudio.paInt16,
            channels=ch,
            rate=rate,
            input=True,
            input_device_index=IDX,
            frames_per_buffer=1024,
        )
        s.close()
        print(f"OK: rate={rate} ch={ch}")
        return True
    except Exception as e:
        print(f"FAIL: rate={rate} ch={ch} -> {e}")
        return False

# пробуем самые частые комбинации
for ch in (1, 2):
    for rate in (16000, 44100, 48000):
        try_open(rate, ch)

pa.terminate()
