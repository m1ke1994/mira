import pyaudio

pa = pyaudio.PyAudio()

def try_open(idx):
    info = pa.get_device_info_by_index(idx)
    if info.get("maxInputChannels", 0) <= 0:
        return False
    name = info["name"]
    for rate in (int(info["defaultSampleRate"]), 44100, 48000, 16000):
        for ch in (1, min(2, int(info["maxInputChannels"]))):
            try:
                s = pa.open(format=pyaudio.paInt16, channels=ch, rate=rate,
                            input=True, input_device_index=idx,
                            frames_per_buffer=1024)
                s.close()
                print(f"OK  idx={idx} rate={rate} ch={ch}  name={name}")
                return True
            except Exception:
                pass
    print(f"FAIL idx={idx} name={name}")
    return False

print("Testing all input devices...\n")
for i in range(pa.get_device_count()):
    try_open(i)

pa.terminate()
