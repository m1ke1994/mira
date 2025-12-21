import pyaudio

pa = pyaudio.PyAudio()

print("\n=== INPUT DEVICES (Microphones) ===\n")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info.get("maxInputChannels", 0) > 0:
        print(
            f"[IN ] Index {i}: {info['name']} | "
            f"rate={int(info['defaultSampleRate'])} | "
            f"channels={info['maxInputChannels']}"
        )

print("\n=== OUTPUT DEVICES (Speakers) ===\n")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info.get("maxOutputChannels", 0) > 0:
        print(
            f"[OUT] Index {i}: {info['name']} | "
            f"rate={int(info['defaultSampleRate'])} | "
            f"channels={info['maxOutputChannels']}"
        )

pa.terminate()
