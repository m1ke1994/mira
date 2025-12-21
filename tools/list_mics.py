import pyaudio


def main():
    pa = pyaudio.PyAudio()
    count = pa.get_device_count()
    print("Input devices:")
    for i in range(count):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            print(
                f"[{i}] {info.get('name')} "
                f"rate={int(info.get('defaultSampleRate', 0))} "
                f"channels={int(info.get('maxInputChannels', 0))}"
            )
    pa.terminate()


if __name__ == "__main__":
    main()
