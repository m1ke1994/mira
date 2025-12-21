import time

from audio_io import FrameAdapter, MicrophoneStream, create_pyaudio, get_input_device
from config import load_config, setup_logging
from wakeword import init_wakeword_detector


def main():
    cfg = load_config(require_gemini=False)
    setup_logging("INFO")
    pa = create_pyaudio()
    device = get_input_device(pa, cfg.input_device_index)
    detector = init_wakeword_detector(cfg.picovoice_access_key, cfg.wakeword_keyword_path, cfg.wakeword_sensitivity)
    if not detector:
        print("Wakeword detector not available")
        return

    with MicrophoneStream(pa, device) as mic:
        adapter = FrameAdapter(device.rate, device.channels, detector.sample_rate, detector.frame_length)
        print("Say the wakeword...")
        detector.wait_for_wakeword(mic, adapter)
        print("detected!")

    pa.terminate()


if __name__ == "__main__":
    main()
