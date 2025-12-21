import logging
import signal
import time
import numpy as np
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread

from audio_io import (
    AudioPlayer,
    FrameAdapter,
    InputDeviceInfo,
    MicrophoneStream,
    RecordingResult,
    create_pyaudio,
    get_input_device,
    record_utterance,
)
from config import Config, load_config, setup_logging
from gemini_native_audio import GeminiNativeAudioClient
from vad import EnergyVAD
from wakeword import PorcupineDetector, init_wakeword_detector

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

logger = logging.getLogger("mira")


class AssistantState(Enum):
    IDLE_LISTENING = "idle_listening"
    RECORDING_QUERY = "recording_query"
    SENDING = "sending"
    PLAYING = "playing"
    GRACE_WINDOW = "grace_window"


def beep() -> None:
    if winsound:
        try:
            winsound.Beep(880, 120)
            return
        except RuntimeError:
            logger.debug("winsound.Beep failed, falling back to log")
    logger.info("Beep")


def graceful_exit(signum, frame) -> None:  # pragma: no cover - signal handler
    logger.info("Shutting down (signal %s)", signum)
    raise KeyboardInterrupt()


signal.signal(signal.SIGINT, graceful_exit)


class AssistantRuntime:
    def __init__(
        self,
        config: Config,
        pa,
        device: InputDeviceInfo,
        player: AudioPlayer,
        client: GeminiNativeAudioClient,
        vad: EnergyVAD,
        wakeword_detector: PorcupineDetector | None,
    ):
        self.config = config
        self.pa = pa
        self.device = device
        self.player = player
        self.client = client
        self.vad = vad
        self.wakeword_detector = wakeword_detector

        self.state_lock = Lock()
        self.state = AssistantState.IDLE_LISTENING if wakeword_detector else AssistantState.RECORDING_QUERY
        self.stop_event = Event()
        self.grace_until = 0.0
        self.audio_queue: "Queue[bytes]" = Queue()
        self.listener_thread: Thread | None = None
        self.worker_thread: Thread | None = None
        self._grace_logged = False
        self._grace_beeped = False
        self.last_record_end = time.time()

    def start(self) -> None:
        self.listener_thread = Thread(target=self._listen_loop, name="listener", daemon=True)
        self.worker_thread = Thread(target=self._process_loop, name="processor", daemon=True)
        self.listener_thread.start()
        self.worker_thread.start()

    def shutdown(self) -> None:
        self.stop_event.set()
        if self.listener_thread:
            self.listener_thread.join(timeout=2)
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        self.player.close()
        self.pa.terminate()

    def _set_state(self, state: AssistantState) -> None:
        with self.state_lock:
            if self.state != state:
                logger.info("State -> %s", state.value)
            self.state = state

    def _listen_loop(self) -> None:
        frame_ms = 30
        wake_adapter = self._build_wakeword_adapter() if self.wakeword_detector else None

        while not self.stop_event.is_set():
            try:
                with MicrophoneStream(self.pa, self.device) as mic:
                    self._listen_with_stream(mic, wake_adapter, frame_ms)
            except OSError as exc:
                logger.error("Audio input error: %s. Try another INPUT_DEVICE_INDEX.", exc)
                time.sleep(1)
            except Exception as exc:
                logger.error("Listener loop error: %s", exc)
                time.sleep(0.2)

    def _listen_with_stream(
        self,
        mic: MicrophoneStream,
        wake_adapter: FrameAdapter | None,
        frame_ms: int,
    ) -> None:
        while not self.stop_event.is_set():
            if self.state in (AssistantState.SENDING, AssistantState.PLAYING):
                time.sleep(0.05)
                continue

            if self.state == AssistantState.GRACE_WINDOW:
                remaining = self.grace_until - time.time()
                if remaining <= 0:
                    self._grace_logged = False
                    self._grace_beeped = False
                    self._set_state(
                        AssistantState.IDLE_LISTENING if self.wakeword_detector else AssistantState.RECORDING_QUERY
                    )
                    continue
                if not self._grace_logged:
                    logger.info("Grace window active for %.1fs", remaining)
                    self._grace_logged = True
                if self._grace_beeped:
                    if not self._has_speech_gate(mic, frame_ms):
                        logger.info("grace_window trigger skipped (reason=speech_gate)")
                        time.sleep(0.05)
                        continue
                else:
                    if not self._has_speech_gate(mic, frame_ms):
                        time.sleep(0.05)
                        continue
                self._start_recording(
                    mic,
                    frame_ms,
                    stay_in_grace=True,
                    play_beep=not self._grace_beeped,
                )
                self._grace_beeped = True
                continue

            if self.wakeword_detector:
                if wake_adapter is None:
                    logger.error("Wakeword adapter missing")
                    time.sleep(0.5)
                    continue
                detected = self.wakeword_detector.wait_for_wakeword_until(mic, wake_adapter, self.stop_event)
                if not detected:
                    continue
            else:
                input("Press Enter to talk (push-to-talk fallback)...")

            self._start_recording(mic, frame_ms, stay_in_grace=False)

    def _start_recording(
        self,
        mic: MicrophoneStream,
        frame_ms: int,
        stay_in_grace: bool,
        play_beep: bool = True,
    ) -> None:
        if self.stop_event.is_set():
            return
        if play_beep:
            beep()
        self._set_state(AssistantState.RECORDING_QUERY)
        record_adapter = self._build_recording_adapter(frame_ms)
        audio = record_utterance(
            mic=mic,
            adapter=record_adapter,
            max_seconds=self.config.max_utterance_seconds,
            vad=self.vad,
            frame_ms=frame_ms,
            stop_event=self.stop_event,
            start_timeout_ms=self.config.no_speech_timeout_ms,
            vad_grace_ms=self.config.vad_grace_ms,
            min_record_ms=self.config.min_record_ms,
            max_record_ms=self.config.max_record_ms,
            preroll_ms=self.config.preroll_ms,
            min_audio_ms=self.config.min_audio_ms,
            rms_speech_start=self.config.rms_speech_start,
            rms_speech_frames=self.config.rms_speech_frames,
        )
        self.last_record_end = time.time()
        if not audio.audio:
            logger.warning("No audio captured (reason=%s)", audio.reason)
            if stay_in_grace:
                self._set_state(AssistantState.GRACE_WINDOW)
            else:
                self._set_state(
                    AssistantState.IDLE_LISTENING if self.wakeword_detector else AssistantState.RECORDING_QUERY
                )
            return
        if (
            not audio.speech_detected
            and audio.max_rms < self.config.rms_speech_start
            and audio.voice_ms < self.config.min_voice_ms
        ):
            logger.info(
                "send skipped (reason=silence_gate max_rms=%.1f voice_ms=%.0f duration_ms=%.0f)",
                audio.max_rms,
                audio.voice_ms,
                audio.duration_ms,
            )
            if stay_in_grace:
                self._set_state(AssistantState.GRACE_WINDOW)
            else:
                self._set_state(
                    AssistantState.IDLE_LISTENING if self.wakeword_detector else AssistantState.RECORDING_QUERY
                )
            return
        logger.info(
            "Recorded %d bytes (speech_detected=%s, reason=%s)",
            len(audio.audio),
            audio.speech_detected,
            audio.reason,
        )
        self.audio_queue.put(audio.audio)
        self._set_state(AssistantState.SENDING)

    def _process_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                t_send_start = time.time()
                first_chunk_time = {"t": None}

                def on_first_chunk() -> None:
                    if first_chunk_time["t"] is None:
                        first_chunk_time["t"] = time.time()

                response_stream = self.client.send_audio(audio, on_first_chunk=on_first_chunk)
                playback_started = False
                received_any = False
                for chunk in response_stream:
                    if not chunk:
                        continue
                    received_any = True
                    if not playback_started:
                        playback_started = True
                        self._set_state(AssistantState.PLAYING)
                        t_playback_start = time.time()
                    self.player.play_bytes(chunk)
                if not received_any:
                    logger.warning("No audio chunks received from Gemini")
                else:
                    logger.info("Response playback finished")
                record_to_send_ms = (
                    (t_send_start - self.last_record_end) * 1000 if self.last_record_end else 0
                )
                send_to_first_ms = (
                    (first_chunk_time["t"] - t_send_start) * 1000 if first_chunk_time["t"] else None
                )
                first_to_play_ms = (
                    (t_playback_start - first_chunk_time["t"]) * 1000
                    if first_chunk_time["t"] and playback_started
                    else None
                )
                logger.info(
                    "timing_ms record_to_send=%s send_to_first=%s first_to_play=%s",
                    int(record_to_send_ms),
                    int(send_to_first_ms) if send_to_first_ms is not None else "n/a",
                    int(first_to_play_ms) if first_to_play_ms is not None else "n/a",
                )
                if send_to_first_ms and send_to_first_ms > 4000:
                    logger.warning("Slow first audio chunk from Gemini: %.0f ms", send_to_first_ms)
                if self.config.conversation_session_seconds > 0:
                    self.grace_until = time.time() + self.config.conversation_session_seconds
                    self._set_state(
                        AssistantState.GRACE_WINDOW if self.wakeword_detector else AssistantState.RECORDING_QUERY
                    )
                    self._grace_logged = False
                    self._grace_beeped = False
                else:
                    self._set_state(
                        AssistantState.IDLE_LISTENING if self.wakeword_detector else AssistantState.RECORDING_QUERY
                    )
            except Exception as exc:  # pragma: no cover - network/IO failures
                logger.error("Failed to process audio: %s", exc)
                self._set_state(
                    AssistantState.IDLE_LISTENING if self.wakeword_detector else AssistantState.RECORDING_QUERY
                )
                self._grace_beeped = False

    def _build_recording_adapter(self, frame_ms: int) -> FrameAdapter:
        return FrameAdapter(
            input_rate=self.device.rate,
            input_channels=self.device.channels,
            target_rate=self.config.input_target_rate,
            target_frame_len=int(self.config.input_target_rate * frame_ms / 1000),
        )

    def _build_wakeword_adapter(self) -> FrameAdapter:
        if not self.wakeword_detector:
            raise RuntimeError("Wakeword detector not configured")
        if self.device.rate != self.wakeword_detector.sample_rate:
            logger.info("Wakeword resample %s -> %s Hz", self.device.rate, self.wakeword_detector.sample_rate)
        return FrameAdapter(
            input_rate=self.device.rate,
            input_channels=self.device.channels,
            target_rate=self.wakeword_detector.sample_rate,
            target_frame_len=self.wakeword_detector.frame_length,
        )

    def _has_speech_gate(self, mic: MicrophoneStream, frame_ms: int, window_ms: int = 600) -> bool:
        adapter = self._build_recording_adapter(frame_ms)
        deadline = time.time() + window_ms / 1000.0
        max_rms = 0.0
        while time.time() < deadline and not self.stop_event.is_set():
            chunk = mic.read_chunk()
            frames = adapter.process(chunk)
            for frame in frames:
                rms = float(np.sqrt(np.mean(np.frombuffer(frame, dtype=np.int16).astype(np.float64) ** 2)))
                if rms > max_rms:
                    max_rms = rms
                if rms >= self.config.rms_speech_start:
                    return True
        logger.info(
            "speech_gate false (max_rms=%.1f threshold=%.1f window_ms=%s)",
            max_rms,
            self.config.rms_speech_start,
            window_ms,
        )
        return False


def main() -> None:
    config = load_config()
    setup_logging(config.log_level)
    logger.info("Starting Mira assistant")
    pa = create_pyaudio()
    device = get_input_device(pa, config.input_device_index)
    if device.rate != config.input_target_rate:
        logger.info("Input will be resampled %s -> %s Hz", device.rate, config.input_target_rate)
    else:
        logger.info("Input sample rate matches target %s Hz", config.input_target_rate)
    vad_threshold = config.vad_threshold
    if vad_threshold <= 1.0:
        vad_threshold = vad_threshold * 1000.0
    logger.info("VAD threshold set to %.1f (RMS amplitude)", vad_threshold)
    player = AudioPlayer(pa, config.output_target_rate)
    vad = EnergyVAD(silence_ms=config.silence_ms, frame_ms=30, threshold=vad_threshold)
    wakeword_detector = init_wakeword_detector(
        config.picovoice_access_key, config.wakeword_keyword_path, config.wakeword_sensitivity
    )
    mode = "wakeword" if wakeword_detector else "ptt"
    logger.info("Mode: %s", mode)

    client = GeminiNativeAudioClient(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model_name,
        system_prompt=config.system_prompt,
        voice_name=config.voice_name,
        input_sample_rate=config.input_target_rate,
        output_sample_rate=config.output_target_rate,
        lang=config.lang,
    )

    runtime = AssistantRuntime(config, pa, device, player, client, vad, wakeword_detector)
    runtime.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    main()
