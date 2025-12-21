import logging
import signal
import time
from datetime import datetime
from enum import Enum
from queue import Empty, Queue
from threading import Event, Lock, Thread

import numpy as np

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
from alarms.intent_router import IntentRouter
from alarms.manager import AlarmManager
from alarms.sounds import AlarmSoundPlayer, LocalSpeaker
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
        self._tzinfo = datetime.now().astimezone().tzinfo

        self.alarm_sound_player = AlarmSoundPlayer(config.alarm_sound_path)
        self.alarm_manager = AlarmManager(
            storage_path=config.alarms_path,
            sound_player=self.alarm_sound_player,
            check_interval=max(0.2, config.alarm_check_interval_ms / 1000.0),
            default_snooze_minutes=config.alarm_default_snooze_min,
            on_alarm_triggered=self._on_alarm_triggered,
            timezone=self._tzinfo,
        )
        self.local_speaker = LocalSpeaker()
        self.intent_router = IntentRouter(
            alarm_manager=self.alarm_manager,
            transcribe_fn=self._transcribe_for_intents if config.enable_alarm_router else None,
            default_snooze_minutes=config.alarm_default_snooze_min,
        )

    def start(self) -> None:
        self.alarm_manager.start()
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
        self.alarm_manager.shutdown()
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
                now_dt = datetime.now(self._tzinfo)
                if self.intent_router:
                    local_intent = self.intent_router.handle_audio(audio, now=now_dt)
                    if local_intent and local_intent.handled:
                        if local_intent.response_text:
                            self._respond_local_text(local_intent.response_text)
                        self._grace_beeped = False
                        if self.config.conversation_session_seconds > 0:
                            self.grace_until = time.time() + self.config.conversation_session_seconds
                            self._set_state(
                                AssistantState.GRACE_WINDOW if self.wakeword_detector else AssistantState.RECORDING_QUERY
                            )
                            self._grace_logged = False
                            self._grace_beeped = False
                        else:
                            self._set_state(
                                AssistantState.IDLE_LISTENING
                                if self.wakeword_detector
                                else AssistantState.RECORDING_QUERY
                            )
                        continue

                record_end_ts = self.last_record_end
                send_start_ts = time.time()
                first_event_ts = {"t": None}
                first_chunk_ts = {"t": None}
                playback_start_ts = {"t": None}
                playback_end_ts = {"t": None}

                def on_first_event() -> None:
                    if first_event_ts["t"] is None:
                        first_event_ts["t"] = time.time()

                def on_first_chunk() -> None:
                    if first_chunk_ts["t"] is None:
                        first_chunk_ts["t"] = time.time()

                playback_queue: "Queue[Optional[bytes]]" = Queue()
                prebuffer: list[bytes] = []
                buffer_bytes = int(self.config.playback_buffer_ms * self.config.output_target_rate * 2 / 1000)
                buffer_bytes = max(buffer_bytes, 1)

                def playback_worker() -> None:
                    started = False
                    while True:
                        item = playback_queue.get()
                        if item is None:
                            break
                        if not started:
                            prebuffer.append(item)
                            if sum(len(b) for b in prebuffer) >= buffer_bytes:
                                started = True
                                self._set_state(AssistantState.PLAYING)
                                if playback_start_ts["t"] is None:
                                    playback_start_ts["t"] = time.time()
                                for pb in prebuffer:
                                    self.player.play_bytes(pb)
                                prebuffer.clear()
                        else:
                            self.player.play_bytes(item)
                    if not started and prebuffer:
                        # Not enough to reach buffer threshold but got some audio, play it
                        self._set_state(AssistantState.PLAYING)
                        if playback_start_ts["t"] is None:
                            playback_start_ts["t"] = time.time()
                        for pb in prebuffer:
                            self.player.play_bytes(pb)
                    playback_end_ts["t"] = time.time()

                worker = Thread(target=playback_worker, daemon=True)
                worker.start()

                response_stream = self.client.send_audio(
                    audio,
                    on_first_chunk=on_first_chunk,
                    on_first_event=on_first_event,
                    chunk_ms=self.config.out_chunk_ms,
                    first_audio_timeout_ms=self.config.first_audio_timeout_ms,
                )
                received_any = False
                for chunk in response_stream:
                    if not chunk:
                        continue
                    received_any = True
                    playback_queue.put(chunk)
                playback_queue.put(None)
                worker.join()

                record_to_send_ms = (send_start_ts - record_end_ts) * 1000 if record_end_ts else 0
                send_to_first_event_ms = (
                    (first_event_ts["t"] - send_start_ts) * 1000 if first_event_ts["t"] else None
                )
                send_to_first_audio_ms = (
                    (first_chunk_ts["t"] - send_start_ts) * 1000 if first_chunk_ts["t"] else None
                )
                first_audio_to_play_ms = (
                    (playback_start_ts["t"] - first_chunk_ts["t"]) * 1000
                    if first_chunk_ts["t"] and playback_start_ts["t"]
                    else None
                )
                logger.info(
                    "timing_ms record_to_send=%s send_to_first_event=%s send_to_first_audio=%s first_audio_to_play=%s",
                    int(record_to_send_ms),
                    int(send_to_first_event_ms) if send_to_first_event_ms is not None else "n/a",
                    int(send_to_first_audio_ms) if send_to_first_audio_ms is not None else "n/a",
                    int(first_audio_to_play_ms) if first_audio_to_play_ms is not None else "n/a",
                )
                if send_to_first_audio_ms and send_to_first_audio_ms > 4000:
                    logger.warning("Slow first audio chunk from Gemini: %.0f ms", send_to_first_audio_ms)
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

    def _respond_local_text(self, text: str, allow_beep: bool = True) -> None:
        logger.info("Local response: %s", text)
        spoken = False
        if self.local_speaker and self.local_speaker.available:
            spoken = self.local_speaker.speak_async(text)
        if not spoken and allow_beep:
            beep()

    def _transcribe_for_intents(self, audio_pcm: bytes) -> str | None:
        try:
            return self.client.transcribe_text(
                audio_pcm,
                model=self.config.alarm_transcribe_model,
                prompt="Распознай голосовую команду пользователя и верни только текст без комментариев.",
            )
        except Exception as exc:
            logger.error("Intent transcription failed: %s", exc)
            return None

    def _on_alarm_triggered(self, alarm) -> None:
        try:
            when = alarm.fire_at.astimezone(self._tzinfo).strftime("%H:%M")
        except Exception:
            when = "сейчас"
        message = f"Саша, будильник! {when}."
        self._respond_local_text(message, allow_beep=False)


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

    if config.enable_alarm_router:
        logger.info("Alarm router enabled (storage=%s)", config.alarms_path)

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
