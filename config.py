import logging
import logging.handlers
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _get_env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip() in {"1", "true", "True", "yes", "YES", "y"}


def _get_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _get_env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float") from exc


@dataclass
class Config:
    gemini_api_key: str
    gemini_model_name: str
    system_prompt: str
    voice_name: Optional[str]
    input_target_rate: int
    output_target_rate: int
    input_device_index: Optional[int]
    max_utterance_seconds: int
    silence_ms: int
    no_speech_timeout_ms: int
    vad_grace_ms: int
    min_record_ms: int
    max_record_ms: int
    preroll_ms: int
    min_audio_ms: int
    min_voice_ms: int
    rms_speech_start: float
    rms_speech_frames: int
    out_chunk_ms: int
    playback_buffer_ms: int
    first_audio_timeout_ms: int
    conversation_session_seconds: int
    debug: bool
    log_level: str
    lang: str
    wakeword_keyword_path: Optional[Path]
    wakeword_sensitivity: float
    picovoice_access_key: Optional[str]
    vad_threshold: float
    alarms_path: Path
    alarm_sound_path: Path
    alarm_check_interval_ms: int
    alarm_default_snooze_min: int
    enable_alarm_router: bool
    alarm_transcribe_model: str


DEFAULT_SYSTEM_PROMPT = (
    "Ты голосовой ассистент Мира. Отвечай по-русски, кратко и дружелюбно. "
    "Используй женский голос, если доступно. Не добавляй лишних пояснений."
)


def load_config(env_path: Optional[Path] = None, require_gemini: bool = True) -> Config:
    if env_path is None:
        env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)

    gemini_api_key = os.getenv("GEMINI_API_KEY") or ""
    if require_gemini and not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is required in .env")

    gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-native-audio-dialog")

    system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    voice_name = os.getenv("VOICE_NAME")
    input_target_rate = _get_env_int("INPUT_TARGET_RATE", 16000)
    output_target_rate = _get_env_int("OUTPUT_TARGET_RATE", 24000)
    input_device_index_env = os.getenv("INPUT_DEVICE_INDEX")
    input_device_index = int(input_device_index_env) if input_device_index_env else None
    max_utterance_seconds = _get_env_int("MAX_UTTERANCE_SECONDS", 15)
    silence_ms = _get_env_int("SILENCE_MS", 1100)
    no_speech_timeout_ms = _get_env_int("NO_SPEECH_TIMEOUT_MS", 3000)
    vad_grace_ms = _get_env_int("VAD_GRACE_MS", 900)
    min_record_ms = _get_env_int("MIN_RECORD_MS", 2500)
    max_record_ms = _get_env_int("MAX_RECORD_MS", 12000)
    preroll_ms = _get_env_int("PREROLL_MS", 500)
    min_audio_ms = _get_env_int("MIN_AUDIO_MS", 400)
    min_voice_ms = _get_env_int("MIN_VOICE_MS", 300)
    rms_speech_start = _get_env_float("RMS_SPEECH_START", 80.0)
    rms_speech_frames = _get_env_int("RMS_SPEECH_FRAMES", 2)
    out_chunk_ms = _get_env_int("OUT_CHUNK_MS", 40)
    playback_buffer_ms = _get_env_int("PLAYBACK_BUFFER_MS", 250)
    first_audio_timeout_ms = _get_env_int("FIRST_AUDIO_TIMEOUT_MS", 12000)
    conversation_session_seconds = _get_env_int("CONVERSATION_SESSION_SECONDS", 10)
    debug = _get_env_bool("DEBUG", False)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    lang = os.getenv("LANG", "ru")
    vad_threshold = _get_env_float("VAD_THRESHOLD", 80.0)
    alarms_path = Path(os.getenv("ALARM_STORAGE_PATH", "data/alarms.json"))
    alarm_sound_path = Path(os.getenv("ALARM_SOUND_PATH", "data/alarm.wav"))
    alarm_check_interval_ms = _get_env_int("ALARM_CHECK_INTERVAL_MS", 800)
    alarm_default_snooze_min = _get_env_int("ALARM_DEFAULT_SNOOZE_MIN", 5)
    enable_alarm_router = _get_env_bool("ENABLE_ALARM_ROUTER", True)
    alarm_transcribe_model = os.getenv("ALARM_TRANSCRIBE_MODEL", "gemini-1.5-flash")

    wakeword_keyword_env = os.getenv("WAKEWORD_KEYWORD_PATH")
    wakeword_keyword_path = Path(wakeword_keyword_env) if wakeword_keyword_env else None
    if wakeword_keyword_path and not wakeword_keyword_path.exists():
        logging.warning("WAKEWORD_KEYWORD_PATH is set but file is missing: %s", wakeword_keyword_path)
    wakeword_sensitivity = float(os.getenv("WAKEWORD_SENSITIVITY", "0.6"))
    picovoice_access_key = os.getenv("PICOVOICE_ACCESS_KEY")

    return Config(
        gemini_api_key=gemini_api_key,
        gemini_model_name=gemini_model_name,
        system_prompt=system_prompt,
        voice_name=voice_name,
        input_target_rate=input_target_rate,
        output_target_rate=output_target_rate,
        input_device_index=input_device_index,
        max_utterance_seconds=max_utterance_seconds,
        silence_ms=silence_ms,
        no_speech_timeout_ms=no_speech_timeout_ms,
        vad_grace_ms=vad_grace_ms,
        min_record_ms=min_record_ms,
        max_record_ms=max_record_ms,
        preroll_ms=preroll_ms,
        min_audio_ms=min_audio_ms,
        min_voice_ms=min_voice_ms,
        rms_speech_start=rms_speech_start,
        rms_speech_frames=rms_speech_frames,
        out_chunk_ms=out_chunk_ms,
        playback_buffer_ms=playback_buffer_ms,
        first_audio_timeout_ms=first_audio_timeout_ms,
        conversation_session_seconds=conversation_session_seconds,
        debug=debug,
        log_level=log_level,
        lang=lang,
        wakeword_keyword_path=wakeword_keyword_path,
        wakeword_sensitivity=wakeword_sensitivity,
        picovoice_access_key=picovoice_access_key,
        vad_threshold=vad_threshold,
        alarms_path=alarms_path,
        alarm_sound_path=alarm_sound_path,
        alarm_check_interval_ms=alarm_check_interval_ms,
        alarm_default_snooze_min=alarm_default_snooze_min,
        enable_alarm_router=enable_alarm_router,
        alarm_transcribe_model=alarm_transcribe_model,
    )


def setup_logging(log_level: str = "INFO") -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    log_path = logs_dir / "mira.log"
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=[file_handler, console_handler],
    )
