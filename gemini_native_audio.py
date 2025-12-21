import asyncio
import logging
import threading
from queue import Queue, Empty
from typing import Callable, Generator, Optional

import google.genai as genai
from google.genai import types

logger = logging.getLogger(__name__)


class GeminiNativeAudioClient:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        system_prompt: str,
        voice_name: Optional[str],
        input_sample_rate: int,
        output_sample_rate: int,
        lang: str = "ru",
    ):
        self.client = genai.Client(api_key=api_key)
        if model_name == "gemini-2.5-flash-native-audio-dialog":
            logger.warning(
                "Model %s not supported by Live API; falling back to gemini-2.5-flash-native-audio-preview-12-2025",
                model_name,
            )
            model_name = "gemini-2.5-flash-native-audio-preview-12-2025"
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.voice_name = voice_name
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.lang = lang
        logger.info(
            "Gemini client ready (model=%s, output_rate=%s, voice=%s)",
            model_name,
            output_sample_rate,
            voice_name or "default",
        )

    def send_audio(
        self,
        audio_pcm: bytes,
        on_first_chunk: Optional[Callable[[], None]] = None,
        queue_timeout_secs: float = 20.0,
    ) -> Generator[bytes, None, None]:
        """Send a single utterance to Live API and yield PCM audio chunks as they arrive."""
        mime_type = f"audio/pcm;rate={self.input_sample_rate}"
        speech_config = None
        if self.voice_name:
            speech_config = types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice_name)
                )
            )

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            speech_config=speech_config,
            system_instruction=self.system_prompt or None,
        )

        queue: "Queue[Optional[bytes]]" = Queue()

        async def _run() -> None:
            try:
                async with self.client.aio.live.connect(model=self.model_name, config=config) as session:
                    await session.send_realtime_input(audio=types.Blob(mime_type=mime_type, data=audio_pcm))
                    await session.send_realtime_input(audio_stream_end=True)
                    async for message in session.receive():
                        server_content = getattr(message, "server_content", None)
                        if server_content and server_content.model_turn:
                            for part in server_content.model_turn.parts or []:
                                inline = getattr(part, "inline_data", None)
                                if inline and getattr(inline, "data", None):
                                    if on_first_chunk:
                                        try:
                                            on_first_chunk()
                                        except Exception:  # pragma: no cover - callback safety
                                            logger.debug("on_first_chunk callback failed", exc_info=True)
                                    queue.put(inline.data)
                        if server_content and server_content.generation_complete:
                            break
            except Exception as exc:  # pragma: no cover - transport failures
                logger.error("Failed to call Gemini: %s", exc)
            finally:
                queue.put(None)

        threading.Thread(target=lambda: asyncio.run(_run()), daemon=True).start()

        received_any = False
        while True:
            try:
                chunk = queue.get(timeout=queue_timeout_secs)
            except Empty:
                logger.error("Timed out waiting for Gemini audio")
                break
            if chunk is None:
                break
            received_any = True
            yield chunk
        if not received_any:
            logger.warning("No audio received from Gemini response stream")
