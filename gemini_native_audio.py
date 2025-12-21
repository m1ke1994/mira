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
        on_first_event: Optional[Callable[[], None]] = None,
        chunk_ms: int = 40,
        first_audio_timeout_ms: int = 12000,
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
        chunk_size = max(2, int(self.input_sample_rate * chunk_ms / 1000) * 2)

        async def _run() -> None:
            try:
                async with self.client.aio.live.connect(model=self.model_name, config=config) as session:
                    # Stream audio in small chunks
                    for idx in range(0, len(audio_pcm), chunk_size):
                        part = audio_pcm[idx : idx + chunk_size]
                        await session.send_realtime_input(audio=types.Blob(mime_type=mime_type, data=part))
                    # Explicitly end user turn
                    await session.send_realtime_input(audio_stream_end=True)

                    async for message in session.receive():
                        server_content = getattr(message, "server_content", None)
                        if server_content and on_first_event:
                            try:
                                on_first_event()
                            except Exception:
                                logger.debug("on_first_event callback failed", exc_info=True)
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
        timeout_secs = max(1.0, first_audio_timeout_ms / 1000)
        while True:
            try:
                chunk = queue.get(timeout=timeout_secs)
            except Empty:
                logger.error("Timed out waiting for Gemini audio")
                break
            if chunk is None:
                break
            received_any = True
            yield chunk
        if not received_any:
            logger.warning("No audio received from Gemini response stream")
