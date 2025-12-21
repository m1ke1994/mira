from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from .manager import AlarmManager
from .parser import parse_alarm_request

logger = logging.getLogger(__name__)


@dataclass
class IntentResult:
    handled: bool
    response_text: Optional[str] = None
    send_to_gemini: bool = False
    action: Optional[str] = None
    transcript: Optional[str] = None


class IntentRouter:
    def __init__(
        self,
        alarm_manager: AlarmManager,
        transcribe_fn: Optional[Callable[[bytes], Optional[str]]] = None,
        default_label: str = "Будильник",
        default_snooze_minutes: int = 5,
    ):
        self.alarm_manager = alarm_manager
        self.transcribe_fn = transcribe_fn
        self.default_label = default_label
        self.default_snooze_minutes = default_snooze_minutes

    def handle_audio(self, audio_pcm: bytes, now: datetime) -> Optional[IntentResult]:
        if not self.transcribe_fn:
            return None
        transcript = self.transcribe_fn(audio_pcm)
        if not transcript:
            return None
        logger.info("Transcript for intent routing: %s", transcript)
        return self.handle_text(transcript, now=now, transcript=transcript)

    def handle_text(self, text: str, now: datetime, transcript: Optional[str] = None) -> Optional[IntentResult]:
        parsed = parse_alarm_request(text, now=now, allow_stop_without_keyword=self.alarm_manager.is_ringing)
        if not parsed:
            return None
        logger.info("Alarm intent detected: %s", parsed)

        if parsed.action == "unknown":
            return IntentResult(handled=True, response_text=parsed.error, action=parsed.action, transcript=transcript)

        if parsed.action == "list":
            alarms = self.alarm_manager.list_alarms()
            if not alarms:
                resp = "Пока нет активных будильников."
            else:
                parts = []
                for idx, alarm in enumerate(alarms, start=1):
                    parts.append(f"{idx}) {format_alarm_time(alarm.fire_at, now)} — {alarm.label}")
                resp = "Твои будильники:\n" + "\n".join(parts)
            return IntentResult(handled=True, response_text=resp, action="list", transcript=transcript)

        if parsed.action == "remove":
            removed = (
                self.alarm_manager.remove_alarm_by_index(parsed.remove_index)
                if parsed.remove_index
                else self.alarm_manager.remove_nearest()
            )
            if removed:
                resp = f"Убрала будильник на {format_alarm_time(removed.fire_at, now)}."
            else:
                resp = "Не нашла такой будильник."
            return IntentResult(handled=True, response_text=resp, action="remove", transcript=transcript)

        if parsed.action == "stop":
            current = self.alarm_manager.stop_ringing()
            if current:
                resp = "Остановила будильник."
            else:
                resp = "Сейчас ничего не звенит."
            return IntentResult(handled=True, response_text=resp, action="stop", transcript=transcript)

        if parsed.action == "snooze":
            minutes = parsed.snooze_minutes or self.default_snooze_minutes
            new_alarm = self.alarm_manager.snooze(minutes)
            if new_alarm:
                resp = f"Отложила на {minutes} минут, прозвенит в {format_alarm_time(new_alarm.fire_at, now)}."
            else:
                resp = "Сейчас ничего не звенит, нечего откладывать."
            return IntentResult(handled=True, response_text=resp, action="snooze", transcript=transcript)

        if parsed.action == "add":
            if parsed.delta_minutes:
                fire_at = now + timedelta(minutes=parsed.delta_minutes)
            else:
                fire_at = parsed.fire_at
            if not fire_at:
                return IntentResult(
                    handled=True,
                    response_text="Не поняла время будильника, повтори пожалуйста.",
                    action="add",
                    transcript=transcript,
                )
            try:
                alarm = self.alarm_manager.add_alarm(fire_at, label=parsed.label or self.default_label)
                resp = f"Окей, будильник поставила на {format_alarm_time(alarm.fire_at, now)}."
            except ValueError:
                resp = "Это время уже прошло. Скажи время позже текущего."
            return IntentResult(handled=True, response_text=resp, action="add", transcript=transcript)

        return IntentResult(handled=True, response_text=None, action=parsed.action, transcript=transcript)


def format_alarm_time(dt: datetime, now: datetime) -> str:
    day_prefix = ""
    if dt.date() == now.date():
        day_prefix = "сегодня "
    elif dt.date() == now.date() + timedelta(days=1):
        day_prefix = "завтра "
    elif dt.date() == now.date() + timedelta(days=2):
        day_prefix = "послезавтра "
    time_part = dt.strftime("%H:%M")
    if dt.date() != now.date() and not day_prefix:
        day_prefix = dt.strftime("%d.%m ")  # Explicit date
    return f"{day_prefix}{time_part}"
