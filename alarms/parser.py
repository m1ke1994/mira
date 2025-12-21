from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}

NUMBER_WORDS = {
    "первый": 1,
    "первую": 1,
    "первого": 1,
    "второй": 2,
    "вторую": 2,
    "третьий": 3,
    "третью": 3,
    "четвертый": 4,
    "четвертую": 4,
    "пятый": 5,
    "пятую": 5,
}


@dataclass
class AlarmCommand:
    action: str
    fire_at: Optional[datetime] = None
    delta_minutes: Optional[int] = None
    remove_index: Optional[int] = None
    remove_nearest: bool = False
    snooze_minutes: Optional[int] = None
    label: Optional[str] = None
    error: Optional[str] = None
    raw_text: str = ""
    parsed_text: Optional[str] = None


def parse_alarm_request(
    text: str,
    now: Optional[datetime] = None,
    allow_stop_without_keyword: bool = False,
) -> Optional[AlarmCommand]:
    """Parse a Russian alarm request into a structured command."""

    now = now or datetime.now().astimezone()
    cleaned = text.strip()
    lower = cleaned.lower()
    mentions_alarm = any(
        keyword in lower for keyword in ("будильник", "разбуди", "подъем", "подъём", "будет меня")
    )

    if not mentions_alarm and not allow_stop_without_keyword:
        # Still allow direct snooze/stop when alarm is ringing via caller flag.
        basic_stop_words = ("стоп", "останов", "выключи", "отключи")
        basic_snooze_words = ("отложи", "подремл", "snooze")
        if not any(w in lower for w in basic_stop_words + basic_snooze_words):
            return None

    # Snooze: "отложи на 5 минут"
    if any(word in lower for word in ("отложи", "подрем", "snooze")):
        minutes = _extract_minutes(lower)
        return AlarmCommand(
            action="snooze",
            snooze_minutes=minutes,
            raw_text=cleaned,
            parsed_text=f"snooze_{minutes or 'default'}",
        )

    # Stop: "стоп будильник"
    if any(word in lower for word in ("стоп", "останов", "выключи", "отключи")):
        return AlarmCommand(action="stop", raw_text=cleaned, parsed_text="stop")

    # List: "покажи список будильников"
    if mentions_alarm and any(word in lower for word in ("список", "покажи", "какие", "есть")):
        return AlarmCommand(action="list", raw_text=cleaned, parsed_text="list")

    # Remove: "удали второй будильник" / "отмени ближайший будильник"
    if mentions_alarm and any(word in lower for word in ("удали", "отмени", "сотри", "отключи")):
        index = _extract_index(lower)
        remove_nearest = "ближай" in lower or "перв" in lower
        return AlarmCommand(
            action="remove",
            remove_index=index,
            remove_nearest=remove_nearest or index is None,
            raw_text=cleaned,
            parsed_text=f"remove_{index or 'nearest'}",
        )

    # Add: explicit time or relative
    if mentions_alarm or "через" in lower or "разбуди" in lower:
        rel_minutes = _extract_relative_minutes(lower)
        if rel_minutes:
            return AlarmCommand(
                action="add",
                delta_minutes=rel_minutes,
                raw_text=cleaned,
                parsed_text=f"in_{rel_minutes}_minutes",
            )

        fire_at = _extract_absolute_time(lower, now)
        if fire_at:
            return AlarmCommand(action="add", fire_at=fire_at, raw_text=cleaned, parsed_text="at_time")
        return AlarmCommand(
            action="unknown",
            error="Не поняла время, повтори пожалуйста.",
            raw_text=cleaned,
            parsed_text="unknown",
        )

    return None


def _extract_minutes(text: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*мин", text)
    if match:
        return int(match.group(1))
    return None


def _extract_relative_minutes(text: str) -> Optional[int]:
    minute_match = re.search(r"через\s+(\d+)\s*мин", text)
    hour_match = re.search(r"через\s+(\d+)\s*час", text)

    minutes = None
    if minute_match:
        minutes = int(minute_match.group(1))
    elif hour_match:
        hours = int(hour_match.group(1))
        minutes = hours * 60

    return minutes


def _extract_index(text: str) -> Optional[int]:
    number_match = re.search(r"(\d+)", text)
    if number_match:
        return int(number_match.group(1))
    for word, idx in NUMBER_WORDS.items():
        if word in text:
            return idx
    return None


def _extract_absolute_time(lower: str, now: datetime) -> Optional[datetime]:
    target_date = _parse_date(lower, now.date())

    time_match = re.search(r"\b(\d{1,2})(?:[:.\s](\d{2}))?\b", lower)
    if not time_match:
        return None
    hour = int(time_match.group(1))
    minute = int(time_match.group(2)) if time_match.group(2) else 0

    qualifier = None
    for key in ("утра", "вечера", "вечер", "дня", "день", "ночи", "ночью", "ночь"):
        if key in lower:
            qualifier = key
            break
    hour = _adjust_hour(hour, qualifier)

    dt = datetime(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        hour=hour,
        minute=minute,
        tzinfo=now.tzinfo,
    )
    if dt <= now:
        # If time already passed today and no explicit future date, shift to tomorrow
        if target_date == now.date():
            dt = dt + timedelta(days=1)
        elif target_date < now.date():
            dt = dt.replace(year=dt.year + 1)
    return dt


def _parse_date(lower: str, today: date) -> date:
    if "послезавтра" in lower:
        return today + timedelta(days=2)
    if "завтра" in lower:
        return today + timedelta(days=1)
    if "сегодня" in lower:
        return today

    month_match = re.search(r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b", lower)
    if month_match:
        day = int(month_match.group(1))
        month = MONTHS[month_match.group(2)]
        year = today.year
        try_date = date(year, month, day)
        if try_date < today:
            try_date = date(year + 1, month, day)
        return try_date

    return today


def _adjust_hour(hour: int, qualifier: Optional[str]) -> int:
    hour = hour % 24
    if qualifier in ("утра", "утро"):
        if hour == 12:
            return 0
        return hour if hour < 12 else hour - 12
    if qualifier in ("вечера", "вечер", "дня", "день"):
        return hour if hour >= 12 else hour + 12
    if qualifier in ("ночи", "ночью", "ночь"):
        if hour == 12:
            return 0
        if 6 <= hour < 12:
            return hour + 12
        return hour
    return hour
