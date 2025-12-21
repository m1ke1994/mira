from datetime import datetime, timezone

from alarms.parser import parse_alarm_request


def _now() -> datetime:
    return datetime(2025, 1, 1, 6, 0, tzinfo=timezone.utc)


def test_parse_absolute_time_same_day():
    result = parse_alarm_request("поставь будильник на 7:30", now=_now())
    assert result
    assert result.action == "add"
    assert result.fire_at.hour == 7
    assert result.fire_at.minute == 30


def test_parse_evening_qualifier():
    result = parse_alarm_request("разбуди меня в 8 вечера", now=_now())
    assert result
    assert result.action == "add"
    assert result.fire_at.hour == 20


def test_parse_relative_minutes():
    result = parse_alarm_request("поставь будильник через 15 минут", now=_now())
    assert result
    assert result.action == "add"
    assert result.delta_minutes == 15


def test_parse_date_with_month():
    base = datetime(2025, 5, 1, 10, 0, tzinfo=timezone.utc)
    result = parse_alarm_request("25 декабря в 7:00", now=base)
    assert result
    assert result.action == "add"
    assert result.fire_at.month == 12
    assert result.fire_at.day == 25


def test_parse_snooze_minutes():
    result = parse_alarm_request("отложи на 5 минут", now=_now(), allow_stop_without_keyword=True)
    assert result
    assert result.action == "snooze"
    assert result.snooze_minutes == 5
