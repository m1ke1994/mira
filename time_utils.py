from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def resolve_timezone(name: Optional[str]) -> timezone:
    preferred = name or "Europe/Moscow"
    try:
        tz = ZoneInfo(preferred)
        return tz  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover - environment-dependent
        logger.warning("Failed to load timezone %s via zoneinfo (%s)", preferred, exc)
    # Explicit Moscow fallback if requested
    if preferred.lower() in {"europe/moscow", "moscow", "msk"}:
        logger.warning("Falling back to fixed MSK offset +03:00")
        return timezone(timedelta(hours=3), name="MSK")
    # Try system local TZ
    local_tz = datetime.now().astimezone().tzinfo
    if local_tz:
        logger.warning("Using system local timezone instead: %s", getattr(local_tz, "key", local_tz))
        return local_tz  # type: ignore[return-value]
    logger.warning("System timezone unavailable, fallback to +03:00")
    return timezone(timedelta(hours=3), name="MSK")


def now_in_tz(tzinfo) -> datetime:
    if tzinfo:
        return datetime.now(tzinfo)
    return datetime.now().astimezone()


def format_tz_offset(tzinfo) -> str:
    sample = now_in_tz(tzinfo)
    offset = tzinfo.utcoffset(sample) if hasattr(tzinfo, "utcoffset") else None
    if offset is None:
        return ""
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"
