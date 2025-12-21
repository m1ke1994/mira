from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Alarm:
    id: str
    fire_at: datetime
    label: str
    created_at: datetime
    snoozed_from: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "fire_at": self.fire_at.isoformat(),
            "label": self.label,
            "created_at": self.created_at.isoformat(),
            "snoozed_from": self.snoozed_from,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Alarm":
        fire_at_raw = data.get("fire_at") or data.get("time")
        created_raw = data.get("created_at") or data.get("created")
        if not fire_at_raw or not created_raw:
            raise ValueError("Alarm payload missing fire_at/created_at fields")
        return cls(
            id=str(data.get("id", "")),
            fire_at=datetime.fromisoformat(fire_at_raw),
            label=str(data.get("label") or "Будильник"),
            created_at=datetime.fromisoformat(created_raw),
            snoozed_from=data.get("snoozed_from"),
        )


def load_alarms(path: Path) -> List[Alarm]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # pragma: no cover - corrupted file
        logger.error("Failed to load alarms from %s: %s", path, exc)
        return []
    alarms: List[Alarm] = []
    for item in payload or []:
        try:
            alarms.append(Alarm.from_dict(item))
        except Exception as exc:  # pragma: no cover - corrupted item
            logger.warning("Skipping alarm item due to parse error: %s", exc)
    return alarms


def save_alarms(path: Path, alarms: List[Alarm]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [a.to_dict() for a in alarms]
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
