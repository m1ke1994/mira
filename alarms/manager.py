from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta
from threading import Event, Lock, Thread
from typing import Callable, List, Optional
from .sounds import AlarmSoundPlayer
from .storage import Alarm, load_alarms, save_alarms

logger = logging.getLogger(__name__)


class AlarmRuntimeState:
    def __init__(self) -> None:
        self.ringing_alarm: Optional[Alarm] = None
        self.last_trigger_ts: Optional[float] = None


class AlarmManager:
    def __init__(
        self,
        storage_path,
        sound_player: AlarmSoundPlayer,
        check_interval: float = 0.8,
        default_snooze_minutes: int = 5,
        on_alarm_triggered: Optional[Callable[[Alarm], None]] = None,
        timezone=None,
    ):
        self.storage_path = storage_path
        self.sound_player = sound_player
        self.check_interval = max(0.2, check_interval)
        self.default_snooze_minutes = max(1, default_snooze_minutes)
        self.on_alarm_triggered = on_alarm_triggered
        self.tzinfo = timezone or datetime.now().astimezone().tzinfo

        self._alarms: List[Alarm] = []
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._runtime = AlarmRuntimeState()

    def start(self) -> None:
        self._alarms = sorted(load_alarms(self.storage_path), key=lambda a: a.fire_at)
        logger.info("Loaded %s alarms from %s", len(self._alarms), self.storage_path)
        self._stop_event.clear()
        self._thread = Thread(target=self._loop, name="alarm-scheduler", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.sound_player.stop_loop()
        self._thread = None

    def add_alarm(self, fire_at: datetime, label: str | None = None, snoozed_from: str | None = None) -> Alarm:
        fire_at = _ensure_tz(fire_at, self.tzinfo)
        now = datetime.now(self.tzinfo)
        if fire_at <= now:
            raise ValueError("Нельзя поставить будильник в прошлом")
        alarm = Alarm(
            id=f"al_{uuid.uuid4().hex[:8]}",
            fire_at=fire_at,
            label=label or "Будильник",
            created_at=now,
            snoozed_from=snoozed_from,
        )
        with self._lock:
            self._alarms.append(alarm)
            self._alarms.sort(key=lambda a: a.fire_at)
            save_alarms(self.storage_path, self._alarms)
        logger.info("Alarm scheduled for %s (label=%s)", fire_at.isoformat(), alarm.label)
        return alarm

    def list_alarms(self) -> List[Alarm]:
        with self._lock:
            return list(sorted(self._alarms, key=lambda a: a.fire_at))

    def remove_alarm_by_index(self, index: int) -> Optional[Alarm]:
        if index < 1:
            return None
        with self._lock:
            if index > len(self._alarms):
                return None
            alarm = sorted(self._alarms, key=lambda a: a.fire_at)[index - 1]
            self._alarms = [a for a in self._alarms if a.id != alarm.id]
            save_alarms(self.storage_path, self._alarms)
            logger.info("Removed alarm %s (index=%s)", alarm.id, index)
            return alarm

    def remove_nearest(self) -> Optional[Alarm]:
        with self._lock:
            if not self._alarms:
                return None
            alarm = sorted(self._alarms, key=lambda a: a.fire_at)[0]
            self._alarms = [a for a in self._alarms if a.id != alarm.id]
            save_alarms(self.storage_path, self._alarms)
            logger.info("Removed nearest alarm %s", alarm.id)
            return alarm

    def stop_ringing(self) -> Optional[Alarm]:
        with self._lock:
            current = self._runtime.ringing_alarm
            self._runtime.ringing_alarm = None
        self.sound_player.stop_loop()
        return current

    def snooze(self, minutes: Optional[int]) -> Optional[Alarm]:
        minutes = minutes or self.default_snooze_minutes
        with self._lock:
            ringing = self._runtime.ringing_alarm
        if not ringing:
            return None
        self.stop_ringing()
        new_time = datetime.now(self.tzinfo) + timedelta(minutes=minutes)
        return self.add_alarm(new_time, label=ringing.label, snoozed_from=ringing.id)

    @property
    def is_ringing(self) -> bool:
        with self._lock:
            return self._runtime.ringing_alarm is not None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            due_alarm = self._pop_due_alarm()
            if due_alarm:
                self._trigger_alarm(due_alarm)
                continue
            self._stop_event.wait(self.check_interval)

    def _pop_due_alarm(self) -> Optional[Alarm]:
        now = datetime.now(self.tzinfo)
        with self._lock:
            if not self._alarms:
                return None
            self._alarms.sort(key=lambda a: a.fire_at)
            next_alarm = self._alarms[0]
            if next_alarm.fire_at <= now:
                self._alarms = self._alarms[1:]
                save_alarms(self.storage_path, self._alarms)
                return next_alarm
        return None

    def _trigger_alarm(self, alarm: Alarm) -> None:
        logger.info("Alarm triggered at %s (label=%s)", alarm.fire_at.isoformat(), alarm.label)
        with self._lock:
            self._runtime.ringing_alarm = alarm
            self._runtime.last_trigger_ts = time.time()
        self.sound_player.start_loop()
        if self.on_alarm_triggered:
            try:
                self.on_alarm_triggered(alarm)
            except Exception:  # pragma: no cover - callback safety
                logger.error("on_alarm_triggered callback failed", exc_info=True)


def _ensure_tz(dt: datetime, tzinfo) -> datetime:
    if dt.tzinfo:
        return dt.astimezone(tzinfo)
    return dt.replace(tzinfo=tzinfo)
