from __future__ import annotations

from dataclasses import dataclass, field

from .analysis_models import Event


@dataclass(frozen=True)
class EventNotification:
    event_type: str
    severity: str
    start_time_seconds: float
    end_time_seconds: float
    confidence: float
    explanation: str

    def as_payload(self) -> dict[str, object]:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class EventManagerUpdate:
    notifications: tuple[EventNotification, ...] = ()
    finalized_events: tuple[Event, ...] = ()
    drowsiness_confirmed: bool = False
    action_confirmed: bool = False
    face_confirmed: tuple[str, ...] = ()
    active_event_types: tuple[str, ...] = ()


@dataclass
class _TemporalSignal:
    session_id: int
    confirm_seconds: float
    source_name: str
    default_severity: str
    event_type: str | None = None
    label: str | None = None
    started_at: float | None = None
    peak_confidence: float = 0.0
    severity: str | None = None
    confirmed: bool = False

    @property
    def active(self) -> bool:
        return self.event_type is not None and self.started_at is not None

    def update(
        self,
        timestamp: float,
        event_type: str | None,
        confidence: float | None,
        label: str | None = None,
        severity: str | None = None,
    ) -> tuple[EventNotification | None, Event | None]:
        finalized = None
        if event_type != self.event_type:
            finalized = self._finalize(timestamp)
            if event_type is not None:
                self.event_type = event_type
                self.label = label
                self.started_at = timestamp
                self.peak_confidence = float(confidence or 0.0)
                self.severity = severity or self.default_severity
                self.confirmed = False
        elif event_type is not None:
            self.peak_confidence = max(self.peak_confidence, float(confidence or 0.0))
            self.severity = severity or self.severity or self.default_severity
            if label:
                self.label = label

        notification = None
        if self.active and not self.confirmed:
            elapsed = timestamp - float(self.started_at)
            if elapsed >= self.confirm_seconds:
                self.confirmed = True
                notification = EventNotification(
                    event_type=str(self.event_type),
                    severity=self.severity or self.default_severity,
                    start_time_seconds=float(self.started_at),
                    end_time_seconds=timestamp,
                    confidence=self.peak_confidence,
                    explanation=self._explanation(),
                )
        return notification, finalized

    def close(self, timestamp: float) -> Event | None:
        return self._finalize(timestamp)

    def _explanation(self) -> str:
        label_detail = f" ({self.label})" if self.label else ""
        return (
            f"Confirmed by {self.source_name}{label_detail} after "
            f"{self.confirm_seconds:.1f} seconds of continuous evidence."
        )

    def _finalize(self, timestamp: float) -> Event | None:
        event = None
        if self.active and self.confirmed:
            started_at = float(self.started_at)
            event = Event(
                session_id=self.session_id,
                event_type=str(self.event_type),
                start_time_seconds=started_at,
                end_time_seconds=timestamp,
                duration_seconds=max(timestamp - started_at, 0.0),
                severity=self.severity or self.default_severity,
                confidence=self.peak_confidence,
                explanation=self._explanation(),
            )
        self.event_type = None
        self.label = None
        self.started_at = None
        self.peak_confidence = 0.0
        self.severity = None
        self.confirmed = False
        return event


@dataclass
class DriverSafetyEventManager:
    session_id: int
    drowsiness_confirm_seconds: float
    action_confirm_seconds: float
    yawn_confirm_seconds: float = 2.0
    looking_away_confirm_seconds: float = 2.0
    head_down_confirm_seconds: float = 2.0
    completed_events: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._drowsiness = _TemporalSignal(
            session_id=self.session_id,
            confirm_seconds=self.drowsiness_confirm_seconds,
            source_name="the temporal drowsiness model",
            default_severity="high",
        )
        self._action = _TemporalSignal(
            session_id=self.session_id,
            confirm_seconds=self.action_confirm_seconds,
            source_name="the OpenVINO driver action model",
            default_severity="medium",
        )
        self._face = {
            "yawn": _TemporalSignal(
                session_id=self.session_id,
                confirm_seconds=self.yawn_confirm_seconds,
                source_name="MediaPipe Face Mesh MAR features",
                default_severity="medium",
            ),
            "looking_away": _TemporalSignal(
                session_id=self.session_id,
                confirm_seconds=self.looking_away_confirm_seconds,
                source_name="MediaPipe Face Mesh head-orientation features",
                default_severity="medium",
            ),
            "head_down": _TemporalSignal(
                session_id=self.session_id,
                confirm_seconds=self.head_down_confirm_seconds,
                source_name="MediaPipe Face Mesh head-orientation features",
                default_severity="high",
            ),
        }

    def update(
        self,
        *,
        timestamp: float,
        drowsiness_active: bool,
        drowsiness_confidence: float | None,
        action_event_type: str | None,
        action_confidence: float | None,
        action_label: str | None,
        face_signals: dict[str, dict[str, object]] | None = None,
    ) -> EventManagerUpdate:
        notifications: list[EventNotification] = []
        finalized_events: list[Event] = []

        notification, finalized = self._drowsiness.update(
            timestamp,
            "drowsiness" if drowsiness_active else None,
            drowsiness_confidence,
            label="drowsy",
            severity="high",
        )
        if notification:
            notifications.append(notification)
        if finalized:
            finalized_events.append(finalized)

        face_signals = face_signals or {}
        face_severity = {"yawn": "medium", "looking_away": "medium", "head_down": "high"}
        for event_type, temporal_signal in self._face.items():
            signal = face_signals.get(event_type) or {}
            active = bool(signal.get("active", False))
            notification, finalized = temporal_signal.update(
                timestamp,
                event_type if active else None,
                float(signal.get("confidence", 0.0) or 0.0),
                label=event_type,
                severity=face_severity[event_type],
            )
            if notification:
                notifications.append(notification)
            if finalized:
                finalized_events.append(finalized)

        action_severity = "high" if action_event_type in {"phone_usage", "phone_call"} else "medium"
        notification, finalized = self._action.update(
            timestamp,
            action_event_type,
            action_confidence,
            label=action_label,
            severity=action_severity,
        )
        if notification:
            notifications.append(notification)
        if finalized:
            finalized_events.append(finalized)

        self.completed_events.extend(finalized_events)
        return EventManagerUpdate(
            notifications=tuple(notifications),
            finalized_events=tuple(finalized_events),
            drowsiness_confirmed=self._drowsiness.confirmed,
            action_confirmed=self._action.confirmed,
            face_confirmed=tuple(
                event_type for event_type, signal in self._face.items() if signal.confirmed
            ),
            active_event_types=self.active_event_types,
        )

    @property
    def active_event_types(self) -> tuple[str, ...]:
        return tuple(
            signal.event_type
            for signal in (self._drowsiness, self._action, *self._face.values())
            if signal.event_type is not None
        )

    @property
    def confirmed_event_count(self) -> int:
        active_confirmed = (
            int(self._drowsiness.confirmed)
            + int(self._action.confirmed)
            + sum(int(signal.confirmed) for signal in self._face.values())
        )
        return len(self.completed_events) + active_confirmed

    def close(self, timestamp: float) -> tuple[Event, ...]:
        finalized = [
            event
            for event in (
                self._drowsiness.close(timestamp),
                self._action.close(timestamp),
                *(signal.close(timestamp) for signal in self._face.values()),
            )
            if event is not None
        ]
        self.completed_events.extend(finalized)
        return tuple(finalized)
