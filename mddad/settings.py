from __future__ import annotations

from dataclasses import dataclass

from flask import current_app


@dataclass(frozen=True)
class SettingDefinition:
    key: str
    label: str
    config_name: str
    description: str
    minimum: float
    maximum: float
    step: str = "0.01"


SETTING_DEFINITIONS = [
    SettingDefinition(
        "ear_threshold",
        "EAR threshold",
        "EAR_THRESHOLD",
        "Lower EAR means more closed eyes. Frames below this value are treated as eye-closed candidates.",
        0.05,
        0.50,
    ),
    SettingDefinition(
        "mar_threshold",
        "MAR threshold",
        "MAR_THRESHOLD",
        "Higher MAR means a more open mouth. Frames above this value are treated as yawn candidates.",
        0.20,
        1.20,
    ),
    SettingDefinition(
        "eye_closure_min_duration",
        "Eye closure minimum duration",
        "EYE_CLOSURE_MIN_DURATION",
        "Seconds that eye closure must persist before it becomes an event.",
        0.10,
        10.0,
    ),
    SettingDefinition(
        "yawn_min_duration",
        "Yawn minimum duration",
        "YAWN_MIN_DURATION",
        "Seconds that a yawn candidate must persist before it becomes an event.",
        0.10,
        10.0,
    ),
    SettingDefinition(
        "looking_away_min_duration",
        "Looking away minimum duration",
        "LOOKING_AWAY_MIN_DURATION",
        "Seconds that left/right head orientation must persist before it becomes an event.",
        0.10,
        15.0,
    ),
    SettingDefinition(
        "head_down_min_duration",
        "Head down minimum duration",
        "HEAD_DOWN_MIN_DURATION",
        "Seconds that downward head orientation must persist before it becomes an event.",
        0.10,
        15.0,
    ),
    SettingDefinition(
        "face_missing_min_duration",
        "Face missing minimum duration",
        "FACE_MISSING_MIN_DURATION",
        "Seconds without a detectable face before a face-missing event is recorded.",
        0.10,
        20.0,
    ),
    SettingDefinition(
        "yolo_confidence_threshold",
        "YOLO confidence threshold",
        "YOLO_CONFIDENCE_THRESHOLD",
        "Minimum YOLO confidence for uploaded-video phone detections.",
        0.05,
        0.95,
    ),
    SettingDefinition(
        "uploaded_sample_fps",
        "Uploaded video sample FPS",
        "UPLOADED_SAMPLE_FPS",
        "Number of uploaded-video frames sampled per second for MediaPipe metadata and behavior analysis.",
        0.5,
        30.0,
    ),
    SettingDefinition(
        "realtime_target_fps",
        "Realtime target FPS",
        "REALTIME_TARGET_FPS",
        "Target webcam analysis rate for realtime MediaPipe monitoring.",
        1.0,
        30.0,
    ),
    SettingDefinition(
        "snapshot_preview_enabled",
        "Snapshot preview enabled",
        "REALTIME_ENABLE_SNAPSHOT_PREVIEW",
        "Use 1 to enable low-rate realtime JPEG snapshots, or 0 to keep text-only realtime monitoring.",
        0.0,
        1.0,
        "1",
    ),
    SettingDefinition(
        "snapshot_fps",
        "Snapshot FPS",
        "REALTIME_SNAPSHOT_FPS",
        "Maximum JPEG snapshots emitted per second through SocketIO. Use 2-3 for a smoother demo while avoiding full video streaming.",
        0.1,
        5.0,
    ),
    SettingDefinition(
        "snapshot_width",
        "Snapshot width",
        "REALTIME_SNAPSHOT_WIDTH",
        "Width in pixels for resized realtime snapshots and event evidence images.",
        160.0,
        960.0,
        "10",
    ),
    SettingDefinition(
        "event_evidence_enabled",
        "Event evidence enabled",
        "REALTIME_ENABLE_EVENT_EVIDENCE",
        "Use 1 to save one small JPEG evidence image for each confirmed realtime event, or 0 to disable.",
        0.0,
        1.0,
        "1",
    ),
    SettingDefinition(
        "realtime_yolo_enabled",
        "Realtime YOLO enabled",
        "REALTIME_ENABLE_YOLO",
        "Use 1 to enable low-rate realtime YOLO phone detection, or 0 to keep realtime MediaPipe-only.",
        0.0,
        1.0,
        "1",
    ),
    SettingDefinition(
        "realtime_yolo_sample_fps",
        "Realtime YOLO sample FPS",
        "REALTIME_YOLO_SAMPLE_FPS",
        "Maximum realtime YOLO phone-detection rate. Keep this low for webcam performance.",
        0.1,
        5.0,
    ),
    SettingDefinition(
        "realtime_yolo_confidence_threshold",
        "Realtime YOLO confidence threshold",
        "REALTIME_YOLO_CONFIDENCE_THRESHOLD",
        "Minimum confidence for realtime YOLO phone detections.",
        0.05,
        0.95,
    ),
]


def default_settings(app=None):
    app = app or current_app
    return {
        definition.key: float(app.config[definition.config_name])
        for definition in SETTING_DEFINITIONS
    }


def get_analysis_settings(db=None, app=None):
    app = app or current_app
    values = default_settings(app)
    if db is None:
        from .database import get_db

        db = get_db()

    rows = db.execute("SELECT key, value FROM app_settings").fetchall()
    allowed = {definition.key for definition in SETTING_DEFINITIONS}
    for row in rows:
        if row["key"] not in allowed:
            continue
        try:
            values[row["key"]] = float(row["value"])
        except (TypeError, ValueError):
            continue
    return values


def validate_settings(form_data):
    values = {}
    errors = []
    for definition in SETTING_DEFINITIONS:
        raw_value = form_data.get(definition.key, "").strip()
        try:
            value = float(raw_value)
        except ValueError:
            errors.append(f"{definition.label} must be a number.")
            continue
        if value < definition.minimum or value > definition.maximum:
            errors.append(
                f"{definition.label} must be between {definition.minimum:g} and {definition.maximum:g}."
            )
            continue
        values[definition.key] = value
    return values, errors


def save_settings(db, values):
    for key, value in values.items():
        db.execute(
            """
            INSERT INTO app_settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, str(value)),
        )


def reset_settings(db):
    db.execute("DELETE FROM app_settings")
