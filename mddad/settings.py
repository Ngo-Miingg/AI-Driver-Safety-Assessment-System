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
    # Upload analysis still uses this sampling rate even when the legacy
    # detector UI is disabled. Keep it in the shared settings contract so
    # upload and analysis routes receive a complete configuration dictionary.
    SettingDefinition(
        "uploaded_sample_fps",
        "Uploaded video sample FPS",
        "UPLOADED_SAMPLE_FPS",
        "Number of video frames sampled per second for uploaded-video analysis.",
        0.5,
        30.0,
    ),
    # These two keys are retained for old stored sessions and compatibility
    # routes. They are not required by the current realtime product flow.
    SettingDefinition(
        "yolo_confidence_threshold",
        "Legacy detector confidence",
        "YOLO_CONFIDENCE_THRESHOLD",
        "Compatibility threshold for legacy uploaded-video detector analysis.",
        0.0,
        1.0,
    ),
    SettingDefinition(
        "realtime_yolo_confidence_threshold",
        "Legacy realtime detector confidence",
        "REALTIME_YOLO_CONFIDENCE_THRESHOLD",
        "Compatibility threshold for legacy realtime detector settings.",
        0.0,
        1.0,
    ),
    SettingDefinition(
        "realtime_target_fps",
        "Realtime target FPS",
        "REALTIME_TARGET_FPS",
        "Target webcam capture/update rate for local multimodal monitoring.",
        1.0,
        30.0,
    ),
    SettingDefinition(
        "multimodal_clip_seconds",
        "Multimodal clip window",
        "MULTIMODAL_CLIP_SECONDS",
        "Seconds of video/audio/skeleton context used for each behavior decision.",
        1.0,
        10.0,
    ),
    SettingDefinition(
        "multimodal_clip_stride_seconds",
        "Multimodal clip stride",
        "MULTIMODAL_CLIP_STRIDE_SECONDS",
        "Seconds between consecutive multimodal behavior decisions.",
        0.5,
        10.0,
    ),
    SettingDefinition(
        "multimodal_frame_count",
        "Multimodal frame count",
        "MULTIMODAL_FRAME_COUNT",
        "Number of sampled frames passed to the multimodal model per clip.",
        4.0,
        64.0,
        "1",
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
