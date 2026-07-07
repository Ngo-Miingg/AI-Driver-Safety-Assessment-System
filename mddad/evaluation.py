from __future__ import annotations

EVENT_TYPES = (
    "normal_driving",
    "phone_interaction",
    "phone_calling",
    "in_vehicle_display_interaction",
    "eating_distraction",
    "drinking_distraction",
    "visual_posture_distraction",
    "eye_closure",
    "yawn",
    "phone_usage",
    "phone_call",
    "talking_to_passenger",
    "looking_away",
    "head_down",
    "mixed_risk",
    "face_missing",
)


def evaluation_query():
    event_columns = ",\n                ".join(
        f"SUM(CASE WHEN e.event_type = '{event_type}' THEN 1 ELSE 0 END) AS {event_type}_count"
        for event_type in EVENT_TYPES
    )
    return f"""
            SELECT
                s.id AS session_id,
                d.display_name AS driver_name,
                s.session_type,
                s.status,
                COALESCE(s.video_duration_seconds, s.duration_seconds, 0) AS duration,
                s.final_safety_score AS safety_score,
                s.final_risk_score AS risk_score,
                s.final_alert_level AS alert_level,
                COUNT(e.id) AS event_count,
                {event_columns},
                s.processed_frame_count,
                s.face_detected_frame_count,
                s.face_missing_frame_count,
                s.average_processing_fps,
                s.phone_usage_event_count,
                s.analysis_mode,
                s.analysis_error,
                s.phone_detection_error,
                s.created_at,
                s.started_at
            FROM sessions s
            JOIN drivers d ON d.id = s.driver_id
            LEFT JOIN events e ON e.session_id = s.id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY COALESCE(s.started_at, s.created_at) DESC, s.id DESC
            """


def rows_for_user(db, user_id):
    return db.execute(evaluation_query(), (user_id,)).fetchall()


def row_to_summary(row):
    processed = row["processed_frame_count"] or 0
    detected = row["face_detected_frame_count"] or 0
    duration = row["duration"] or 0
    face_detection_rate = (detected / processed) if processed else None
    approximate_processing_fps = row["average_processing_fps"]
    if approximate_processing_fps is None and duration and processed:
        approximate_processing_fps = processed / duration
    event_counts = {
        event_type: row[f"{event_type}_count"] or 0
        for event_type in EVENT_TYPES
    }
    dominant_event_type = max(event_counts, key=event_counts.get) if any(event_counts.values()) else None
    warnings = [
        warning
        for warning in (row["analysis_error"], row["phone_detection_error"])
        if warning
    ]
    return {
        "face_detection_rate": face_detection_rate,
        "approximate_processing_fps": approximate_processing_fps,
        "event_counts": event_counts,
        "dominant_event_type": dominant_event_type,
        "warnings": warnings,
    }


def csv_header():
    return [
        "session_id",
        "driver_name",
        "session_type",
        "status",
        "duration",
        "safety_score",
        "risk_score",
        "alert_level",
        "event_count",
        "eye_closure_count",
        "yawn_count",
        "phone_usage_count",
        "phone_call_count",
        "talking_to_passenger_count",
        "looking_away_count",
        "head_down_count",
        "mixed_risk_count",
        "face_missing_count",
        "processed_frame_count",
        "face_detected_frame_count",
        "face_detection_rate",
        "analysis_mode",
    ]


def csv_row(row):
    summary = row_to_summary(row)
    rate = summary["face_detection_rate"]
    return [
        row["session_id"],
        row["driver_name"],
        row["session_type"],
        row["status"],
        row["duration"],
        row["safety_score"],
        row["risk_score"],
        row["alert_level"],
        row["event_count"],
        row["eye_closure_count"] or 0,
        row["yawn_count"] or 0,
        row["phone_usage_count"] or 0,
        row["phone_call_count"] or 0,
        row["talking_to_passenger_count"] or 0,
        row["looking_away_count"] or 0,
        row["head_down_count"] or 0,
        row["mixed_risk_count"] or 0,
        row["face_missing_count"] or 0,
        row["processed_frame_count"] or 0,
        row["face_detected_frame_count"] or 0,
        round(rate, 4) if rate is not None else "",
        row["analysis_mode"] or "",
    ]
