from __future__ import annotations

from pathlib import Path

from .ai.detectors import DetectorFactory
from .analysis_models import Event
from .risk_engine import contribution_for_event
from .video_processing import iter_sampled_frames


def _severity_for_phone(duration, max_confidence, detection_count):
    if duration > 10 or detection_count >= 10 or max_confidence >= 0.85:
        return "critical"
    if duration > 5 or detection_count >= 5 or max_confidence >= 0.70:
        return "high"
    if duration >= 2.5 or detection_count >= 3:
        return "medium"
    return "low"


def _build_phone_event(session_id, start_time, end_time, max_confidence, detection_count):
    duration = max(end_time - start_time, 0.1)
    event = Event(
        session_id=session_id,
        event_type="phone_usage",
        start_time_seconds=round(start_time, 2),
        end_time_seconds=round(end_time, 2),
        duration_seconds=round(duration, 2),
        severity=_severity_for_phone(duration, max_confidence, detection_count),
        confidence=round(max_confidence, 2),
        explanation=(
            "YOLO detected a phone in repeated sampled frames. "
            f"Detections: {detection_count}, max confidence: {max_confidence:.2f}."
        ),
    )
    return Event(**{**event.__dict__, "risk_contribution": contribution_for_event(event)})


def detections_to_phone_events(session_id, phone_frames, merge_gap_seconds=2.0, min_detections=2):
    if len(phone_frames) < min_detections:
        return []

    events = []
    current = {
        "start": phone_frames[0]["timestamp_seconds"],
        "end": phone_frames[0]["timestamp_seconds"],
        "max_confidence": phone_frames[0]["confidence"],
        "count": 1,
    }

    for frame in phone_frames[1:]:
        timestamp = frame["timestamp_seconds"]
        if timestamp - current["end"] <= merge_gap_seconds:
            current["end"] = timestamp
            current["max_confidence"] = max(current["max_confidence"], frame["confidence"])
            current["count"] += 1
        else:
            if current["count"] >= min_detections:
                events.append(_build_phone_event(
                    session_id,
                    current["start"],
                    current["end"],
                    current["max_confidence"],
                    current["count"],
                ))
            current = {
                "start": timestamp,
                "end": timestamp,
                "max_confidence": frame["confidence"],
                "count": 1,
            }

    if current["count"] >= min_detections:
        events.append(_build_phone_event(
            session_id,
            current["start"],
            current["end"],
            current["max_confidence"],
            current["count"],
        ))

    return events


def analyze_video_for_phone(
    session_id,
    video_path,
    model_path,
    confidence_threshold,
    sample_fps,
    detector_backend="yolo_pretrained",
    driver_roi="0,0,1,1",
    temporal_min_detections=2,
    temporal_merge_gap_seconds=2.0,
):
    detector = DetectorFactory.create(
        detector_backend,
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        strict=True,
    )
    phone_frames = []
    processed_frames = 0
    ignored_phone_detections = 0

    for frame, _frame_index, timestamp_seconds in iter_sampled_frames(
        Path(video_path),
        sample_fps=sample_fps,
    ):
        processed_frames += 1
        detections = detector.detect_frame(frame)
        phone_detections = [detection for detection in detections if detection["is_phone"]]
        driver_phone_detections = [
            detection
            for detection in phone_detections
            if _detection_in_driver_roi(detection, frame, driver_roi)
        ]
        ignored_phone_detections += max(len(phone_detections) - len(driver_phone_detections), 0)
        if driver_phone_detections:
            max_confidence = max(detection["confidence"] for detection in driver_phone_detections)
            phone_frames.append(
                {
                    "timestamp_seconds": timestamp_seconds,
                    "confidence": max_confidence,
                    "detections": driver_phone_detections,
                }
            )

    events = detections_to_phone_events(
        session_id,
        phone_frames,
        merge_gap_seconds=temporal_merge_gap_seconds,
        min_detections=temporal_min_detections,
    )
    max_confidence = max((frame["confidence"] for frame in phone_frames), default=None)
    metrics = {
        "yolo_processed_frame_count": processed_frames,
        "phone_detected_frame_count": len(phone_frames),
        "ignored_phone_detection_count": ignored_phone_detections,
        "max_phone_confidence": max_confidence,
        "phone_usage_event_count": len(events),
        "phone_detection_error": None,
    }
    return events, metrics


def _detection_in_driver_roi(detection, frame, driver_roi):
    try:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = detection["bbox"]
        center_x = ((x1 + x2) / 2.0) / max(width, 1)
        center_y = ((y1 + y2) / 2.0) / max(height, 1)
        roi_x1, roi_y1, roi_x2, roi_y2 = _parse_roi(driver_roi)
        return roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2
    except Exception:
        return True


def _parse_roi(raw_value):
    try:
        values = [float(part.strip()) for part in str(raw_value).split(",")]
        if len(values) != 4:
            raise ValueError
        x1, y1, x2, y2 = values
        return (
            min(max(x1, 0.0), 1.0),
            min(max(y1, 0.0), 1.0),
            min(max(x2, 0.0), 1.0),
            min(max(y2, 0.0), 1.0),
        )
    except Exception:
        return (0.0, 0.0, 1.0, 1.0)
