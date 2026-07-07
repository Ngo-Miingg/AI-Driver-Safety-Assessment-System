from __future__ import annotations

from pathlib import Path

from .ai.face_analyzer import FaceAnalyzerConfig, MediaPipeFaceAnalyzer
from .analysis_models import Event
from .risk_engine import contribution_for_event
from .video_processing import iter_sampled_frames


DEFAULT_SIGNAL_TO_EVENT = {
    "eye_closed": {
        "event_type": "eye_closure",
        "minimum_duration": 1.0,
        "explanation": "MediaPipe EAR stayed below threshold for a sustained period.",
    },
    "yawn_candidate": {
        "event_type": "yawn",
        "minimum_duration": 1.2,
        "explanation": "MediaPipe MAR stayed above threshold for a sustained period.",
    },
    "looking_away": {
        "event_type": "looking_away",
        "minimum_duration": 1.5,
        "explanation": "MediaPipe landmark geometry indicated sustained left/right head orientation.",
    },
    "head_down": {
        "event_type": "head_down",
        "minimum_duration": 1.2,
        "explanation": "MediaPipe landmark geometry indicated sustained downward head orientation.",
    },
    "face_missing": {
        "event_type": "face_missing",
        "minimum_duration": 2.0,
        "explanation": "MediaPipe could not detect a usable face for a sustained period.",
    },
}


def signal_config(settings=None):
    config = {key: value.copy() for key, value in DEFAULT_SIGNAL_TO_EVENT.items()}
    if not settings:
        return config
    mapping = {
        "eye_closed": "eye_closure_min_duration",
        "yawn_candidate": "yawn_min_duration",
        "looking_away": "looking_away_min_duration",
        "head_down": "head_down_min_duration",
        "face_missing": "face_missing_min_duration",
    }
    for signal, setting_key in mapping.items():
        if setting_key in settings:
            config[signal]["minimum_duration"] = float(settings[setting_key])
    return config


def face_analyzer_config(settings=None):
    if not settings:
        return FaceAnalyzerConfig()
    return FaceAnalyzerConfig(
        ear_threshold=float(settings.get("ear_threshold", FaceAnalyzerConfig.ear_threshold)),
        mar_threshold=float(settings.get("mar_threshold", FaceAnalyzerConfig.mar_threshold)),
    )


def _severity_for_event(event_type, duration):
    if event_type == "eye_closure":
        if duration >= 6:
            return "critical"
        if duration >= 3.5:
            return "high"
        if duration >= 2:
            return "medium"
        return "low"
    if event_type == "yawn":
        if duration > 4:
            return "high"
        if duration >= 2.5:
            return "medium"
        return "low"
    if event_type == "head_down":
        if duration > 7:
            return "critical"
        if duration >= 4:
            return "high"
        if duration >= 2.5:
            return "medium"
        return "low"
    if event_type == "face_missing":
        if duration > 12:
            return "critical"
        if duration >= 7:
            return "high"
        if duration >= 4:
            return "medium"
        return "low"
    if duration > 8:
        return "critical"
    if duration >= 5:
        return "high"
    if duration >= 3:
        return "medium"
    return "low"


def _build_event(session_id, event_type, start_time, end_time, confidence, explanation):
    duration = max(end_time - start_time, 0.1)
    event = Event(
        session_id=session_id,
        event_type=event_type,
        start_time_seconds=round(start_time, 2),
        end_time_seconds=round(end_time, 2),
        duration_seconds=round(duration, 2),
        severity=_severity_for_event(event_type, duration),
        confidence=round(confidence, 2),
        explanation=explanation,
    )
    return Event(**{**event.__dict__, "risk_contribution": contribution_for_event(event)})


def _close_event_if_valid(session_id, active, end_time, signals):
    config = signals[active["signal"]]
    duration = end_time - active["start_time"]
    if duration < config["minimum_duration"]:
        return None
    confidence_values = active["confidence_values"] or [0.75]
    confidence = sum(confidence_values) / len(confidence_values)
    return _build_event(
        session_id=session_id,
        event_type=config["event_type"],
        start_time=active["start_time"],
        end_time=end_time,
        confidence=confidence,
        explanation=config["explanation"],
    )


def frame_results_to_events(session_id, frame_results, settings=None):
    events = []
    active_by_signal = {}
    signals = signal_config(settings)

    for result in frame_results:
        timestamp = result["timestamp_seconds"]
        signal_state = {
            "eye_closed": result["eye_closed"],
            "yawn_candidate": result["yawn_candidate"],
            "looking_away": result["looking_away"],
            "head_down": result["head_down"],
            "face_missing": not result["face_detected"],
        }
        for signal, is_active in signal_state.items():
            active = active_by_signal.get(signal)
            if is_active and active is None:
                active_by_signal[signal] = {
                    "signal": signal,
                    "start_time": timestamp,
                    "last_time": timestamp,
                    "confidence_values": [],
                }
                active = active_by_signal[signal]
            if is_active:
                active["last_time"] = timestamp
                active["confidence_values"].append(0.8 if signal == "face_missing" else 0.85)
            elif active is not None:
                event = _close_event_if_valid(session_id, active, active["last_time"], signals)
                if event is not None:
                    events.append(event)
                active_by_signal.pop(signal, None)

    for active in list(active_by_signal.values()):
        event = _close_event_if_valid(session_id, active, active["last_time"], signals)
        if event is not None:
            events.append(event)

    events.sort(key=lambda event: event.start_time_seconds)
    return events


def analyze_video_with_mediapipe(session_id, video_path, sample_fps=5.0, settings=None):
    analyzer = MediaPipeFaceAnalyzer(face_analyzer_config(settings))
    frame_results = []
    try:
        for frame, _frame_index, timestamp_seconds in iter_sampled_frames(
            Path(video_path),
            sample_fps=sample_fps,
        ):
            frame_results.append(analyzer.analyze_frame(frame, timestamp_seconds))
    finally:
        analyzer.close()

    events = frame_results_to_events(session_id, frame_results, settings)
    ear_values = [result["ear"] for result in frame_results if result["ear"] is not None]
    mar_values = [result["mar"] for result in frame_results if result["mar"] is not None]
    face_detected_count = sum(1 for result in frame_results if result["face_detected"])
    processed_count = len(frame_results)

    metrics = {
        "processed_frame_count": processed_count,
        "face_detected_frame_count": face_detected_count,
        "face_missing_frame_count": processed_count - face_detected_count,
        "average_ear": sum(ear_values) / len(ear_values) if ear_values else None,
        "average_mar": sum(mar_values) / len(mar_values) if mar_values else None,
        "min_ear": min(ear_values) if ear_values else None,
        "max_mar": max(mar_values) if mar_values else None,
    }
    return frame_results, events, metrics
