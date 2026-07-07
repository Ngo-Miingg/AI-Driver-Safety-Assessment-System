from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .activity7_infer import Activity7Package
from .analysis_models import Event
from .mmrt_v2_infer import MMRTV2Package
from .risk_engine import contribution_for_event


MULTIMODAL_EVENT_LABELS = {
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
    "phone_use",
    "talking_to_passenger",
    "looking_away",
    "yawning",
    "head_down",
    "mixed_risk",
    "face_missing",
}


@dataclass(frozen=True)
class MultimodalModelConfig:
    backend: str
    model_path: Path
    package_path: Path
    clip_seconds: float
    clip_stride_seconds: float
    frame_count: int
    labels: tuple[str, ...]
    decision_mode: str
    device: str = "cpu"


@dataclass(frozen=True)
class MultimodalAnalysisResult:
    events: list[Event]
    metrics: dict
    clips: list[dict]


def _is_mmrt_backend(backend: str) -> bool:
    return str(backend or "").strip().lower().startswith("mmrt")


def config_from_app(app):
    backend = str(app.config.get("MULTIMODAL_MODEL_BACKEND", "activity7_package"))
    if _is_mmrt_backend(backend):
        package_path = Path(app.config.get("MMRT_V2_MODEL_PATH", app.config.get("MULTIMODAL_MODEL_PATH", "")))
        model_path = package_path
        labels = ("normal_driving", "eye_closure", "yawning", "looking_away", "phone_use")
        device = "cuda" if str(app.config.get("MMRT_V2_DEVICE", "cpu")).lower() == "cuda" else "cpu"
    else:
        package_path = Path(app.config["ACTIVITY7_MODEL_PACKAGE_PATH"])
        model_path = Path(app.config.get("MULTIMODAL_MODEL_PATH", package_path))
        labels = tuple(
            label.strip()
            for label in str(app.config.get("MULTIMODAL_LABELS", "")).split(",")
            if label.strip()
        )
        if not labels:
            labels = ("normal_driving", "phone_interaction", "phone_calling")
        device = "cuda" if app.config.get("MULTIMODAL_DEVICE") == "cuda" else "cpu"
    resolved_path = package_path if package_path.exists() else model_path
    return MultimodalModelConfig(
        backend=backend,
        model_path=resolved_path,
        package_path=package_path,
        clip_seconds=float(app.config.get("MULTIMODAL_CLIP_SECONDS", 4.0)),
        clip_stride_seconds=float(app.config.get("MULTIMODAL_CLIP_STRIDE_SECONDS", 2.0)),
        frame_count=int(app.config.get("MULTIMODAL_FRAME_COUNT", 16)),
        labels=labels,
        decision_mode=str(app.config.get("ACTIVITY7_DECISION_MODE", "final")),
        device=device,
    )


def _severity_for_label(label: str, duration_seconds: float) -> str:
    if label == "normal_driving":
        return "low"
    if label in {"phone_calling", "phone_call", "phone_interaction", "phone_use"}:
        if duration_seconds >= 8:
            return "critical"
        if duration_seconds >= 4:
            return "high"
        if duration_seconds >= 2:
            return "medium"
        return "low"
    if label == "mixed_risk":
        if duration_seconds >= 6:
            return "critical"
        if duration_seconds >= 3:
            return "high"
        return "medium"
    if label in {"eye_closure", "yawn", "yawning"}:
        if duration_seconds >= 4:
            return "critical"
        if duration_seconds >= 2.5:
            return "high"
        if duration_seconds >= 1.2:
            return "medium"
        return "low"
    if duration_seconds >= 6:
        return "critical"
    if duration_seconds >= 3.5:
        return "high"
    if duration_seconds >= 1.8:
        return "medium"
    return "low"


def _event_type_for_label(label: str) -> str:
    return label if label in MULTIMODAL_EVENT_LABELS else "mixed_risk"


def _merge_segments(segments, gap_seconds):
    merged = []
    for segment in segments:
        if not merged:
            merged.append(segment)
            continue
        last = merged[-1]
        same_label = last["label"] == segment["label"]
        close_enough = segment["start_seconds"] <= last["end_seconds"] + gap_seconds
        if same_label and close_enough:
            last["end_seconds"] = max(last["end_seconds"], segment["end_seconds"])
            last["confidence"] = max(last["confidence"], segment["confidence"])
            last["clip_count"] += 1
            last["class_scores"].append(segment["class_scores"])
            continue
        merged.append(segment)
    return merged


def _segment_to_event(session_id: int, segment: dict):
    label = segment["label"]
    if label == "normal_driving":
        return None
    duration = max(float(segment["end_seconds"] - segment["start_seconds"]), 0.1)
    event_type = _event_type_for_label(label)
    event = Event(
        session_id=session_id,
        event_type=event_type,
        start_time_seconds=round(float(segment["start_seconds"]), 2),
        end_time_seconds=round(float(segment["end_seconds"]), 2),
        duration_seconds=round(duration, 2),
        severity=_severity_for_label(event_type, duration),
        confidence=round(float(segment["confidence"]), 4),
        explanation=(
            f"Multimodal model predicted {event_type} from video, audio, and skeleton cues "
            f"across {segment['clip_count']} clip window(s)."
        ),
        risk_contribution=0.0,
    )
    return Event(**{**event.__dict__, "risk_contribution": contribution_for_event(event)})


def _load_package(model_config: MultimodalModelConfig):
    if not model_config.model_path.exists() and not model_config.package_path.exists():
        raise FileNotFoundError(
            f"Multimodal model package not found at {model_config.package_path} "
            f"or {model_config.model_path}."
        )
    package_path = model_config.package_path if model_config.package_path.exists() else model_config.model_path
    if _is_mmrt_backend(model_config.backend):
        return MMRTV2Package(package_path, device=model_config.device)
    return Activity7Package(package_path, device=model_config.device, decision_mode=model_config.decision_mode)


def _clip_windows(duration_seconds: float, clip_seconds: float, stride_seconds: float):
    if duration_seconds <= 0:
        return [(0.0, max(float(clip_seconds), 1.0))]
    clip_seconds = max(float(clip_seconds), 1.0)
    stride_seconds = max(float(stride_seconds), 0.5)
    windows = []
    start = 0.0
    while start < duration_seconds:
        end = min(duration_seconds, start + clip_seconds)
        windows.append((round(start, 3), round(max(end, start + 0.5), 3)))
        if end >= duration_seconds:
            break
        start += stride_seconds
        if start >= duration_seconds and windows[-1][1] < duration_seconds:
            windows.append((round(max(duration_seconds - clip_seconds, 0.0), 3), round(duration_seconds, 3)))
            break
    return windows


def analyze_video_with_multimodal_model(session_id, video_path, model_config):
    package = _load_package(model_config)
    clips = []
    events = []
    try:
        duration_seconds = 0.0
        try:
            import cv2

            capture = cv2.VideoCapture(str(video_path))
            if capture.isOpened():
                fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                duration_seconds = frames / fps if fps > 0 else 0.0
            capture.release()
        except Exception:
            duration_seconds = 0.0

        windows = _clip_windows(duration_seconds, model_config.clip_seconds, model_config.clip_stride_seconds)
        if not windows:
            windows = [(0.0, max(model_config.clip_seconds, 1.0))]

        clip_predictions = []
        audio_detected = False
        sampled_frames = 0
        pose_frames = 0

        for start_seconds, end_seconds in windows:
            clip_duration = max(end_seconds - start_seconds, 0.5)
            if _is_mmrt_backend(model_config.backend):
                clip = package.predict_clip(
                    video_path,
                    frame_count=model_config.frame_count,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                )
                label = clip.label
                confidence = clip.confidence
                probabilities = clip.probabilities
                audio_detected = audio_detected or clip.audio_track_detected
                sampled_frames += int(clip.frame_count)
                pose_frames += int(clip.skeleton_frames_detected)
            else:
                clip = package.predict_clip(
                    video_path,
                    sample_frames=model_config.frame_count,
                    start_seconds=start_seconds,
                    duration_seconds=clip_duration,
                )
                pred = clip.prediction
                label = pred.label
                confidence = pred.confidence
                probabilities = pred.probabilities
                audio_detected = audio_detected or clip.audio_track_detected
                sampled_frames += clip.sampled_frame_count
                pose_frames += clip.pose_frame_count

            clip_predictions.append(
                {
                    "start_seconds": float(getattr(clip, "start_seconds", start_seconds)),
                    "end_seconds": float(getattr(clip, "end_seconds", end_seconds)),
                    "label": label,
                    "confidence": confidence,
                    "probabilities": probabilities,
                }
            )
            if label != "normal_driving":
                events.append(
                    Event(
                        session_id=session_id,
                        event_type=_event_type_for_label(label),
                        start_time_seconds=round(float(getattr(clip, "start_seconds", start_seconds)), 2),
                        end_time_seconds=round(float(getattr(clip, "end_seconds", end_seconds)), 2),
                        duration_seconds=round(max(float(getattr(clip, "end_seconds", end_seconds)) - float(getattr(clip, "start_seconds", start_seconds)), 0.1), 2),
                        severity=_severity_for_label(_event_type_for_label(label), float(getattr(clip, "end_seconds", end_seconds)) - float(getattr(clip, "start_seconds", start_seconds))),
                        confidence=round(float(confidence), 4),
                        explanation=(
                            f"Multimodal model predicted {label} with audio, video, and skeleton cues "
                            f"in the clip from {float(getattr(clip, 'start_seconds', start_seconds)):.1f}s to {float(getattr(clip, 'end_seconds', end_seconds)):.1f}s."
                        ),
                        risk_contribution=0.0,
                    )
                )

        merged_events = _merge_segments(
            [
                {
                    "label": event.event_type,
                    "start_seconds": event.start_time_seconds,
                    "end_seconds": event.end_time_seconds,
                    "confidence": event.confidence,
                    "clip_count": 1,
                    "class_scores": [event.confidence],
                }
                for event in events
            ],
            gap_seconds=max(model_config.clip_stride_seconds * 0.75, 0.5),
        )
        final_events = []
        for segment in merged_events:
            event = _segment_to_event(session_id, segment)
            if event is not None:
                final_events.append(event)

        if not final_events and clip_predictions:
            top_clip = max(clip_predictions, key=lambda item: item["confidence"])
            if top_clip["label"] != "normal_driving":
                duration = max(top_clip["end_seconds"] - top_clip["start_seconds"], 0.1)
                event = Event(
                    session_id=session_id,
                    event_type=_event_type_for_label(top_clip["label"]),
                    start_time_seconds=round(top_clip["start_seconds"], 2),
                    end_time_seconds=round(top_clip["end_seconds"], 2),
                    duration_seconds=round(duration, 2),
                    severity=_severity_for_label(_event_type_for_label(top_clip["label"]), duration),
                    confidence=round(float(top_clip["confidence"]), 4),
                    explanation=(
                        f"Multimodal package predicted {top_clip['label']} for the strongest clip window."
                    ),
                    risk_contribution=0.0,
                )
                final_events.append(Event(**{**event.__dict__, "risk_contribution": contribution_for_event(event)}))

        best_label = clip_predictions[0]["label"] if clip_predictions else "normal_driving"
        best_confidence = clip_predictions[0]["confidence"] if clip_predictions else 0.0
        if clip_predictions:
            top = max(clip_predictions, key=lambda item: item["confidence"])
            best_label = top["label"]
            best_confidence = top["confidence"]

        metrics = {
            "processed_frame_count": sampled_frames,
            "average_processing_fps": (sampled_frames / duration_seconds) if duration_seconds else None,
            "face_detected_frame_count": pose_frames,
            "face_missing_frame_count": max(sampled_frames - pose_frames, 0),
            "average_ear": None,
            "average_mar": None,
            "min_ear": None,
            "max_mar": None,
            "phone_detected_frame_count": 0,
            "max_phone_confidence": None,
            "phone_usage_event_count": sum(1 for event in final_events if event.event_type in {"phone_usage", "phone_call", "phone_interaction", "phone_calling", "phone_use"}),
            "phone_detection_error": None,
            "multimodal_model_backend": model_config.backend,
            "multimodal_model_path": str(model_config.model_path),
            "multimodal_clip_count": len(windows),
            "audio_track_detected": int(audio_detected),
            "skeleton_frame_count": pose_frames,
            "clip_predictions": clip_predictions,
            "predicted_label": best_label,
            "predicted_confidence": best_confidence,
        }
        return MultimodalAnalysisResult(events=final_events, metrics=metrics, clips=clip_predictions)
    finally:
        package.close()
