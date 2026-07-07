from __future__ import annotations

import base64
import logging
import sqlite3
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analysis_routes import _clear_previous_results, _store_events, _store_report, _store_risk_logs
from .analysis_models import Event
from .i18n import TRANSLATIONS, VALUE_TEXT_VI
from .mediapipe_analysis import face_analyzer_config, frame_results_to_events
from .report_generator import generate_report
from .risk_engine import contribution_for_event, summarize_session
from .settings import get_analysis_settings
from .ai.face_analyzer import MediaPipeFaceAnalyzer
from .ai.detectors import DetectorFactory
from .live_multimodal import LiveMultimodalWindow
from .yolo_phone_analysis import detections_to_phone_events


logger = logging.getLogger(__name__)
_UNICODE_FONT = Path("C:/Windows/Fonts/arial.ttf")


def _translation(language, key, default=None):
    if language == "vi":
        return TRANSLATIONS.get("vi", {}).get(key, default or key)
    return default or key


def _translate_value(language, value, default=None):
    if value is None:
        return default or "--"
    normalized = str(value).strip().lower().replace(" ", "_")
    if language == "vi":
        return VALUE_TEXT_VI.get(normalized, default or str(value))
    return default or str(value).replace("_", " ").title()


def _prediction_to_dict(prediction):
    probabilities = getattr(prediction, "probabilities", []) or []
    return {
        "label": getattr(prediction, "label", "normal_driving"),
        "class_index": getattr(prediction, "class_index", -1),
        "confidence": float(getattr(prediction, "confidence", 0.0) or 0.0),
        "probabilities": probabilities,
        "geom2_label": getattr(prediction, "geom2_label", getattr(prediction, "label", "normal_driving")),
        "geom2_confidence": float(getattr(prediction, "geom2_confidence", getattr(prediction, "confidence", 0.0)) or 0.0),
        "baseline_label": getattr(prediction, "baseline_label", getattr(prediction, "label", "normal_driving")),
        "baseline_confidence": float(getattr(prediction, "baseline_confidence", getattr(prediction, "confidence", 0.0)) or 0.0),
        "specialist_overrides": getattr(prediction, "specialist_overrides", {}) or {},
    }


def _connect_db(app):
    connection = sqlite3.connect(app.config["DATABASE_PATH"])
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _configure_capture(capture, target_fps):
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if target_fps:
            capture.set(cv2.CAP_PROP_FPS, float(target_fps))
    except Exception:
        pass


def _open_camera(camera_index):
    indexes = [camera_index]
    for fallback in (0, 1, 2):
        if fallback not in indexes:
            indexes.append(fallback)
    last_capture = None
    for index in indexes:
        capture = cv2.VideoCapture(int(index), getattr(cv2, "CAP_DSHOW", 0))
        if capture.isOpened():
            return capture, index
        last_capture = capture
        capture.release()
    return last_capture, camera_index


def _resize_frame(frame, width):
    if frame is None:
        return None
    if not width or width <= 0:
        return frame
    height, current_width = frame.shape[:2]
    if current_width <= width:
        return frame
    ratio = width / float(current_width)
    new_height = max(int(height * ratio), 1)
    return cv2.resize(frame, (int(width), new_height), interpolation=cv2.INTER_AREA)


def _encode_snapshot(frame):
    if frame is None:
        return None
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _encode_jpeg_bytes(frame, width=None, quality=80):
    if frame is None:
        return None
    encoded_frame = _resize_frame(frame, width) if width else frame
    ok, buffer = cv2.imencode(".jpg", encoded_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return buffer.tobytes()


def _overlay_snapshot(frame, risk_state, labels, language="en"):
    if frame is None:
        return None
    snapshot = frame.copy()
    state_label = _translation(language, "realtime.state", "State")
    risk_label = _translation(language, "realtime.risk", "Risk")
    safety_label = _translation(language, "realtime.safety", "Safety")
    alert_label = _translation(language, "realtime.alert", "Alert")
    lines = [
        f"{state_label}: {labels}",
        f"{risk_label}: {risk_state.get('risk_score', 0):.1f} | {safety_label}: {risk_state.get('safety_score', 100):.1f}",
        f"{alert_label}: {_translate_value(language, risk_state.get('alert_level', 'Normal'))}",
    ]
    rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(str(_UNICODE_FONT), 20)
    except Exception:
        font = ImageFont.load_default()
    y = 16
    for line in lines:
        draw.text((12, y), line, font=font, fill=(255, 255, 255))
        y += 26
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _render_skeleton_preview(points, width=640, height=360):
    frame = np.full((int(height), int(width), 3), (8, 19, 32), dtype=np.uint8)
    if not points:
        return frame

    edges = [
        (0, 1), (0, 2),
        (1, 2),
        (1, 3), (3, 5),
        (2, 4), (4, 6),
        (1, 7), (2, 8),
        (7, 8),
        (7, 9), (9, 11),
        (8, 10), (10, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    visible = [point for point in points if float(point.get("visibility", 0.0) or 0.0) >= 0.15]
    source = visible or points
    xs = [float(point.get("x", 0.0)) for point in source]
    ys = [float(point.get("y", 0.0)) for point in source]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max(max_x - min_x, 0.12)
    range_y = max(max_y - min_y, 0.18)
    pad_x = width * 0.14
    pad_y = height * 0.12
    scale_x = (width - pad_x * 2) / range_x
    scale_y = (height - pad_y * 2) / range_y
    scale = min(scale_x, scale_y)

    mapped = []
    for point in points:
        mapped.append(
            {
                "x": int(round(pad_x + (float(point.get("x", 0.0)) - min_x) * scale)),
                "y": int(round(pad_y + (float(point.get("y", 0.0)) - min_y) * scale)),
                "visibility": float(point.get("visibility", 0.0) or 0.0),
            }
        )

    for from_idx, to_idx in edges:
        if from_idx >= len(mapped) or to_idx >= len(mapped):
            continue
        a = mapped[from_idx]
        b = mapped[to_idx]
        if a["visibility"] < 0.12 or b["visibility"] < 0.12:
            continue
        cv2.line(frame, (a["x"], a["y"]), (b["x"], b["y"]), (125, 211, 252), 3, cv2.LINE_AA)

    for point in mapped:
        if point["visibility"] < 0.1:
            continue
        cv2.circle(frame, (point["x"], point["y"]), 6, (248, 250, 252), -1, cv2.LINE_AA)
        cv2.circle(frame, (point["x"], point["y"]), 6, (56, 189, 248), 2, cv2.LINE_AA)

    return frame


def _live_risk_state(prediction, payload):
    label = str(getattr(prediction, "label", "normal_driving"))
    confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
    should_log = bool(payload.get("should_log_event"))
    severity = str(payload.get("event_severity") or "low")
    warning_key = str(payload.get("warning_key") or "")

    if should_log and label != "normal_driving":
        risk_score = min(96.0, 30.0 + confidence * 40.0 + float(payload.get("risk_multiplier", 0.0)) * 8.0)
    elif label == "normal_driving":
        risk_score = max(2.0, (1.0 - confidence) * 12.0)
    elif warning_key == "realtime.warning_candidate":
        risk_score = min(24.0, 8.0 + confidence * 18.0)
    else:
        risk_score = min(38.0, 12.0 + confidence * 20.0)

    safety_score = max(0.0, 100.0 - risk_score)
    if should_log and severity == "critical":
        alert_level = "Danger"
    elif should_log and severity in {"high", "medium"}:
        alert_level = "Warning"
    elif warning_key == "realtime.warning_candidate":
        alert_level = "Caution"
    else:
        alert_level = "Normal"

    return {
        "risk_score": round(risk_score, 1),
        "safety_score": round(safety_score, 1),
        "alert_level": alert_level,
    }


def _event_key(event):
    return (
        event.event_type,
        round(float(event.start_time_seconds), 2),
        round(float(event.end_time_seconds), 2),
        round(float(event.confidence or 0.0), 2),
    )


def _combined_events(face_events, phone_events):
    combined = []
    seen = set()
    for event in [*face_events, *phone_events]:
        key = _event_key(event)
        if key in seen:
            continue
        seen.add(key)
        combined.append(event)
    combined.sort(key=lambda event: (event.start_time_seconds, event.end_time_seconds, event.event_type))
    return combined


def _attach_evidence_paths(events, evidence_map):
    if not evidence_map:
        return events
    updated = []
    for event in events:
        evidence_path = evidence_map.get(_event_key(event))
        if evidence_path:
            event = Event(**{**event.__dict__, "evidence_image_path": evidence_path})
        updated.append(event)
    return updated


def _active_labels(face_result, phone_frames, language="en"):
    labels = []
    if face_result:
        if face_result.get("face_detected"):
            labels.append(_translation(language, "realtime.face_detected", "Face detected"))
        else:
            labels.append(_translation(language, "realtime.face_missing", "Face missing"))
        if face_result.get("eye_closed"):
            labels.append(_translation(language, "realtime.eye_closed", "Eye closed"))
        if face_result.get("yawn_candidate"):
            labels.append(_translation(language, "realtime.yawn_candidate", "Yawn candidate"))
        if face_result.get("looking_away"):
            labels.append(_translation(language, "realtime.looking_away", "Looking away"))
        if face_result.get("head_down"):
            labels.append(_translation(language, "realtime.head_down", "Head down"))
    if phone_frames:
        labels.append(
            f"{_translation(language, 'realtime.phone_detected', 'Phone detected')} ({phone_frames[-1]['confidence']:.2f})"
        )
    return ", ".join(labels) if labels else _translation(language, "realtime.monitoring", "monitoring")


def _save_event_evidence(frame, event, evidence_dir, snapshot_width):
    if frame is None:
        return None
    resized = _resize_frame(frame, snapshot_width)
    if resized is None:
        return None
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp_ms = int(time.time() * 1000)
    filename = f"realtime_{event.event_type}_{timestamp_ms}.jpg"
    file_path = evidence_dir / filename
    cv2.imwrite(str(file_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return f"uploads/evidence/{filename}"


def _metrics_from_frame_results(frame_results):
    ear_values = [result["ear"] for result in frame_results if result.get("ear") is not None]
    mar_values = [result["mar"] for result in frame_results if result.get("mar") is not None]
    face_detected_count = sum(1 for result in frame_results if result.get("face_detected"))
    processed_count = len(frame_results)
    return {
        "processed_frame_count": processed_count,
        "face_detected_frame_count": face_detected_count,
        "face_missing_frame_count": max(processed_count - face_detected_count, 0),
        "average_ear": sum(ear_values) / len(ear_values) if ear_values else None,
        "average_mar": sum(mar_values) / len(mar_values) if mar_values else None,
        "min_ear": min(ear_values) if ear_values else None,
        "max_mar": max(mar_values) if mar_values else None,
    }


def _phone_metrics(phone_frames):
    return {
        "phone_detected_frame_count": len(phone_frames),
        "max_phone_confidence": max((frame["confidence"] for frame in phone_frames), default=None),
        "phone_usage_event_count": 0,
        "phone_detection_error": None,
    }


def _audio_metrics(audio_samples, audio_sample_rate):
    audio_detected = bool(audio_samples is not None and getattr(audio_samples, "size", 0))
    return {
        "audio_track_detected": int(audio_detected),
        "audio_sample_rate": int(audio_sample_rate or 0),
    }


class RealtimeSessionManager:
    def __init__(self):
        self._active = {}
        self._lock = threading.Lock()

    def set_language(self, sid, language):
        language = str(language or "en").strip().lower()
        if language not in {"en", "vi"}:
            language = "en"
        with self._lock:
            active = self._active.get(sid)
            if active is not None:
                active["language"] = language
            return language

    def _language_for(self, sid, default="en"):
        with self._lock:
            active = self._active.get(sid)
            if not active:
                return default
            language = str(active.get("language") or default).strip().lower()
        return language if language in {"en", "vi"} else default

    def start(self, app, socketio, sid, user_id, driver_id, options):
        options = dict(options or {})
        options["language"] = str(options.get("language") or "en").strip().lower()
        if options["language"] not in {"en", "vi"}:
            options["language"] = "en"
        with self._lock:
            if sid in self._active:
                return False, "Trình duyệt này đã có một phiên realtime đang chạy."

            stop_event = threading.Event()
            background_task = socketio.start_background_task(
                self._run,
                app,
                socketio,
                sid,
                user_id,
                driver_id,
                stop_event,
                options or {},
            )
            self._active[sid] = {
                "stop_event": stop_event,
                "task": background_task,
                "options": options or {},
                "language": options["language"],
                "user_id": user_id,
                "latest_preview_frame": None,
                "latest_preview_updated_at": None,
            }
        logger.info(
            "Starting realtime background task sid=%s user_id=%s driver_id=%s options=%s",
            sid,
            user_id,
            driver_id,
            options,
        )
        return True, None

    def stop(self, sid):
        with self._lock:
            active = self._active.get(sid)
            if not active:
                return False
            active["stop_event"].set()
            return True

    def discard(self, sid):
        return self.stop(sid)

    def push_audio_chunk(self, sid, samples, sample_rate):
        if samples is None:
            return
        if isinstance(samples, (bytes, bytearray, memoryview)):
            audio = np.frombuffer(samples, dtype=np.float32)
        else:
            audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        with self._lock:
            active = self._active.get(sid)
            if active is None:
                return
            audio_chunks = active.setdefault("audio_chunks", [])
            audio_chunks.append((audio, int(sample_rate or 16000)))

    def _drain_audio_chunks(self, sid):
        with self._lock:
            active = self._active.get(sid)
            if not active:
                return []
            chunks = active.get("audio_chunks", [])
            active["audio_chunks"] = []
            return chunks

    def update_preview_frame(self, sid, frame):
        payload = _encode_jpeg_bytes(frame, width=960, quality=80)
        if payload is None:
            return
        with self._lock:
            active = self._active.get(sid)
            if active is None:
                return
            active["latest_preview_frame"] = payload
            active["latest_preview_updated_at"] = time.monotonic()

    def get_latest_preview_frame(self, user_id):
        with self._lock:
            candidates = [
                active
                for active in self._active.values()
                if active.get("user_id") == user_id and active.get("latest_preview_frame")
            ]
            if not candidates:
                return None
            latest = max(candidates, key=lambda active: active.get("latest_preview_updated_at") or 0.0)
            return latest.get("latest_preview_frame")

    def has_active_session_for_user(self, user_id):
        with self._lock:
            return any(active.get("user_id") == user_id for active in self._active.values())

    def _run_multimodal_live(self, app, socketio, sid, user_id, driver_id, stop_event, options):
        db = None
        capture = None
        detector = None
        session_id = None
        runtime = None
        start_time = time.monotonic()
        last_snapshot_at = 0.0
        last_emitted_window = None
        predictions_for_report = []
        live_events = []
        phone_frames = []
        emitted_event_keys = set()
        last_phone_sample_at = 0.0
        try:
            with app.app_context():
                db = _connect_db(app)
                settings = get_analysis_settings(db, app)
                behavior_backend = str(options.get("behavior_model_backend") or "mmrt_multimodal_full_v2_t4x2").strip().lower()
                if behavior_backend.startswith("mmrt"):
                    model_path = app.config.get("MMRT_V2_MODEL_PATH") or app.config.get("MULTIMODAL_MODEL_PATH")
                    live_device = str(app.config.get("MMRT_V2_DEVICE", app.config.get("MULTIMODAL_DEVICE", "cpu")))
                else:
                    model_path = app.config.get("ACTIVITY7_MODEL_PACKAGE_PATH") or app.config.get("MULTIMODAL_MODEL_PATH")
                    live_device = str(app.config.get("MULTIMODAL_DEVICE", "cpu"))
                live_clip_seconds = float(app.config.get("MULTIMODAL_CLIP_SECONDS", 4.0))
                live_stride_seconds = float(app.config.get("MULTIMODAL_CLIP_STRIDE_SECONDS", 2.0))
                live_frame_count = int(app.config.get("MULTIMODAL_FRAME_COUNT", 16))
                phone_backend = str(options.get("phone_detector_backend") or "disabled")
                if behavior_backend.startswith("mmrt"):
                    phone_backend = "disabled"
                phone_detection_enabled = phone_backend != "disabled" and float(settings.get("realtime_yolo_enabled", 1.0)) > 0.0
                phone_sample_fps = max(float(settings.get("realtime_yolo_sample_fps", 1.0)), 0.1)
                phone_sample_interval = 1.0 / phone_sample_fps
                driver = db.execute(
                    "SELECT display_name FROM drivers WHERE id = ? AND user_id = ?",
                    (driver_id, user_id),
                ).fetchone()
                if driver is None:
                    socketio.emit("realtime_error", {"message": "Tài xế đã chọn không còn khả dụng."}, to=sid)
                    return

                db.execute(
                    """
                    INSERT INTO sessions (
                        user_id,
                        driver_id,
                        session_type,
                        status,
                        started_at,
                        analysis_mode,
                        multimodal_model_backend,
                        multimodal_model_path
                    ) VALUES (?, ?, 'realtime_webcam', 'active', CURRENT_TIMESTAMP, ?, ?, ?)
                    """,
                    (
                        user_id,
                        driver_id,
                        "multimodal_live",
                        behavior_backend,
                        str(model_path),
                    ),
                )
                session_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
                db.commit()

                camera_index = int(app.config.get("REALTIME_CAMERA_INDEX", 0))
                capture, opened_index = _open_camera(camera_index)
                if capture is None or not capture.isOpened():
                    raise RuntimeError(f"Không thể mở webcam ở chỉ số {camera_index}.")
                _configure_capture(capture, float(settings.get("realtime_target_fps", 8.0)))

                runtime = LiveMultimodalWindow(
                    model_path,
                    backend=behavior_backend,
                    device=live_device,
                    decision_mode=str(app.config.get("ACTIVITY7_DECISION_MODE", "final")),
                    clip_seconds=live_clip_seconds,
                    stride_seconds=live_stride_seconds,
                    frame_count=live_frame_count,
                    audio_sample_rate=16000,
                )
                if phone_detection_enabled:
                    detector = DetectorFactory.create(
                        phone_backend,
                        model_path=DetectorFactory.model_path_for_backend(app, phone_backend, realtime=True),
                        confidence_threshold=float(settings.get("realtime_yolo_confidence_threshold", 0.35)),
                        strict=True,
                    )

                socketio.emit(
                    "realtime_status",
                    {
                        "status": "running",
                        "session_id": session_id,
                        "analysis_mode": "multimodal_live",
                        "camera_index": opened_index,
                        "video_state": _translation(options.get("language", "en"), "realtime.video_live", "Video trực tiếp"),
                        "audio_state": _translation(options.get("language", "en"), "realtime.audio_live", "Âm thanh trực tiếp từ mic"),
                        "skeleton_state": _translation(options.get("language", "en"), "realtime.skeleton_live", "Khung xương live qua MediaPipe Pose"),
                        "live_mode_note": _translation(options.get("language", "en"), "realtime.multimodal_live_note", "Model multimodal đang nhận video, audio và skeleton thật."),
                    },
                    to=sid,
                )

                while not stop_event.is_set():
                    language = self._language_for(sid, options.get("language", "en"))
                    for samples, sample_rate in self._drain_audio_chunks(sid):
                        runtime.push_audio_chunk(samples, sample_rate)

                    ok, frame = capture.read()
                    if not ok or frame is None:
                        socketio.emit("realtime_error", {"message": "Không thể đọc khung hình từ webcam."}, to=sid)
                        time.sleep(0.15)
                        continue

                    self.update_preview_frame(sid, frame)
                    timestamp_seconds = time.monotonic() - start_time
                    current_phone_frames = []
                    if detector is not None and (timestamp_seconds - last_phone_sample_at) >= phone_sample_interval:
                        last_phone_sample_at = timestamp_seconds
                        detections = detector.detect_frame(frame)
                        phone_detections = [item for item in detections if item.get("is_phone")]
                        if phone_detections:
                            max_confidence = max(item["confidence"] for item in phone_detections)
                            phone_entry = {
                                "timestamp_seconds": timestamp_seconds,
                                "confidence": max_confidence,
                                "detections": phone_detections,
                            }
                            phone_frames.append(phone_entry)
                            current_phone_frames = [phone_entry]
                    prediction_payload = runtime.push_frame(frame, timestamp_seconds)
                    if prediction_payload is not None:
                        prediction = prediction_payload["prediction"]
                        pred_dict = _prediction_to_dict(prediction)
                        predictions_for_report.append(
                            {
                                "start_seconds": prediction_payload["start_seconds"],
                                "end_seconds": prediction_payload["end_seconds"],
                                "action_label": prediction_payload.get("action_label", prediction.label),
                                "action_confidence": prediction_payload.get("action_confidence", prediction.confidence),
                                "label": prediction.label,
                                "confidence": prediction.confidence,
                                "probabilities": pred_dict["probabilities"],
                                "audio_track_detected": prediction_payload["audio_track_detected"],
                                "sampled_frame_count": prediction_payload["video_sample_count"],
                                "pose_frame_count": prediction_payload["pose_frame_count"],
                                "driver_state": prediction_payload.get("driver_state"),
                                "driver_state_severity": prediction_payload.get("driver_state_severity"),
                                "warning_key": prediction_payload.get("warning_key"),
                                "warning": prediction_payload.get("warning"),
                                "temporal_streak": prediction_payload.get("temporal_streak"),
                                "temporal_buffer_size": prediction_payload.get("temporal_buffer_size"),
                            }
                        )
                        if prediction_payload.get("should_log_event") and prediction.label != "normal_driving":
                            event = Event(
                                session_id=session_id or 0,
                                event_type=prediction.label if prediction.label in {
                                    "phone_interaction",
                                    "phone_calling",
                                    "phone_use",
                                    "in_vehicle_display_interaction",
                                    "eating_distraction",
                                    "drinking_distraction",
                                    "visual_posture_distraction",
                                    "eye_closure",
                                    "yawning",
                                    "looking_away",
                                } else "mixed_risk",
                                start_time_seconds=round(prediction_payload["start_seconds"], 2),
                                end_time_seconds=round(prediction_payload["end_seconds"], 2),
                                duration_seconds=round(max(prediction_payload["end_seconds"] - prediction_payload["start_seconds"], 0.1), 2),
                                severity=str(prediction_payload.get("event_severity") or "low"),
                                confidence=round(float(prediction.confidence), 4),
                                explanation=str(
                                    prediction_payload.get("warning")
                                    or f"Multimodal model predicted {prediction.label} from live video, audio, and skeleton cues."
                                ),
                                risk_contribution=0.0,
                            )
                            live_events.append(Event(**{**event.__dict__, "risk_contribution": contribution_for_event(event)}))

                        last_emitted_window = (
                            round(prediction_payload["start_seconds"], 2),
                            round(prediction_payload["end_seconds"], 2),
                            prediction.label,
                        )

                    latest_prediction = runtime.last_prediction
                    if latest_prediction:
                        phone_events = detections_to_phone_events(
                            session_id or 0,
                            phone_frames,
                            merge_gap_seconds=2.0,
                            min_detections=2,
                        ) if phone_frames else []
                        combined_live_events = _combined_events(live_events, phone_events)
                        pred = latest_prediction["prediction"]
                        live_risk = _live_risk_state(pred, latest_prediction)
                        video_state = _translation(language, "realtime.video_live", "Video trực tiếp")
                        audio_state = (
                            _translation(language, "realtime.audio_live", "Âm thanh trực tiếp từ mic")
                            if latest_prediction["audio_track_detected"]
                            else _translation(language, "realtime.audio_waiting", "Đang chờ audio từ mic")
                        )
                        skeleton_state = (
                            _translation(language, "realtime.skeleton_live", "Khung xương live qua MediaPipe Pose")
                            if latest_prediction["pose_frame_count"] > 0
                            else _translation(language, "realtime.skeleton_waiting", "Đang chờ skeleton")
                        )
                        skeleton_frame = _render_skeleton_preview(latest_prediction.get("pose_points", []))
                        skeleton_image = _encode_snapshot(skeleton_frame) if skeleton_frame is not None else None
                        is_confirmed_state = bool(latest_prediction.get("should_log_event")) or pred.label == "normal_driving"
                        translated_action_label = (
                            _translate_value(language, pred.label, pred.label)
                            if is_confirmed_state
                            else _translation(language, "realtime.monitoring", "Monitoring")
                        )
                        if current_phone_frames:
                            translated_action_label = f"{translated_action_label}, {_translation(language, 'realtime.phone_detected', 'Phone detected')}"
                        snapshot_frame = _overlay_snapshot(
                            _resize_frame(frame, int(settings.get("snapshot_width", 320))),
                            live_risk,
                            f"{translated_action_label} ({pred.confidence:.2f})",
                            language=language,
                        )
                        if snapshot_frame is not None and (timestamp_seconds - last_snapshot_at) >= max(float(settings.get("snapshot_fps", 2.0)), 0.1) ** -1:
                            last_snapshot_at = timestamp_seconds
                            snapshot_url = _encode_snapshot(snapshot_frame)
                            if snapshot_url:
                                socketio.emit(
                                    "realtime_snapshot",
                                    {
                                        "session_id": session_id,
                                        "timestamp_seconds": round(timestamp_seconds, 2),
                                        "image": snapshot_url,
                                    },
                                    to=sid,
                                )

                        metrics = {
                            "session_id": session_id,
                            "timestamp_seconds": round(timestamp_seconds, 2),
                            "analysis_mode": "multimodal_live",
                            "frame_count": len(runtime._samples),
                            "face_detected": latest_prediction["pose_frame_count"] > 0,
                            "eye_closed": None,
                            "yawn_candidate": None,
                            "looking_away": pred.label in {"visual_posture_distraction", "looking_away"},
                            "head_down": None,
                            "ear": None,
                            "mar": None,
                            "risk_score": live_risk["risk_score"],
                            "safety_score": live_risk["safety_score"],
                            "alert_level": live_risk["alert_level"],
                            "event_count": len(combined_live_events),
                            "phone_detected": bool(current_phone_frames) or pred.label in {"phone_interaction", "phone_calling", "phone_use"},
                            "phone_detected_frame_count": len(phone_frames),
                            "phone_backend": phone_backend,
                            "behavior_backend": behavior_backend,
                            "active_labels": translated_action_label,
                            "action_label": pred.label,
                            "should_log_event": bool(latest_prediction.get("should_log_event")),
                            "warning": latest_prediction.get("warning"),
                            "warning_key": latest_prediction.get("warning_key"),
                            "event_type": latest_prediction.get("event_type"),
                            "event_severity": latest_prediction.get("event_severity"),
                            "temporal_buffer_size": latest_prediction.get("temporal_buffer_size"),
                            "temporal_streak": latest_prediction.get("temporal_streak"),
                            "pose_points": latest_prediction.get("pose_points", []),
                            "skeleton_image": skeleton_image,
                            "state": "running",
                            "video_state": video_state,
                            "audio_state": audio_state,
                            "skeleton_state": skeleton_state,
                            "live_mode_note": _translation(language, "realtime.multimodal_live_note", "Model multimodal đang nhận video, audio và skeleton thật."),
                            "predicted_label": pred.label,
                            "predicted_confidence": pred.confidence,
                            "audio_track_detected": latest_prediction["audio_track_detected"],
                            "skeleton_frame_count": latest_prediction["pose_frame_count"],
                            "multimodal_clip_count": len(predictions_for_report),
                        }
                        socketio.emit("realtime_update", metrics, to=sid)

                        for event in combined_live_events:
                            key = _event_key(event)
                            if key in emitted_event_keys:
                                continue
                            emitted_event_keys.add(key)
                            socketio.emit(
                                "realtime_event",
                                {
                                    "session_id": session_id,
                                    "event_type": event.event_type,
                                    "severity": event.severity,
                                    "start_time_seconds": event.start_time_seconds,
                                    "end_time_seconds": event.end_time_seconds,
                                    "duration_seconds": event.duration_seconds,
                                    "confidence": event.confidence,
                                    "explanation": event.explanation,
                                    "evidence_image_path": event.evidence_image_path,
                                },
                                to=sid,
                            )

                    time.sleep(max(1.0 / max(float(settings.get("realtime_target_fps", 8.0)), 1.0), 0.02))

                final_phone_events = detections_to_phone_events(
                    session_id or 0,
                    phone_frames,
                    merge_gap_seconds=2.0,
                    min_detections=2,
                ) if phone_frames else []
                final_events = _combined_events(live_events, final_phone_events)
                self._finalize_multimodal_live(
                    app,
                    socketio,
                    sid,
                    db,
                    session_id,
                    user_id,
                    driver["display_name"],
                    predictions_for_report,
                    final_events,
                    phone_frames,
                    runtime,
                )
        except Exception as exc:
            logger.exception("Realtime multimodal live failed for sid=%s", sid)
            if db is not None and session_id is not None:
                self._mark_failed(db, session_id, user_id, str(exc))
                db.commit()
            socketio.emit("realtime_error", {"message": f"Giám sát realtime thất bại: {exc}"}, to=sid)
        finally:
            if runtime is not None:
                runtime.close()
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
            if capture is not None:
                capture.release()
            if db is not None:
                db.close()
            with self._lock:
                self._active.pop(sid, None)

    def _finalize_multimodal_live(self, app, socketio, sid, db, session_id, user_id, driver_name, predictions, events, phone_frames, runtime):
        if session_id is None:
            return
        duration_seconds = max(predictions[-1]["end_seconds"] if predictions else 0.0, 1.0)
        summary, logs = summarize_session(session_id, events, duration_seconds)
        report = generate_report(
            {
                "id": session_id,
                "driver_name": driver_name,
                "video_duration_seconds": duration_seconds,
                "duration_seconds": duration_seconds,
            },
            events,
            summary,
            analysis_mode="multimodal_live",
        )
        processed_frame_count = sum(item.get("sampled_frame_count", 0) for item in predictions) if predictions else 0
        face_detected_count = sum(item.get("pose_frame_count", 0) for item in predictions) if predictions else 0
        metrics = {
            "processed_frame_count": processed_frame_count or (len(runtime._samples) if runtime else 0),
            "average_processing_fps": ((processed_frame_count or 0) / duration_seconds) if duration_seconds else None,
            "face_detected_frame_count": face_detected_count,
            "face_missing_frame_count": max((processed_frame_count or 0) - face_detected_count, 0),
            "average_ear": None,
            "average_mar": None,
            "min_ear": None,
            "max_mar": None,
            "phone_detected_frame_count": len(phone_frames or []),
            "max_phone_confidence": max((frame["confidence"] for frame in (phone_frames or [])), default=None),
            "phone_usage_event_count": sum(1 for event in events if event.event_type in {"phone_interaction", "phone_calling", "phone_use", "phone_call", "phone_usage"}),
            "phone_detection_error": None,
            "multimodal_model_backend": str(runtime.backend if runtime is not None else app.config.get("REALTIME_BEHAVIOR_MODEL_BACKEND", "mmrt_multimodal_full_v2_t4x2")),
            "multimodal_model_path": str(
                app.config.get("MMRT_V2_MODEL_PATH")
                if runtime is not None and str(getattr(runtime, "backend", "")).startswith("mmrt")
                else (app.config.get("ACTIVITY7_MODEL_PACKAGE_PATH") or app.config.get("MULTIMODAL_MODEL_PATH"))
            ),
            "multimodal_clip_count": len(predictions),
            "audio_track_detected": int(any(item.get("audio_track_detected") for item in predictions)),
            "skeleton_frame_count": sum(item.get("pose_frame_count", 0) for item in predictions),
        }
        _clear_previous_results(db, session_id)
        _store_events(db, events)
        _store_risk_logs(db, session_id, logs)
        _store_report(db, session_id, report)
        db.execute(
            """
            UPDATE sessions
            SET status = 'analyzed',
                analysis_mode = ?,
                multimodal_model_backend = ?,
                multimodal_model_path = ?,
                multimodal_clip_count = ?,
                audio_track_detected = ?,
                skeleton_frame_count = ?,
                processed_frame_count = ?,
                average_processing_fps = ?,
                face_detected_frame_count = ?,
                face_missing_frame_count = ?,
                phone_detected_frame_count = ?,
                max_phone_confidence = ?,
                phone_usage_event_count = ?,
                phone_detection_error = ?,
                final_risk_score = ?,
                final_safety_score = ?,
                final_alert_level = ?,
                fatigue_trend_label = ?,
                ended_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (
                "multimodal_live",
                metrics["multimodal_model_backend"],
                metrics["multimodal_model_path"],
                metrics["multimodal_clip_count"],
                metrics["audio_track_detected"],
                metrics["skeleton_frame_count"],
                metrics["processed_frame_count"],
                metrics["average_processing_fps"],
                metrics["face_detected_frame_count"],
                metrics["face_missing_frame_count"],
                metrics["phone_detected_frame_count"],
                metrics["max_phone_confidence"],
                metrics["phone_usage_event_count"],
                metrics["phone_detection_error"],
                summary.final_risk_score,
                summary.final_safety_score,
                summary.alert_level,
                summary.fatigue_trend,
                session_id,
                user_id,
            ),
        )
        db.commit()
        socketio.emit(
            "realtime_status",
            {
                "status": "completed",
                "session_id": session_id,
                "final_risk_score": summary.final_risk_score,
                "final_safety_score": summary.final_safety_score,
                "final_alert_level": summary.alert_level,
            },
            to=sid,
        )
        socketio.emit(
            "realtime_completed",
            {
                "session_id": session_id,
                "session_url": f"/sessions/{session_id}",
                "summary": {
                    "final_risk_score": summary.final_risk_score,
                    "final_safety_score": summary.final_safety_score,
                    "alert_level": summary.alert_level,
                    "event_count": summary.event_count,
                },
            },
            to=sid,
        )

    def _run(self, app, socketio, sid, user_id, driver_id, stop_event, options):
        db = None
        capture = None
        analyzer = None
        detector = None
        session_id = None
        start_time = time.monotonic()
        frame_results = []
        phone_frames = []
        combined_events = []
        evidence_map = {}
        last_snapshot_at = 0.0
        last_phone_sample_at = 0.0
        try:
            with app.app_context():
                db = _connect_db(app)
                settings = get_analysis_settings(db, app)
                behavior_backend = str(options.get("behavior_model_backend") or "builtin_mediapipe")
                if behavior_backend in {"activity7_multimodal_live", "mmrt_multimodal_full_v2_t4x2"} or behavior_backend.startswith("mmrt"):
                    self._run_multimodal_live(app, socketio, sid, user_id, driver_id, stop_event, options)
                    return
                phone_backend = str(options.get("phone_detector_backend") or "disabled")
                target_fps = float(settings.get("realtime_target_fps", 5.0))
                snapshot_enabled = float(settings.get("snapshot_preview_enabled", 1.0)) > 0.0
                evidence_enabled = float(settings.get("event_evidence_enabled", 1.0)) > 0.0
                snapshot_fps = max(float(settings.get("snapshot_fps", 2.0)), 0.1)
                snapshot_width = int(settings.get("snapshot_width", 480))
                yolo_enabled = phone_backend != "disabled" and float(settings.get("realtime_yolo_enabled", 1.0)) > 0.0
                yolo_sample_fps = max(float(settings.get("realtime_yolo_sample_fps", 1.0)), 0.1)
                yolo_interval = 1.0 / yolo_sample_fps
                analysis_mode = "realtime_mediapipe_yolo" if yolo_enabled else "realtime_mediapipe"

                driver = db.execute(
                    "SELECT display_name FROM drivers WHERE id = ? AND user_id = ?",
                    (driver_id, user_id),
                ).fetchone()
                if driver is None:
                    socketio.emit(
                        "realtime_error",
                        {"message": "Tài xế đã chọn không còn khả dụng."},
                        to=sid,
                    )
                    return

                db.execute(
                    """
                    INSERT INTO sessions (
                        user_id,
                        driver_id,
                        session_type,
                        status,
                        started_at,
                        analysis_mode
                    ) VALUES (?, ?, 'realtime_webcam', 'active', CURRENT_TIMESTAMP, ?)
                    """,
                    (user_id, driver_id, analysis_mode),
                )
                session_id = db.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]
                db.commit()

                camera_index = int(app.config.get("REALTIME_CAMERA_INDEX", 0))
                capture, opened_index = _open_camera(camera_index)
                if capture is None or not capture.isOpened():
                    raise RuntimeError(f"Không thể mở webcam ở chỉ số {camera_index}.")
                _configure_capture(capture, target_fps)

                analyzer = MediaPipeFaceAnalyzer(face_analyzer_config(settings))
                if yolo_enabled:
                    model_path = DetectorFactory.model_path_for_backend(app, phone_backend, realtime=True)
                    detector = DetectorFactory.create(
                        phone_backend,
                        model_path=model_path,
                        confidence_threshold=float(settings.get("realtime_yolo_confidence_threshold", 0.35)),
                        strict=True,
                    )

                language = self._language_for(sid, options.get("language", "en"))
                socketio.emit(
                    "realtime_status",
                    {
                        "status": "running",
                        "session_id": session_id,
                        "analysis_mode": analysis_mode,
                        "camera_index": opened_index,
                        "video_state": _translation(language, "realtime.video_live", "Video live"),
                        "audio_state": _translation(language, "realtime.audio_not_captured", "Audio not captured in live mode"),
                        "skeleton_state": _translation(language, "realtime.skeleton_not_rendered", "Skeleton not rendered in live mode"),
                        "live_mode_note": _translation(language, "realtime.live_mode_notice", "Live mode currently has no real audio or skeleton input."),
                    },
                    to=sid,
                )

                while not stop_event.is_set():
                    language = self._language_for(sid, options.get("language", "en"))
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        socketio.emit(
                            "realtime_error",
                            {"message": "Không thể đọc khung hình từ webcam."},
                            to=sid,
                        )
                        time.sleep(0.15)
                        continue

                    self.update_preview_frame(sid, frame)
                    timestamp_seconds = time.monotonic() - start_time
                    face_result = analyzer.analyze_frame(frame, timestamp_seconds)
                    frame_results.append(face_result)

                    current_phone_frames = []
                    if detector is not None and (timestamp_seconds - last_phone_sample_at) >= yolo_interval:
                        last_phone_sample_at = timestamp_seconds
                        detections = detector.detect_frame(frame)
                        phone_detections = [item for item in detections if item.get("is_phone")]
                        if phone_detections:
                            max_confidence = max(item["confidence"] for item in phone_detections)
                            phone_entry = {
                                "timestamp_seconds": timestamp_seconds,
                                "confidence": max_confidence,
                                "detections": phone_detections,
                            }
                            phone_frames.append(phone_entry)
                            current_phone_frames = [phone_entry]

                    face_events = frame_results_to_events(session_id or 0, frame_results, settings)
                    phone_events = detections_to_phone_events(
                        session_id or 0,
                        phone_frames,
                        merge_gap_seconds=2.0,
                        min_detections=2,
                    ) if phone_frames else []
                    updated_events = _combined_events(face_events, phone_events)
                    event_key_set = {_event_key(event) for event in combined_events}
                    new_events = [event for event in updated_events if _event_key(event) not in event_key_set]
                    if new_events and evidence_enabled:
                        for event in new_events:
                            evidence_path = _save_event_evidence(frame, event, Path(app.config["UPLOAD_FOLDER"]) / "evidence", snapshot_width)
                            if evidence_path:
                                evidence_map[_event_key(event)] = evidence_path
                    combined_events = _attach_evidence_paths(updated_events, evidence_map)

                    duration = max(timestamp_seconds, 1.0)
                    summary, logs = summarize_session(session_id or 0, combined_events, duration)
                    metrics = _metrics_from_frame_results(frame_results)
                    metrics.update(_phone_metrics(phone_frames))
                    metrics["average_processing_fps"] = metrics["processed_frame_count"] / duration if duration else None
                    metrics["phone_usage_event_count"] = sum(1 for event in combined_events if event.event_type in {"phone_usage", "phone_call", "phone_interaction", "phone_calling", "phone_use"})

                    labels = _active_labels(face_result, current_phone_frames, language=language)
                    snapshot_frame = None
                    if snapshot_enabled and (timestamp_seconds - last_snapshot_at) >= (1.0 / snapshot_fps):
                        last_snapshot_at = timestamp_seconds
                        snapshot_frame = _overlay_snapshot(_resize_frame(frame, snapshot_width), {
                            "risk_score": summary.final_risk_score,
                            "safety_score": summary.final_safety_score,
                            "alert_level": summary.alert_level,
                        }, labels, language=language)
                        snapshot_url = _encode_snapshot(snapshot_frame)
                        if snapshot_url:
                            socketio.emit(
                                "realtime_snapshot",
                                {
                                    "session_id": session_id,
                                    "timestamp_seconds": round(timestamp_seconds, 2),
                                    "image": snapshot_url,
                                },
                                to=sid,
                            )

                    socketio.emit(
                        "realtime_update",
                        {
                            "session_id": session_id,
                            "timestamp_seconds": round(timestamp_seconds, 2),
                            "analysis_mode": analysis_mode,
                            "frame_count": metrics["processed_frame_count"],
                            "face_detected": face_result.get("face_detected", False),
                            "eye_closed": face_result.get("eye_closed", False),
                            "yawn_candidate": face_result.get("yawn_candidate", False),
                            "looking_away": face_result.get("looking_away", False),
                            "head_down": face_result.get("head_down", False),
                            "ear": face_result.get("ear"),
                            "mar": face_result.get("mar"),
                            "head_pose_label": face_result.get("head_pose_label"),
                            "risk_score": summary.final_risk_score,
                            "safety_score": summary.final_safety_score,
                            "alert_level": summary.alert_level,
                            "event_count": summary.event_count,
                            "phone_detected": metrics["phone_detected_frame_count"] > 0,
                            "phone_detected_frame_count": metrics["phone_detected_frame_count"],
                            "phone_backend": phone_backend,
                            "behavior_backend": behavior_backend,
                            "active_labels": labels,
                            "state": "running",
                            "video_state": _translation(language, "realtime.video_live", "Video live"),
                            "audio_state": _translation(language, "realtime.audio_not_captured", "Audio not captured in live mode"),
                            "skeleton_state": _translation(language, "realtime.skeleton_not_rendered", "Skeleton not rendered in live mode"),
                            "live_mode_note": _translation(language, "realtime.live_mode_notice", "Live mode currently has no real audio or skeleton input."),
                        },
                        to=sid,
                    )

                    for event in new_events:
                        evidence_path = evidence_map.get(_event_key(event))
                        payload = {
                            "session_id": session_id,
                            "event_type": event.event_type,
                            "severity": event.severity,
                            "start_time_seconds": event.start_time_seconds,
                            "end_time_seconds": event.end_time_seconds,
                            "duration_seconds": event.duration_seconds,
                            "confidence": event.confidence,
                            "explanation": event.explanation,
                            "evidence_image_path": evidence_path or event.evidence_image_path,
                        }
                        socketio.emit("realtime_event", payload, to=sid)

                    time.sleep(max(1.0 / max(target_fps, 1.0), 0.02))

                self._finalize(
                    app,
                    socketio,
                    sid,
                    db,
                    session_id,
                    user_id,
                    driver["display_name"],
                    frame_results,
                    combined_events,
                    phone_frames,
                    settings,
                    analysis_mode,
                )
        except Exception as exc:
            logger.exception("Realtime monitoring failed for sid=%s", sid)
            if db is not None and session_id is not None:
                self._mark_failed(db, session_id, user_id, str(exc))
                db.commit()
            socketio.emit(
                "realtime_error",
                {"message": f"Giám sát realtime thất bại: {exc}"},
                to=sid,
            )
        finally:
            if analyzer is not None:
                analyzer.close()
            if capture is not None:
                capture.release()
            if db is not None:
                db.close()
            with self._lock:
                self._active.pop(sid, None)

    def _finalize(
        self,
        app,
        socketio,
        sid,
        db,
        session_id,
        user_id,
        driver_name,
        frame_results,
        events,
        phone_frames,
        settings,
        analysis_mode,
    ):
        duration_seconds = max((frame_results[-1]["timestamp_seconds"] if frame_results else 0.0), 1.0)
        summary, logs = summarize_session(session_id, events, duration_seconds)
        report = generate_report(
            {
                "id": session_id,
                "driver_name": driver_name,
                "video_duration_seconds": duration_seconds,
                "duration_seconds": duration_seconds,
            },
            events,
            summary,
            analysis_mode=analysis_mode,
        )
        metrics = _metrics_from_frame_results(frame_results)
        metrics.update(_phone_metrics(phone_frames))
        metrics["average_processing_fps"] = metrics["processed_frame_count"] / duration_seconds if duration_seconds else None
        metrics["phone_usage_event_count"] = sum(1 for event in events if event.event_type in {"phone_usage", "phone_call", "phone_interaction", "phone_calling"})

        _clear_previous_results(db, session_id)
        _store_events(db, events)
        _store_risk_logs(db, session_id, logs)
        _store_report(db, session_id, report)
        db.execute(
            """
            UPDATE sessions
            SET status = 'analyzed',
                analysis_mode = ?,
                processed_frame_count = ?,
                average_processing_fps = ?,
                face_detected_frame_count = ?,
                face_missing_frame_count = ?,
                average_ear = ?,
                average_mar = ?,
                min_ear = ?,
                max_mar = ?,
                phone_detected_frame_count = ?,
                max_phone_confidence = ?,
                phone_usage_event_count = ?,
                phone_detection_error = ?,
                final_risk_score = ?,
                final_safety_score = ?,
                final_alert_level = ?,
                fatigue_trend_label = ?,
                ended_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (
                analysis_mode,
                metrics["processed_frame_count"],
                metrics["average_processing_fps"],
                metrics["face_detected_frame_count"],
                metrics["face_missing_frame_count"],
                metrics["average_ear"],
                metrics["average_mar"],
                metrics["min_ear"],
                metrics["max_mar"],
                metrics["phone_detected_frame_count"],
                metrics["max_phone_confidence"],
                metrics["phone_usage_event_count"],
                metrics.get("phone_detection_error"),
                summary.final_risk_score,
                summary.final_safety_score,
                summary.alert_level,
                summary.fatigue_trend,
                session_id,
                user_id,
            ),
        )
        db.commit()
        socketio.emit(
            "realtime_status",
            {
                "status": "completed",
                "session_id": session_id,
                "final_risk_score": summary.final_risk_score,
                "final_safety_score": summary.final_safety_score,
                "final_alert_level": summary.alert_level,
            },
            to=sid,
        )
        socketio.emit(
            "realtime_completed",
            {
                "session_id": session_id,
                "session_url": f"/sessions/{session_id}",
                "summary": {
                    "final_risk_score": summary.final_risk_score,
                    "final_safety_score": summary.final_safety_score,
                    "alert_level": summary.alert_level,
                    "event_count": summary.event_count,
                },
            },
            to=sid,
        )

    def _mark_failed(self, db, session_id, user_id, reason):
        db.execute(
            """
            UPDATE sessions
            SET status = 'failed',
                analysis_error = ?,
                failure_reason = ?,
                ended_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (reason, reason, session_id, user_id),
        )


realtime_manager = RealtimeSessionManager()
