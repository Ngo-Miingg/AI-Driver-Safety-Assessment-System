from __future__ import annotations

import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .live_multimodal import LiveMultimodalWindow
from .realtime_monitor import (
    _encode_snapshot,
    _live_risk_state,
    _open_camera,
    _render_skeleton_preview,
    _resize_frame,
)


logger = logging.getLogger(__name__)


CLASS_ORDER = ["normal_driving", "eye_closure", "yawning", "looking_away", "phone_use"]


def _blank_probabilities():
    return [{"class_name": name, "confidence": 0.0} for name in CLASS_ORDER]


def _audio_rms(samples):
    if samples is None:
        return 0.0
    if isinstance(samples, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(samples, dtype=np.float32)
    else:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(arr.astype(np.float32)))))


class MMRTModelLabManager:
    """Isolated live test runner for the MMRT model.

    This manager intentionally does not create DB sessions, reports, or events in
    the production realtime flow. It exists only to inspect the model inputs and
    raw outputs in a clean lab page.
    """

    def __init__(self):
        self._active = {}
        self._lock = threading.Lock()

    def start(self, app, socketio, sid, options=None):
        options = options or {}
        with self._lock:
            if sid in self._active:
                return False, "MMRT model lab is already running in this browser."
            stop_event = threading.Event()
            self._active[sid] = {
                "stop_event": stop_event,
                "audio_chunks": deque(maxlen=64),
                "audio_rms": 0.0,
            }

        task = socketio.start_background_task(
            self._run,
            app,
            socketio,
            sid,
            stop_event,
            dict(options),
        )
        with self._lock:
            active = self._active.get(sid)
            if active is not None:
                active["task"] = task
        return True, None

    def stop(self, sid):
        with self._lock:
            active = self._active.get(sid)
            if active is None:
                return False
            active["stop_event"].set()
        return True

    def discard(self, sid):
        self.stop(sid)
        with self._lock:
            self._active.pop(sid, None)

    def push_audio_chunk(self, sid, samples, sample_rate):
        rms = _audio_rms(samples)
        with self._lock:
            active = self._active.get(sid)
            if active is None:
                return
            active["audio_chunks"].append((samples, sample_rate))
            active["audio_rms"] = rms

    def _drain_audio_chunks(self, sid):
        with self._lock:
            active = self._active.get(sid)
            if active is None:
                return [], 0.0
            chunks = list(active["audio_chunks"])
            active["audio_chunks"].clear()
            return chunks, float(active.get("audio_rms", 0.0) or 0.0)

    def _run(self, app, socketio, sid, stop_event, options):
        capture = None
        runtime = None
        try:
            camera_index_value = options.get("camera_index")
            if camera_index_value is None or camera_index_value == "":
                camera_index_value = app.config.get("REALTIME_CAMERA_INDEX", 0)
            camera_index = int(camera_index_value)
            backend = str(options.get("behavior_model_backend") or "mmrt_multimodal_full_v2_t4x2").strip().lower()
            if not backend.startswith("mmrt"):
                backend = "mmrt_multimodal_full_v2_t4x2"

            model_path = Path(app.config.get("MMRT_V2_MODEL_PATH") or app.config.get("MULTIMODAL_MODEL_PATH"))
            device = app.config.get("MMRT_V2_DEVICE", app.config.get("MULTIMODAL_DEVICE", "cpu"))
            clip_seconds = float(app.config.get("MMRT_V2_CLIP_SECONDS", 3.0))
            stride_seconds = float(app.config.get("MMRT_V2_STRIDE_SECONDS", 1.5))
            frame_count = int(app.config.get("MMRT_V2_FRAME_COUNT", 16))

            capture, opened_index = _open_camera(camera_index)
            if capture is None or not capture.isOpened():
                socketio.emit("mmrt_lab_error", {"message": "Could not open webcam."}, to=sid)
                return

            runtime = LiveMultimodalWindow(
                model_path,
                backend=backend,
                device=device,
                clip_seconds=clip_seconds,
                stride_seconds=stride_seconds,
                frame_count=frame_count,
                audio_sample_rate=16000,
            )
            socketio.emit(
                "mmrt_lab_status",
                {
                    "status": "running",
                    "camera_index": opened_index,
                    "backend": backend,
                    "model_path": str(model_path),
                    "clip_seconds": clip_seconds,
                    "stride_seconds": stride_seconds,
                    "frame_count_required": frame_count,
                },
                to=sid,
            )

            start_time = time.monotonic()
            last_emit_at = 0.0
            last_snapshot_at = 0.0
            last_prediction_seen = None

            while not stop_event.is_set():
                chunks, audio_rms = self._drain_audio_chunks(sid)
                for samples, sample_rate in chunks:
                    runtime.push_audio_chunk(samples, sample_rate)

                ok, frame = capture.read()
                if not ok or frame is None:
                    socketio.emit("mmrt_lab_error", {"message": "Could not read a frame from webcam."}, to=sid)
                    time.sleep(0.15)
                    continue

                now = time.monotonic() - start_time
                prediction_payload = runtime.push_frame(frame, now)
                observation = runtime.latest_input_status or {}
                latest_prediction = runtime.last_prediction

                if (now - last_emit_at) < 0.25:
                    time.sleep(0.01)
                    continue
                last_emit_at = now

                pose_points = observation.get("pose_points") or []
                skeleton_frame = _render_skeleton_preview(pose_points)
                skeleton_image = _encode_snapshot(skeleton_frame)
                frame_image = _encode_snapshot(_resize_frame(frame, 960))

                payload = {
                    "timestamp_seconds": round(now, 2),
                    "state": "collecting",
                    "prediction_ready": False,
                    "backend": backend,
                    "model_path": str(model_path),
                    "frame_count_required": frame_count,
                    "buffered_frame_count": int(observation.get("sample_count", 0) or 0),
                    "face_detected": bool(observation.get("face_detected", False)),
                    "pose_present": bool(observation.get("pose_present", False)),
                    "audio_track_detected": bool(observation.get("audio_track_detected", False)),
                    "audio_rms": audio_rms,
                    "probabilities": _blank_probabilities(),
                    "frame_image": frame_image,
                    "skeleton_image": skeleton_image,
                    "top_label": None,
                    "top_confidence": None,
                    "stable_label": None,
                    "should_log_event": False,
                    "temporal_streak": 0,
                    "risk_score": None,
                    "safety_score": None,
                    "alert_level": None,
                    "warning": "Collecting a complete clip before model inference.",
                }

                if latest_prediction:
                    pred = latest_prediction["prediction"]
                    risk = _live_risk_state(pred, latest_prediction)
                    stable_label = pred.label if (latest_prediction.get("should_log_event") or pred.label == "normal_driving") else None
                    payload.update(
                        {
                            "state": "inferred",
                            "prediction_ready": True,
                            "top_label": pred.label,
                            "top_confidence": float(pred.confidence),
                            "probabilities": pred.probabilities,
                            "stable_label": stable_label,
                            "should_log_event": bool(latest_prediction.get("should_log_event")),
                            "temporal_streak": int(latest_prediction.get("temporal_streak") or 0),
                            "risk_score": risk["risk_score"],
                            "safety_score": risk["safety_score"],
                            "alert_level": risk["alert_level"],
                            "warning": latest_prediction.get("warning"),
                            "audio_track_detected": bool(latest_prediction.get("audio_track_detected")),
                            "face_frame_count": int(latest_prediction.get("face_frame_count") or 0),
                            "skeleton_frame_count": int(latest_prediction.get("pose_frame_count") or 0),
                        }
                    )
                    current_seen = (
                        round(float(latest_prediction.get("start_seconds") or 0.0), 2),
                        round(float(latest_prediction.get("end_seconds") or 0.0), 2),
                        pred.label,
                        round(float(pred.confidence or 0.0), 3),
                    )
                    if prediction_payload is not None and current_seen != last_prediction_seen:
                        last_prediction_seen = current_seen
                        socketio.emit(
                            "mmrt_lab_prediction",
                            {
                                "start_seconds": latest_prediction.get("start_seconds"),
                                "end_seconds": latest_prediction.get("end_seconds"),
                                "label": pred.label,
                                "confidence": float(pred.confidence),
                                "should_log_event": bool(latest_prediction.get("should_log_event")),
                                "temporal_streak": int(latest_prediction.get("temporal_streak") or 0),
                                "warning": latest_prediction.get("warning"),
                            },
                            to=sid,
                        )

                socketio.emit("mmrt_lab_update", payload, to=sid)

                if payload["prediction_ready"] and (now - last_snapshot_at) >= 0.8:
                    last_snapshot_at = now
                    socketio.emit(
                        "mmrt_lab_snapshot",
                        {
                            "timestamp_seconds": round(now, 2),
                            "image": frame_image,
                            "skeleton_image": skeleton_image,
                        },
                        to=sid,
                    )
        except Exception as exc:
            logger.exception("MMRT model lab failed sid=%s", sid)
            socketio.emit("mmrt_lab_error", {"message": str(exc)}, to=sid)
        finally:
            if runtime is not None:
                runtime.close()
            if capture is not None:
                capture.release()
            with self._lock:
                self._active.pop(sid, None)
            socketio.emit("mmrt_lab_status", {"status": "stopped"}, to=sid)


mmrt_model_lab_manager = MMRTModelLabManager()
