from __future__ import annotations

"""Runtime contract for the report-backed Driver Safety v1 models.

The six facial features below intentionally mirror the training repository.
Changing landmark indices, formulas, feature order, or scaler invalidates the
input distribution expected by ``best_model.h5``.
"""

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any
import warnings

import cv2
import numpy as np


warnings.filterwarnings(
    "ignore",
    message=r"SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
    module=r"google\.protobuf\.symbol_database",
)


FACEMESH_LIPS = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308)
FACEMESH_LEFT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398)
FACEMESH_RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173)
FEATURE_ORDER = ("left_ear", "right_ear", "mar", "pitch_degrees", "yaw_degrees", "roll_degrees")


def _resolve_torch_device(configured_device: str | None) -> str:
    requested = str(configured_device or "auto").strip().lower()
    try:
        import torch

        cuda_ready = bool(torch.cuda.is_available())
    except Exception:
        cuda_ready = False
    if requested in {"auto", ""}:
        return "cuda:0" if cuda_ready else "cpu"
    if requested.startswith("cuda") and not cuda_ready:
        return "cpu"
    return configured_device or "cpu"


@dataclass(frozen=True)
class DriverSafetyObservation:
    face_detected: bool
    face_box: tuple[int, int, int, int] | None
    face_mesh_detected: bool
    features: np.ndarray | None
    buffer_size: int
    drowsiness_probability: float | None
    model_ready: bool
    buffer_duration_seconds: float | None = None
    feature_sample_fps: float | None = None
    prediction_updated: bool = False
    face_behavior_geometry: dict[str, float] | None = None


def model_readiness(app) -> dict[str, Any]:
    from .driver_action_openvino import action_model_readiness

    paths = {
        "face_detector": Path(app.config["DRIVER_SAFETY_FACE_DETECTOR_PATH"]),
        "face_verifier": Path(app.config["DRIVER_SAFETY_FACE_VERIFIER_PATH"]),
        "drowsiness_model": Path(app.config["DRIVER_SAFETY_DROWSINESS_MODEL_PATH"]),
        "scaler": Path(app.config["DRIVER_SAFETY_SCALER_PATH"]),
    }
    available = {name: path.exists() and path.is_file() for name, path in paths.items()}
    drowsiness_artifacts_ready = all(
        available[name] for name in ("face_detector", "drowsiness_model", "scaler")
    )
    try:
        import tensorflow as tf  # noqa: F401
        import tf_keras as tf_keras_runtime  # noqa: F401

        verifier_runtime_available = True
        verifier_runtime_error = None
    except Exception as exc:
        verifier_runtime_available = False
        verifier_runtime_error = str(exc)
    action_readiness = action_model_readiness(app)
    model_status = {
        name: {"path": str(path), "available": available[name]}
        for name, path in paths.items()
    }
    model_status.update(
        {
            "action_encoder": {
                "path": app.config["DRIVER_ACTION_ENCODER_PATH"],
                "available": action_readiness["models"]["action_encoder"]["available"],
            },
            "action_decoder": {
                "path": app.config["DRIVER_ACTION_DECODER_PATH"],
                "available": action_readiness["models"]["action_decoder"]["available"],
            },
        }
    )
    return {
        "pipeline": "driver_safety_v1",
        "models": model_status,
        "drowsiness_artifacts_ready": drowsiness_artifacts_ready,
        "drowsiness_ready": drowsiness_artifacts_ready,
        "action_recognition_ready": action_readiness["available"],
        "action_recognition_runtime_available": action_readiness["runtime_available"],
        "action_recognition_runtime_error": action_readiness["runtime_error"],
        "action_recognition_classes": action_readiness["openvino_classes"],
        "identity_artifact_available": available["face_verifier"],
        "identity_runtime_available": verifier_runtime_available,
        "identity_runtime_error": verifier_runtime_error,
        "identity_threshold": float(app.config.get("DRIVER_SAFETY_FACE_VERIFY_THRESHOLD", 0.94)),
        "identity_ready": available["face_verifier"] and verifier_runtime_available,
        "identity_status": (
            "ready_for_enrollment"
            if available["face_verifier"] and verifier_runtime_available
            else "requires_tensorflow_runtime"
        ),
        "face_detector_device": _resolve_torch_device(app.config.get("DRIVER_SAFETY_FACE_DEVICE", "auto")),
    }


class DriverSafetyV1Runtime:
    def __init__(self, app):
        self.app = app
        self.sequence_length = int(app.config["DRIVER_SAFETY_SEQUENCE_LENGTH"])
        self.threshold = float(app.config["DRIVER_SAFETY_DECISION_THRESHOLD"])
        self.inference_stride = max(int(app.config.get("DRIVER_SAFETY_INFERENCE_STRIDE", 2)), 1)
        self.face_detector_fps = max(float(app.config.get("DRIVER_SAFETY_FACE_DETECTOR_FPS", 2.0)), 0.1)
        self.face_imgsz = int(app.config.get("DRIVER_SAFETY_FACE_IMGSZ", 640))
        self.face_device = _resolve_torch_device(app.config.get("DRIVER_SAFETY_FACE_DEVICE", "auto"))
        self.feature_gap_reset_seconds = max(
            float(app.config.get("DRIVER_SAFETY_FEATURE_GAP_RESET_SECONDS", 0.25)), 0.05
        )
        self.buffer: deque[np.ndarray] = deque(maxlen=self.sequence_length)
        self._feature_timestamps: deque[float] = deque(maxlen=self.sequence_length)
        self._detector = None
        self._face_mesh = None
        self._drowsiness_model = None
        self._drowsiness_compiled = None
        self._drowsiness_input = None
        self._drowsiness_output = None
        self._drowsiness_backend = None
        self._drowsiness_device = None
        self._scaler = None
        self._drowsiness_error: str | None = None
        self._load_error: str | None = None
        self._last_face_box: tuple[int, int, int, int] | None = None
        self._last_detector_at = float("-inf")
        self._last_feature_at: float | None = None
        self._valid_feature_count = 0
        self._last_inference_feature_count = 0
        self._last_probability: float | None = None
        self._last_face_behavior_geometry: dict[str, float] | None = None
        self._load_runtime()

    @property
    def model_ready(self) -> bool:
        return (
            self._load_error is None
            and self._drowsiness_error is None
            and (self._drowsiness_model is not None or self._drowsiness_compiled is not None)
            and self._scaler is not None
        )

    @property
    def load_error(self) -> str | None:
        return self._load_error or self._drowsiness_error

    def _load_runtime(self):
        readiness = model_readiness(self.app)
        if not readiness["models"]["face_detector"]["available"]:
            self._load_error = "Missing required artifact: face_detection.pt"
            return
        try:
            import mediapipe as mp
            from ultralytics import YOLO

            self._detector = YOLO(self.app.config["DRIVER_SAFETY_FACE_DETECTOR_PATH"])
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:
            self._load_error = f"Driver Safety v1 could not load: {exc}"
            return

        if not readiness["models"]["scaler"]["available"]:
            self._drowsiness_error = "Missing required artifact: scaler.joblib"
            return
        if not readiness["models"]["drowsiness_model"]["available"]:
            self._drowsiness_error = "Missing required artifact: best_model.h5"
            return
        try:
            import joblib

            self._scaler = joblib.load(self.app.config["DRIVER_SAFETY_SCALER_PATH"])
            openvino_path = Path(self.app.config.get("DRIVER_SAFETY_DROWSINESS_OPENVINO_PATH", ""))
            if openvino_path.exists() and openvino_path.with_suffix(".bin").exists():
                from .driver_action_openvino import _prepare_openvino_windows_runtime, resolve_openvino_device

                _prepare_openvino_windows_runtime()
                import openvino as ov

                core = ov.Core()
                self._drowsiness_device = resolve_openvino_device(
                    self.app.config.get("DRIVER_ACTION_DEVICE", "auto"),
                    list(core.available_devices),
                )
                self._drowsiness_compiled = core.compile_model(str(openvino_path), self._drowsiness_device)
                self._drowsiness_input = self._drowsiness_compiled.input(0)
                self._drowsiness_output = self._drowsiness_compiled.output(0)
                self._drowsiness_backend = "openvino"
                # The exported IR keeps a dynamic batch dimension. The
                # remaining dimensions are enforced by the runtime input
                # tensor below and by the source model contract.
            else:
                os.environ.setdefault("KERAS_BACKEND", "torch")
                import keras

                @keras.saving.register_keras_serializable()
                class CompatLSTM(keras.layers.LSTM):
                    def __init__(self, *args, **kwargs):
                        kwargs.pop("time_major", None)
                        super().__init__(*args, **kwargs)

                self._drowsiness_model = keras.saving.load_model(
                    self.app.config["DRIVER_SAFETY_DROWSINESS_MODEL_PATH"],
                    compile=False,
                    custom_objects={
                        "LSTM": CompatLSTM,
                        "CompatLSTM": CompatLSTM,
                        "Bidirectional": keras.layers.Bidirectional,
                        "Dropout": keras.layers.Dropout,
                        "Dense": keras.layers.Dense,
                        "InputLayer": keras.layers.InputLayer,
                    },
                )
                if tuple(self._drowsiness_model.input_shape[1:]) != (self.sequence_length, len(FEATURE_ORDER)):
                    raise ValueError(
                        f"Unexpected model input shape {self._drowsiness_model.input_shape}; "
                        f"expected (None, {self.sequence_length}, {len(FEATURE_ORDER)})."
                    )
                self._drowsiness_backend = "keras"
            if int(getattr(self._scaler, "n_features_in_", -1)) != len(FEATURE_ORDER):
                raise ValueError("The persisted scaler does not contain the required six features.")
        except Exception as exc:
            self._drowsiness_error = f"Drowsiness model unavailable: {exc}"

    def close(self):
        if self._face_mesh is not None:
            self._face_mesh.close()

    @staticmethod
    def _landmark_distance(a, b) -> float:
        return float(np.linalg.norm([a.x - b.x, a.y - b.y]))

    @classmethod
    def _calculate_ear(cls, landmarks, indices) -> float:
        # Exact mapping used by datapreprocessing/utils.py in the training source.
        points = [landmarks[index] for index in indices]
        p1, p2, p3, p4, p5, p6 = points[8], points[13], points[11], points[0], points[3], points[4]
        return (
            cls._landmark_distance(p2, p6) + cls._landmark_distance(p3, p5)
        ) / (2.0 * cls._landmark_distance(p1, p4) + 1e-6)

    @classmethod
    def _calculate_mar(cls, landmarks) -> float:
        p1, p4 = landmarks[61], landmarks[291]
        p2, p6 = landmarks[0], landmarks[17]
        p3, p5 = landmarks[40], landmarks[37]
        mouth_width = cls._landmark_distance(p1, p4)
        mouth_height = (cls._landmark_distance(p2, p6) + cls._landmark_distance(p3, p5)) / 2.0
        return mouth_height / (mouth_width + 1e-6)

    @staticmethod
    def _get_head_pose(landmarks, image_shape) -> tuple[float, float, float] | None:
        height, width = image_shape
        face_3d = np.asarray(
            [
                (0.0, 0.0, 0.0),
                (0.0, -330.0, -65.0),
                (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0),
                (-150.0, -150.0, -125.0),
                (150.0, -150.0, -125.0),
            ],
            dtype=np.float64,
        )
        face_2d = np.asarray(
            [
                (landmarks[4].x * width, landmarks[4].y * height),
                (landmarks[152].x * width, landmarks[152].y * height),
                (landmarks[263].x * width, landmarks[263].y * height),
                (landmarks[33].x * width, landmarks[33].y * height),
                (landmarks[61].x * width, landmarks[61].y * height),
                (landmarks[291].x * width, landmarks[291].y * height),
            ],
            dtype=np.float64,
        )
        camera = np.asarray(
            ((width, 0, width / 2), (0, width, height / 2), (0, 0, 1)),
            dtype=np.float64,
        )
        success, rotation_vector, _ = cv2.solvePnP(
            face_3d, face_2d, camera, np.zeros((4, 1), dtype=np.float64)
        )
        if not success:
            return None
        rotation, _ = cv2.Rodrigues(rotation_vector)
        angles, *_ = cv2.RQDecomp3x3(rotation)
        return tuple(float(value) for value in angles)

    @staticmethod
    def _largest_face(result) -> tuple[int, int, int, int] | None:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return None
        values = boxes.xyxy.detach().cpu().numpy()
        scores = boxes.conf.detach().cpu().numpy()
        areas = (values[:, 2] - values[:, 0]) * (values[:, 3] - values[:, 1])
        index = int(np.argmax(areas * scores))
        return tuple(values[index].astype(int).tolist())

    def _extract_features(self, frame: np.ndarray) -> np.ndarray | None:
        self._last_face_behavior_geometry = None
        if self._face_mesh is None:
            return None
        result = self._face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        pose = self._get_head_pose(landmarks, frame.shape[:2])
        if pose is None:
            return None
        top = landmarks[10]
        bottom = landmarks[152]
        left = landmarks[234]
        right = landmarks[454]
        nose = landmarks[1]
        face_width = max(abs(right.x - left.x), 1e-6)
        face_height = max(abs(bottom.y - top.y), 1e-6)
        horizontal_offset = nose.x - ((left.x + right.x) / 2.0)
        vertical_offset = nose.y - ((top.y + bottom.y) / 2.0)
        mar = self._calculate_mar(landmarks)
        self._last_face_behavior_geometry = {
            "mar": float(mar),
            "horizontal_offset": float(horizontal_offset / face_width),
            "vertical_offset": float(vertical_offset / face_height),
            "pitch_degrees": float(pose[0]),
            "yaw_degrees": float(pose[1]),
            "roll_degrees": float(pose[2]),
        }
        features = np.asarray(
            [
                self._calculate_ear(landmarks, FACEMESH_LEFT_EYE),
                self._calculate_ear(landmarks, FACEMESH_RIGHT_EYE),
                mar,
                *pose,
            ],
            dtype=np.float32,
        )
        return features if np.all(np.isfinite(features)) else None

    def _detect_face_box(self, frame: np.ndarray, timestamp_seconds: float):
        if timestamp_seconds - self._last_detector_at < 1.0 / self.face_detector_fps:
            return
        self._last_detector_at = timestamp_seconds
        result = self._detector(
            frame,
            verbose=False,
            conf=float(self.app.config["DRIVER_SAFETY_FACE_CONFIDENCE"]),
            imgsz=self.face_imgsz,
            device=self.face_device,
        )[0]
        self._last_face_box = self._largest_face(result)

    def _reset_temporal_state(self):
        self.buffer.clear()
        self._feature_timestamps.clear()
        self._last_probability = None
        self._last_inference_feature_count = self._valid_feature_count

    def observe(self, frame: np.ndarray, timestamp_seconds: float | None = None) -> DriverSafetyObservation:
        if self._detector is None:
            return DriverSafetyObservation(False, None, False, None, len(self.buffer), None, False)
        timestamp_seconds = time.monotonic() if timestamp_seconds is None else float(timestamp_seconds)
        self._detect_face_box(frame, timestamp_seconds)
        features = self._extract_features(frame)

        if features is None:
            if self._last_feature_at is None or timestamp_seconds - self._last_feature_at >= self.feature_gap_reset_seconds:
                self._reset_temporal_state()
            return DriverSafetyObservation(
                self._last_face_box is not None,
                self._last_face_box,
                False,
                None,
                len(self.buffer),
                self._last_probability,
                self.model_ready,
            )

        if self._last_feature_at is not None and timestamp_seconds - self._last_feature_at >= self.feature_gap_reset_seconds:
            self._reset_temporal_state()
        self._last_feature_at = timestamp_seconds
        self.buffer.append(features)
        self._feature_timestamps.append(timestamp_seconds)
        self._valid_feature_count += 1

        prediction_updated = False
        if (
            self.model_ready
            and len(self.buffer) == self.sequence_length
            and self._valid_feature_count - self._last_inference_feature_count >= self.inference_stride
        ):
            sequence = np.asarray(self.buffer, dtype=np.float32)
            scaled = self._scaler.transform(sequence).reshape(1, self.sequence_length, len(FEATURE_ORDER))
            if self._drowsiness_compiled is not None:
                probability = float(
                    np.asarray(self._drowsiness_compiled({self._drowsiness_input: scaled})[self._drowsiness_output])
                    .reshape(-1)[0]
                )
            else:
                probability = float(np.asarray(self._drowsiness_model.predict(scaled, verbose=0)).reshape(-1)[0])
            self._last_probability = float(np.clip(probability, 0.0, 1.0))
            self._last_inference_feature_count = self._valid_feature_count
            prediction_updated = True

        duration = None
        sample_fps = None
        if len(self._feature_timestamps) >= 2:
            duration = self._feature_timestamps[-1] - self._feature_timestamps[0]
            if duration > 0:
                sample_fps = (len(self._feature_timestamps) - 1) / duration
        return DriverSafetyObservation(
            True,
            self._last_face_box,
            True,
            features,
            len(self.buffer),
            self._last_probability,
            self.model_ready,
            duration,
            sample_fps,
            prediction_updated,
            self._last_face_behavior_geometry,
        )
