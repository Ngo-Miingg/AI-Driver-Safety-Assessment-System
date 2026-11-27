from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_VERIFIER_CACHE: dict[tuple[str, float], "FaceIdentityVerifier"] = {}


@dataclass(frozen=True)
class FaceIdentityResult:
    status: str
    verified: bool
    distance: float | None = None
    threshold: float | None = None
    message: str | None = None


class FaceIdentityVerifier:
    """Face embedding runtime for the report-backed MobileNetV2 verifier."""

    def __init__(self, app):
        self.app = app
        self.threshold = float(app.config.get("DRIVER_SAFETY_FACE_VERIFY_THRESHOLD", 0.94))
        self.model_path = Path(app.config["DRIVER_SAFETY_FACE_VERIFIER_PATH"])
        self._model = None
        self._compiled_model = None
        self._compiled_input = None
        self._compiled_output = None
        self._device = None
        self._error: str | None = None
        self._load()

    @property
    def ready(self) -> bool:
        return (self._model is not None or self._compiled_model is not None) and self._error is None

    @property
    def error(self) -> str | None:
        return self._error

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model_path": str(self.model_path),
            "model_available": self.model_path.exists(),
            "runtime_backend": "openvino" if self._compiled_model is not None else "keras",
            "device": self._device,
            "threshold": self.threshold,
            "error": self._error,
        }

    def _load(self):
        if not self.model_path.exists():
            self._error = "Missing face_verification.h5"
            return
        try:
            openvino_path = Path(
                self.app.config.get(
                    "DRIVER_SAFETY_FACE_VERIFIER_OPENVINO_PATH",
                    self.model_path.parent / "openvino" / "face_verification.xml",
                )
            )
            if openvino_path.exists() and openvino_path.with_suffix(".bin").exists():
                from .driver_action_openvino import _prepare_openvino_windows_runtime, resolve_openvino_device

                _prepare_openvino_windows_runtime()
                import openvino as ov

                core = ov.Core()
                self._device = resolve_openvino_device(
                    self.app.config.get("DRIVER_ACTION_DEVICE", "auto"),
                    list(core.available_devices),
                )
                self._compiled_model = core.compile_model(str(openvino_path), self._device)
                self._compiled_input = self._compiled_model.input(0)
                self._compiled_output = self._compiled_model.output(0)
                return
            try:
                import tf_keras as keras_runtime
            except Exception:
                import tensorflow as tf

                keras_runtime = tf.keras

            self._model = keras_runtime.models.load_model(self.model_path, compile=False)
            input_shape = tuple(self._model.input_shape[1:])
            output_shape = tuple(self._model.output_shape[1:])
            if input_shape != (112, 112, 3):
                raise ValueError(f"Unexpected verifier input shape {self._model.input_shape}")
            if output_shape != (128,):
                raise ValueError(f"Unexpected verifier output shape {self._model.output_shape}")
        except Exception as exc:
            self._model = None
            self._error = f"Face verifier unavailable: {exc}"

    @staticmethod
    def crop_face(frame: np.ndarray, face_box: tuple[int, int, int, int] | None, padding: float = 0.18) -> np.ndarray | None:
        if frame is None or face_box is None:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face_box]
        box_w = max(x2 - x1, 1)
        box_h = max(y2 - y1, 1)
        pad_x = int(box_w * padding)
        pad_y = int(box_h * padding)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    @staticmethod
    def prepare_face(face_bgr: np.ndarray) -> np.ndarray | None:
        if face_bgr is None or face_bgr.size == 0:
            return None
        resized = cv2.resize(face_bgr, (112, 112), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb.astype("float32")[None, ...]

    def embed_face_crop(self, face_bgr: np.ndarray) -> np.ndarray:
        if not self.ready:
            raise RuntimeError(self._error or "Face verifier is not ready.")
        prepared = self.prepare_face(face_bgr)
        if prepared is None:
            raise ValueError("No valid face crop was provided.")
        if self._compiled_model is not None:
            embedding = np.asarray(
                self._compiled_model({self._compiled_input: prepared})[self._compiled_output],
                dtype=np.float32,
            ).reshape(-1)
        else:
            embedding = np.asarray(self._model.predict(prepared, verbose=0), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("The face verifier returned an invalid embedding.")
        return (embedding / norm).astype(np.float32)

    def embed_frame(self, frame: np.ndarray, face_box: tuple[int, int, int, int] | None) -> np.ndarray:
        crop = self.crop_face(frame, face_box)
        if crop is None:
            raise ValueError("No detected face is available for identity verification.")
        return self.embed_face_crop(crop)

    def compare(self, embedding: np.ndarray, enrolled_embedding: np.ndarray | None) -> FaceIdentityResult:
        if enrolled_embedding is None:
            return FaceIdentityResult(
                status="not_enrolled",
                verified=False,
                threshold=self.threshold,
                message="The selected driver has no enrolled face embedding.",
            )
        distance = float(np.linalg.norm(np.asarray(embedding, dtype=np.float32) - np.asarray(enrolled_embedding, dtype=np.float32)))
        verified = distance <= self.threshold
        return FaceIdentityResult(
            status="verified" if verified else "mismatch",
            verified=verified,
            distance=distance,
            threshold=self.threshold,
        )


def embedding_to_json(embedding: np.ndarray) -> str:
    import json

    values = [round(float(value), 8) for value in np.asarray(embedding, dtype=np.float32).reshape(-1)]
    return json.dumps(values, separators=(",", ":"))


def embedding_from_json(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    import json

    parsed = json.loads(value)
    embedding = np.asarray(parsed, dtype=np.float32).reshape(-1)
    if embedding.shape != (128,) or not np.all(np.isfinite(embedding)):
        return None
    norm = float(np.linalg.norm(embedding))
    if norm <= 0:
        return None
    return (embedding / norm).astype(np.float32)


def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("At least one face embedding is required.")
    stacked = np.stack(
        [np.asarray(embedding, dtype=np.float32).reshape(-1) for embedding in embeddings],
        axis=0,
    )
    if stacked.ndim != 2 or stacked.shape[1] != 128:
        raise ValueError(f"Expected face embeddings with shape (N, 128), got {stacked.shape}.")
    centroid = stacked.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("The averaged face embedding is invalid.")
    return (centroid / norm).astype(np.float32)


def embedding_consistency(embeddings: list[np.ndarray]) -> dict[str, float | int | None]:
    if not embeddings:
        return {"sample_count": 0, "mean_distance_to_centroid": None, "max_distance_to_centroid": None}
    centroid = average_embeddings(embeddings)
    distances = [
        float(np.linalg.norm(np.asarray(embedding, dtype=np.float32).reshape(-1) - centroid))
        for embedding in embeddings
    ]
    return {
        "sample_count": len(embeddings),
        "mean_distance_to_centroid": round(float(np.mean(distances)), 6),
        "max_distance_to_centroid": round(float(np.max(distances)), 6),
    }


def get_face_identity_verifier(app) -> FaceIdentityVerifier:
    key = (
        str(Path(app.config["DRIVER_SAFETY_FACE_VERIFIER_PATH"]).resolve()),
        float(app.config.get("DRIVER_SAFETY_FACE_VERIFY_THRESHOLD", 0.94)),
    )
    verifier = _VERIFIER_CACHE.get(key)
    if verifier is None:
        verifier = FaceIdentityVerifier(app)
        _VERIFIER_CACHE[key] = verifier
    return verifier
