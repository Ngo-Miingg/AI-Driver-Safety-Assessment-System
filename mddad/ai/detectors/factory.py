from __future__ import annotations

from pathlib import Path

from .base import BaseDetector


class UnavailableDetector(BaseDetector):
    backend_name = "unavailable"

    def __init__(self, backend_name, error, confidence_threshold=0.35):
        super().__init__(confidence_threshold=confidence_threshold)
        self.backend_name = backend_name
        self.error = error

    def detect_frame(self, frame):
        return []


class MediaPipeOnlyDetector(BaseDetector):
    backend_name = "mediapipe_only"

    def detect_frame(self, frame):
        return []


class DetectorFactory:
    SUPPORTED_BACKENDS = {
        "mediapipe_only",
        "yolo_pretrained",
        "yolo_finetuned",
        "rtdetr_finetuned",
    }

    @classmethod
    def model_path_for_backend(cls, app, backend, realtime=False):
        backend = (backend or "mediapipe_only").strip().lower()
        if backend == "yolo_finetuned":
            return app.config["YOLO_FINETUNED_MODEL_PATH"]
        if backend == "rtdetr_finetuned":
            return app.config["RTDETR_MODEL_PATH"]
        if backend == "yolo_pretrained":
            if realtime:
                return app.config.get("REALTIME_YOLO_MODEL_PATH") or app.config["YOLO_PRETRAINED_MODEL_PATH"]
            return app.config["YOLO_PRETRAINED_MODEL_PATH"]
        return None

    @classmethod
    def create(cls, backend, model_path=None, confidence_threshold=0.35, strict=False):
        backend = (backend or "mediapipe_only").strip().lower()
        if backend not in cls.SUPPORTED_BACKENDS:
            error = f"Unsupported detector backend: {backend}."
            if strict:
                raise RuntimeError(error)
            return UnavailableDetector(backend, error, confidence_threshold)

        if backend == "mediapipe_only":
            return MediaPipeOnlyDetector(confidence_threshold=confidence_threshold)

        if not model_path:
            error = f"Detector backend {backend} requires a model path."
            if strict:
                raise FileNotFoundError(error)
            return UnavailableDetector(backend, error, confidence_threshold)

        path = Path(model_path)
        if not path.exists():
            error = f"Detector model file not found at {path}."
            if strict:
                raise FileNotFoundError(error)
            return UnavailableDetector(backend, error, confidence_threshold)

        try:
            from .ultralytics_detector import UltralyticsPhoneDetector

            return UltralyticsPhoneDetector(
                backend=backend,
                model_path=path,
                confidence_threshold=confidence_threshold,
            )
        except Exception as exc:
            if strict:
                raise
            return UnavailableDetector(backend, str(exc), confidence_threshold)
