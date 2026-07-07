from __future__ import annotations

from .detectors import DetectorFactory


class YOLODetector:
    def __init__(self, model_path, confidence_threshold=0.35):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.detector = DetectorFactory.create(
            "yolo_pretrained",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            strict=True,
        )

    def detect_frame(self, frame):
        return self.detector.detect_frame(frame)
