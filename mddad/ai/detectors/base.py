from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: list[float]
    is_phone: bool = False

    def as_dict(self):
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "is_phone": self.is_phone,
        }


class BaseDetector:
    backend_name = "base"

    def __init__(self, confidence_threshold=0.35):
        self.confidence_threshold = confidence_threshold
        self.error = None

    @property
    def is_available(self):
        return self.error is None

    def detect_frame(self, frame):
        raise NotImplementedError
