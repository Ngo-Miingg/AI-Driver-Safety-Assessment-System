from __future__ import annotations

from .base import BaseDetector, Detection


PHONE_CLASS_NAMES = {"cell phone", "mobile phone", "mobile-phone", "phone", "cellphone", "mobilephone"}


class UltralyticsPhoneDetector(BaseDetector):
    def __init__(self, backend, model_path, confidence_threshold=0.35):
        super().__init__(confidence_threshold=confidence_threshold)
        self.backend_name = backend
        self.model_path = model_path

        try:
            if str(backend).startswith("rtdetr"):
                from ultralytics import RTDETR

                self.model = RTDETR(str(model_path))
            else:
                from ultralytics import YOLO

                self.model = YOLO(str(model_path))
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run pip install -r requirements.txt.") from exc
        except Exception as exc:
            raise RuntimeError(f"Could not load {backend} detector model: {exc}") from exc

    def detect_frame(self, frame):
        try:
            results = self.model.predict(
                frame,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"{self.backend_name} inference failed: {exc}") from exc

        detections = []
        for result in results:
            names = result.names or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = str(names.get(class_id, class_id))
                xyxy = box.xyxy[0].tolist()
                detection = Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[round(float(value), 2) for value in xyxy],
                    is_phone=class_name.lower() in PHONE_CLASS_NAMES,
                )
                detections.append(detection.as_dict())
        return detections
