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
        frame_height, frame_width = frame.shape[:2]
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
                x1, y1, x2, y2 = [float(value) for value in xyxy]
                box_width = max(x2 - x1, 0.0)
                box_height = max(y2 - y1, 0.0)
                frame_area = max(float(frame_width * frame_height), 1.0)
                box_area_ratio = (box_width * box_height) / frame_area
                width_ratio = box_width / max(float(frame_width), 1.0)
                height_ratio = box_height / max(float(frame_height), 1.0)
                aspect_ratio = box_width / max(box_height, 1.0)

                # Reject obvious false positives such as boxes covering almost
                # the entire frame or extremely stretched regions.
                if (
                    box_area_ratio >= 0.45
                    or width_ratio >= 0.82
                    or height_ratio >= 0.88
                    or aspect_ratio >= 4.0
                    or aspect_ratio <= 0.2
                ):
                    continue

                detection = Detection(
                    class_name=class_name,
                    confidence=confidence,
                    bbox=[round(float(value), 2) for value in xyxy],
                    is_phone=class_name.lower() in PHONE_CLASS_NAMES,
                )
                detections.append(detection.as_dict())
        return detections
