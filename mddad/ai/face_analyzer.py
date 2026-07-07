from __future__ import annotations

from dataclasses import dataclass
from math import dist

import cv2


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH = (78, 13, 308, 14)
FACE_BOUNDS = (10, 152, 234, 454)
NOSE_TIP = 1


@dataclass
class FaceAnalyzerConfig:
    ear_threshold: float = 0.23
    mar_threshold: float = 0.60
    horizontal_pose_threshold: float = 0.07
    down_pose_threshold: float = 0.08
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class MediaPipeFaceAnalyzer:
    def __init__(self, config=None):
        self.config = config or FaceAnalyzerConfig()
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed.") from exc

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def close(self):
        self._face_mesh.close()

    def analyze_frame(self, frame, timestamp_seconds):
        try:
            if frame is None:
                return self._empty_result(timestamp_seconds, "invalid_frame", "Frame is missing.")

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._face_mesh.process(rgb_frame)
            if not result.multi_face_landmarks:
                return self._empty_result(timestamp_seconds, "no_face", None)

            landmarks = result.multi_face_landmarks[0].landmark
            points = [(landmark.x, landmark.y) for landmark in landmarks]
            ear = self._average_ear(points)
            mar = self._mar(points)
            head_pose_label = self._head_pose_label(points)

            return {
                "timestamp_seconds": float(timestamp_seconds),
                "face_detected": True,
                "ear": ear,
                "mar": mar,
                "eye_closed": ear is not None and ear < self.config.ear_threshold,
                "yawn_candidate": mar is not None and mar > self.config.mar_threshold,
                "head_pose_label": head_pose_label,
                "looking_away": head_pose_label in {"looking_left", "looking_right"},
                "head_down": head_pose_label == "looking_down",
                "quality_status": "ok",
                "error": None,
            }
        except Exception as exc:
            return self._empty_result(timestamp_seconds, "error", str(exc))

    def _empty_result(self, timestamp_seconds, quality_status, error):
        return {
            "timestamp_seconds": float(timestamp_seconds),
            "face_detected": False,
            "ear": None,
            "mar": None,
            "eye_closed": False,
            "yawn_candidate": False,
            "head_pose_label": "unknown",
            "looking_away": False,
            "head_down": False,
            "quality_status": quality_status,
            "error": error,
        }

    def _average_ear(self, points):
        left = self._ear(points, LEFT_EYE)
        right = self._ear(points, RIGHT_EYE)
        if left is None and right is None:
            return None
        values = [value for value in (left, right) if value is not None]
        return sum(values) / len(values)

    def _ear(self, points, indices):
        try:
            p1, p2, p3, p4, p5, p6 = [points[index] for index in indices]
            horizontal = dist(p1, p4)
            if horizontal <= 0:
                return None
            return (dist(p2, p6) + dist(p3, p5)) / (2.0 * horizontal)
        except (IndexError, ValueError):
            return None

    def _mar(self, points):
        try:
            left, top, right, bottom = [points[index] for index in MOUTH]
            horizontal = dist(left, right)
            if horizontal <= 0:
                return None
            return dist(top, bottom) / horizontal
        except (IndexError, ValueError):
            return None

    def _head_pose_label(self, points):
        try:
            top = points[FACE_BOUNDS[0]]
            bottom = points[FACE_BOUNDS[1]]
            left = points[FACE_BOUNDS[2]]
            right = points[FACE_BOUNDS[3]]
            nose = points[NOSE_TIP]
            center_x = (left[0] + right[0]) / 2.0
            center_y = (top[1] + bottom[1]) / 2.0
            face_width = max(abs(right[0] - left[0]), 0.001)
            face_height = max(abs(bottom[1] - top[1]), 0.001)
            horizontal_offset = (nose[0] - center_x) / face_width
            vertical_offset = (nose[1] - center_y) / face_height

            if vertical_offset > self.config.down_pose_threshold:
                return "looking_down"
            if horizontal_offset < -self.config.horizontal_pose_threshold:
                return "looking_left"
            if horizontal_offset > self.config.horizontal_pose_threshold:
                return "looking_right"
            return "forward"
        except (IndexError, ValueError):
            return "unknown"
