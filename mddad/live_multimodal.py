from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

from .activity7_infer import Activity7Package
from .mmrt_v2_infer import MMRTV2Package, MMRT_V2_FACE_POINTS, _face_crop, _image_transform


POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


@dataclass(frozen=True)
class LiveMultimodalPrediction:
    label: str
    confidence: float
    probabilities: list[dict]
    geom2_label: str
    geom2_confidence: float
    baseline_label: str
    baseline_confidence: float
    specialist_overrides: dict
    class_index: int = -1


@dataclass(frozen=True)
class DriverStateDecision:
    state: str
    severity: str
    warning: str
    risk_multiplier: float


@dataclass(frozen=True)
class TemporalBufferEntry:
    timestamp_seconds: float
    label: str
    confidence: float
    driver_state: str
    severity: str
    warning: str
    audio_track_detected: bool
    pose_frame_count: int
    video_sample_count: int


class DriverStateModel:
    def predict(self, prediction: LiveMultimodalPrediction) -> DriverStateDecision:
        label = str(getattr(prediction, "label", "normal_driving"))
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        if label == "normal_driving":
            return DriverStateDecision("attentive", "low", "Driver remains attentive.", 0.0)
        if label in {"phone_calling", "phone_interaction", "phone_use"}:
            return DriverStateDecision("phone_distraction", "high", "Phone interaction detected.", 1.0 + confidence)
        if label == "eye_closure":
            return DriverStateDecision("drowsy_eye_closure", "high", "Prolonged eye closure detected.", 1.0 + confidence)
        if label == "yawning":
            return DriverStateDecision("drowsy_yawn", "medium", "Yawning detected.", 0.7 + confidence * 0.5)
        if label == "looking_away":
            return DriverStateDecision("posture_distraction", "medium", "Looking away detected.", 0.8 + confidence * 0.5)
        if label == "in_vehicle_display_interaction":
            return DriverStateDecision("in_vehicle_distraction", "medium", "Driver interacts with in-vehicle display.", 0.8 + confidence * 0.5)
        if label in {"eating_distraction", "drinking_distraction"}:
            return DriverStateDecision("manual_distraction", "medium", "Driver is eating or drinking.", 0.7 + confidence * 0.4)
        if label == "visual_posture_distraction":
            return DriverStateDecision("posture_distraction", "medium", "Driver posture suggests distraction.", 0.8 + confidence * 0.5)
        return DriverStateDecision("mixed_risk", "medium", "Mixed-risk behavior detected.", 0.9 + confidence * 0.5)


class TemporalBuffer:
    def __init__(self, maxlen=32):
        self._entries = deque(maxlen=max(1, int(maxlen)))

    def append(self, entry: TemporalBufferEntry):
        self._entries.append(entry)

    def __len__(self):
        return len(self._entries)

    def latest(self):
        return self._entries[-1] if self._entries else None

    def consecutive_label_streak(self, label: str):
        if not self._entries:
            return 0
        streak = 0
        for entry in reversed(self._entries):
            if entry.label != label:
                break
            streak += 1
        return streak

    def recent_states(self, limit=8):
        if not self._entries:
            return []
        return list(self._entries)[-max(1, int(limit)) :]


class EventRules:
    def evaluate(self, prediction: LiveMultimodalPrediction, driver_state: DriverStateDecision, temporal_buffer: TemporalBuffer):
        label = str(getattr(prediction, "label", "normal_driving"))
        confidence = float(getattr(prediction, "confidence", 0.0) or 0.0)
        streak = temporal_buffer.consecutive_label_streak(label)
        is_risky = label != "normal_driving"
        min_confidence = 0.45
        min_streak = 2

        if label == "normal_driving":
            return {
                "event_type": "normal_driving",
                "severity": "low",
                "warning_key": "realtime.warning_normal",
                "warning": "Driver state is normal.",
                "should_log": False,
                "risk_multiplier": 0.0,
                "streak": streak,
            }

        if label in {"eating_distraction", "drinking_distraction"}:
            min_confidence = 0.90
            min_streak = 4
        elif label in {"visual_posture_distraction", "in_vehicle_display_interaction"}:
            min_confidence = 0.58
            min_streak = 3
        elif label in {"phone_calling", "phone_interaction", "phone_use"}:
            min_confidence = 0.52
            min_streak = 2
        elif label in {"eye_closure", "yawning", "looking_away"}:
            min_confidence = 0.52
            min_streak = 2

        if confidence < min_confidence or streak < min_streak:
            return {
                "event_type": label,
                "severity": "low",
                "warning_key": "realtime.warning_candidate",
                "warning": "Behavior candidate is not stable enough to log as an event yet.",
                "should_log": False,
                "risk_multiplier": max(0.1, confidence * 0.4),
                "streak": streak,
            }

        if label in {"phone_calling", "phone_interaction", "phone_use"}:
            severity = "critical" if streak >= 3 or driver_state.severity == "high" else "high"
            warning_key = "realtime.warning_phone"
            warning = "Phone call or interaction detected."
        elif label == "eye_closure":
            severity = "high"
            warning_key = "realtime.warning_eye_closure"
            warning = "Eye closure detected."
        elif label == "yawning":
            severity = "medium"
            warning_key = "realtime.warning_yawning"
            warning = "Yawning detected."
        elif label == "looking_away":
            severity = "medium"
            warning_key = "realtime.warning_looking_away"
            warning = "Looking away detected."
        elif label == "in_vehicle_display_interaction":
            severity = "medium"
            warning_key = "realtime.warning_display"
            warning = "In-vehicle display distraction detected."
        elif label in {"eating_distraction", "drinking_distraction"}:
            severity = "medium"
            warning_key = "realtime.warning_food"
            warning = "Eating or drinking distraction detected."
        elif label == "visual_posture_distraction":
            severity = "medium"
            warning_key = "realtime.warning_posture"
            warning = "Posture distraction detected."
        else:
            severity = "medium"
            warning_key = "realtime.warning_mixed"
            warning = "Mixed-risk behavior detected."

        if streak >= 4 and severity != "critical":
            severity = "critical"
            warning = f"{warning} Temporal streak is sustained."

        risk_multiplier = max(driver_state.risk_multiplier, 0.5 + confidence)
        return {
            "event_type": label,
            "severity": severity,
            "warning_key": warning_key,
            "warning": warning,
            "should_log": is_risky,
            "risk_multiplier": risk_multiplier,
            "streak": streak,
        }


def _build_frame_embedding_model(device):
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)
        preprocess = weights.transforms()
    except Exception:
        backbone = models.mobilenet_v3_small(weights=None)

        def preprocess(image):
            tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            return F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)

    backbone.classifier = nn.Identity()
    backbone.eval().to(device)
    return backbone, preprocess


class LiveMultimodalWindow:
    def __init__(
        self,
        package_path,
        backend="mmrt_multimodal_full_v2_t4x2",
        device="cpu",
        decision_mode="final",
        clip_seconds=4.0,
        stride_seconds=2.0,
        frame_count=16,
        audio_sample_rate=16000,
    ):
        self.backend = str(backend or "mmrt_multimodal_full_v2_t4x2").strip().lower()
        self._uses_mmrt = self.backend.startswith("mmrt")
        if self._uses_mmrt:
            self.package = MMRTV2Package(package_path, device=device)
        else:
            self.package = Activity7Package(package_path, device=device, decision_mode=decision_mode)
        self.device = torch.device(device)
        self.clip_seconds = float(clip_seconds)
        self.stride_seconds = float(stride_seconds)
        self.frame_count = int(frame_count)
        self.audio_sample_rate = int(audio_sample_rate)
        self.sample_interval = max(self.clip_seconds / max(self.frame_count, 1), 0.05)
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is not installed.") from exc
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3,
        )
        if self._uses_mmrt:
            self._frame_model = None
            self._preprocess = _image_transform()
            self._face_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.3,
            )
        else:
            self._frame_model, self._preprocess = _build_frame_embedding_model(self.device)
            self._face_detector = None
        self.driver_state_model = DriverStateModel()
        self.event_rules = EventRules()
        self._samples = deque(maxlen=max(self.frame_count * 6, 64))
        self._temporal_buffer = TemporalBuffer(maxlen=max(self.frame_count * 4, 32))
        self._audio = np.zeros(0, dtype=np.float32)
        self._audio_source_rate = self.audio_sample_rate
        self._last_sample_at = None
        self._last_prediction_at = None
        self._last_prediction = None
        self._latest_pose_points = []

    def close(self):
        try:
            self._pose.close()
        except Exception:
            pass
        if self._face_detector is not None:
            try:
                self._face_detector.close()
            except Exception:
                pass
        try:
            self.package.close()
        except Exception:
            pass

    def push_audio_chunk(self, samples, sample_rate):
        if samples is None:
            return
        if isinstance(samples, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(samples, dtype=np.float32)
        else:
            arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        sample_rate = int(sample_rate or self.audio_sample_rate)
        if self._audio.size == 0:
            self._audio_source_rate = sample_rate
            self._audio = arr.astype(np.float32, copy=False)
        else:
            if sample_rate != self._audio_source_rate:
                self._audio = librosa.resample(self._audio.astype(np.float32), orig_sr=self._audio_source_rate, target_sr=sample_rate)
                self._audio_source_rate = sample_rate
            self._audio = np.concatenate([self._audio, arr.astype(np.float32, copy=False)])
        max_samples = int(max(self.clip_seconds * self._audio_source_rate * 2, self.audio_sample_rate * 8))
        if self._audio.size > max_samples:
            self._audio = self._audio[-max_samples:]

    def _pose_features(self, frame_rgb):
        result = self._pose.process(frame_rgb)
        if not result.pose_landmarks:
            return np.zeros(51, dtype=np.float32), np.zeros(50, dtype=np.float32), 0, []
        landmarks = result.pose_landmarks.landmark
        vec = []
        mmrt_vec = []
        points = []
        for idx in POSE_INDICES:
            if idx < len(landmarks):
                lm = landmarks[idx]
                vec.extend([lm.x, lm.y, lm.z])
                points.append(
                    {
                        "index": int(idx),
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(getattr(lm, "visibility", 0.0) or 0.0),
                    }
                )
            else:
                vec.extend([0.0, 0.0, 0.0])
        for idx in MMRT_V2_FACE_POINTS:
            if idx < len(landmarks):
                lm = landmarks[idx]
                mmrt_vec.extend([lm.x, lm.y])
            else:
                mmrt_vec.extend([0.0, 0.0])
        return np.asarray(vec[:51], dtype=np.float32), np.asarray(mmrt_vec[:50], dtype=np.float32), 1, points

    def _frame_embedding(self, frame_rgb):
        pil = Image.fromarray(frame_rgb)
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._frame_model(tensor).detach().cpu().numpy()[0]
        return feat.astype(np.float32)

    def _frame_tensor(self, frame_rgb):
        return self._preprocess(Image.fromarray(frame_rgb)).to(dtype=torch.float32)

    def _roi_embedding(self, frame):
        h, w = frame.shape[:2]
        x1 = max(0, int(w * 0.15))
        y1 = max(0, int(h * 0.05))
        x2 = min(w, int(w * 0.85))
        y2 = min(h, int(h * 0.95))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensor = self._preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._frame_model(tensor).detach().cpu().numpy()[0]
        return feat.astype(np.float32)

    def push_frame(self, frame, timestamp_seconds):
        if frame is None:
            return None
        timestamp_seconds = float(timestamp_seconds or 0.0)
        sampled = False
        if self._last_sample_at is None or (timestamp_seconds - self._last_sample_at) >= self.sample_interval:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_vector, mmrt_pose_vector, pose_present, pose_points = self._pose_features(rgb_frame)
            self._latest_pose_points = pose_points
            if self._uses_mmrt:
                face_crop, face_detected = _face_crop(frame, self._face_detector)
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                self._samples.append(
                    {
                        "timestamp": timestamp_seconds,
                        "body_frame": self._frame_tensor(rgb_frame),
                        "face_frame": self._frame_tensor(face_rgb),
                        "pose": mmrt_pose_vector,
                        "pose_present": bool(pose_present),
                        "face_present": bool(face_detected),
                    }
                )
            else:
                self._samples.append(
                    {
                        "timestamp": timestamp_seconds,
                        "video": self._frame_embedding(rgb_frame),
                        "pose": pose_vector,
                        "roi": self._roi_embedding(frame),
                        "pose_present": bool(pose_present),
                    }
                )
            self._last_sample_at = timestamp_seconds
            sampled = True
        return self._predict_if_ready(timestamp_seconds) if sampled else None

    def _window_samples(self, timestamp_seconds):
        start_seconds = max(0.0, timestamp_seconds - self.clip_seconds)
        samples = [item for item in self._samples if item["timestamp"] >= start_seconds]
        if len(samples) > self.frame_count:
            samples = samples[-self.frame_count :]
        return start_seconds, samples

    def _pad_sequence(self, items, dim):
        if not items:
            return np.zeros((self.frame_count, dim), dtype=np.float32)
        arr = np.asarray(items, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] > self.frame_count:
            arr = arr[-self.frame_count :]
        if arr.shape[0] < self.frame_count:
            pad = np.zeros((self.frame_count - arr.shape[0], dim), dtype=np.float32)
            arr = np.vstack([pad, arr])
        return arr.astype(np.float32)

    def _audio_window(self, timestamp_seconds):
        if self._audio.size == 0:
            return np.zeros(int(self.clip_seconds * self.audio_sample_rate), dtype=np.float32), False
        target_len = int(self.clip_seconds * self._audio_source_rate)
        audio = self._audio[-target_len:] if self._audio.size >= target_len else self._audio
        audio = audio.astype(np.float32, copy=False)
        if self._audio_source_rate != self.audio_sample_rate:
            audio = librosa.resample(audio, orig_sr=self._audio_source_rate, target_sr=self.audio_sample_rate)
        min_len = int(self.clip_seconds * self.audio_sample_rate)
        if audio.size < min_len:
            audio = np.pad(audio, (min_len - audio.size, 0))
        elif audio.size > min_len:
            audio = audio[-min_len:]
        return audio.astype(np.float32), True

    def _predict_if_ready(self, timestamp_seconds):
        if self._last_prediction_at is not None and (timestamp_seconds - self._last_prediction_at) < self.stride_seconds:
            return None
        start_seconds, samples = self._window_samples(timestamp_seconds)
        if not samples:
            return None
        audio_wave, audio_detected = self._audio_window(timestamp_seconds)
        if self._uses_mmrt:
            body_frames = torch.stack([item["body_frame"] for item in samples], dim=0)
            face_frames = torch.stack([item["face_frame"] for item in samples], dim=0)
            pose_seq = np.stack([item["pose"] for item in samples], axis=0).astype(np.float32)
            face_hits = sum(1 for item in samples if item.get("face_present"))
            pose_frame_count = int(np.count_nonzero(np.any(np.abs(pose_seq) > 0, axis=1)))
            mmrt_result = self.package.predict_from_modalities(
                body_frames=body_frames,
                face_frames=face_frames,
                skeleton_seq=pose_seq,
                audio_wave=audio_wave,
                audio_sample_rate=self.audio_sample_rate,
                audio_detected=audio_detected,
                face_hits=face_hits,
                skeleton_hits=pose_frame_count,
                start_seconds=start_seconds,
                end_seconds=timestamp_seconds,
                frame_count=len(samples),
            )
            class_index = next(
                (idx for idx, item in enumerate(mmrt_result.probabilities) if item["class_name"] == mmrt_result.label),
                0,
            )
            prediction = LiveMultimodalPrediction(
                label=mmrt_result.label,
                confidence=mmrt_result.confidence,
                probabilities=mmrt_result.probabilities,
                geom2_label=mmrt_result.label,
                geom2_confidence=mmrt_result.confidence,
                baseline_label=mmrt_result.label,
                baseline_confidence=mmrt_result.confidence,
                specialist_overrides={},
                class_index=class_index,
            )
        else:
            video_seq = np.stack([item["video"] for item in samples], axis=0).astype(np.float32)
            pose_seq = np.stack([item["pose"] for item in samples], axis=0).astype(np.float32)
            roi_seq = np.stack([item["roi"] for item in samples], axis=0).astype(np.float32)
            prediction = self.package.predict_from_feature_pack(
                video_seq=video_seq,
                pose_seq=pose_seq,
                audio_wave=audio_wave,
                roi_seq=roi_seq,
            )
            pose_frame_count = int(np.count_nonzero(np.any(np.abs(pose_seq) > 0, axis=1)))
        driver_state = self.driver_state_model.predict(LiveMultimodalPrediction(**{
            "label": prediction.label,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "geom2_label": prediction.geom2_label,
            "geom2_confidence": prediction.geom2_confidence,
            "baseline_label": prediction.baseline_label,
            "baseline_confidence": prediction.baseline_confidence,
            "specialist_overrides": prediction.specialist_overrides,
            "class_index": getattr(prediction, "class_index", -1),
        }))
        event_rule = self.event_rules.evaluate(prediction, driver_state, self._temporal_buffer)
        temporal_entry = TemporalBufferEntry(
            timestamp_seconds=float(timestamp_seconds),
            label=str(prediction.label),
            confidence=float(prediction.confidence),
            driver_state=driver_state.state,
            severity=event_rule["severity"],
            warning=event_rule["warning"],
            audio_track_detected=bool(audio_detected),
            pose_frame_count=pose_frame_count,
            video_sample_count=int(len(samples)),
        )
        self._temporal_buffer.append(temporal_entry)
        self._last_prediction_at = timestamp_seconds
        payload = {
            "start_seconds": round(start_seconds, 2),
            "end_seconds": round(timestamp_seconds, 2),
            "prediction": prediction,
            "action_label": prediction.label,
            "action_confidence": prediction.confidence,
            "driver_state": driver_state.state,
            "driver_state_severity": driver_state.severity,
            "driver_state_warning": driver_state.warning,
            "risk_multiplier": driver_state.risk_multiplier,
            "event_type": event_rule["event_type"],
            "event_severity": event_rule["severity"],
            "warning_key": event_rule["warning_key"],
            "warning": event_rule["warning"],
            "should_log_event": event_rule["should_log"],
            "temporal_streak": event_rule["streak"],
            "temporal_buffer_size": len(self._temporal_buffer),
            "audio_track_detected": audio_detected,
            "video_sample_count": int(len(samples)),
            "pose_frame_count": pose_frame_count,
            "skeleton_landmarks_detected": bool(pose_frame_count),
            "skeleton_landmarks_frame_count": pose_frame_count,
            "pose_points": list(self._latest_pose_points),
            "audio_sample_count": int(audio_wave.size),
        }
        self._last_prediction = payload
        return payload

    @property
    def last_prediction(self):
        return self._last_prediction
