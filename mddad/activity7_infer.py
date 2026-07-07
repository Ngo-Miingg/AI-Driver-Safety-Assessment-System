from __future__ import annotations

import json
import math
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import joblib
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models


ACTIVITY7_DEFAULT_CLASSES = [
    "normal_driving",
    "phone_interaction",
    "phone_calling",
    "in_vehicle_display_interaction",
    "eating_distraction",
    "drinking_distraction",
    "visual_posture_distraction",
]

ACTIVITY7_SUPPRESSED_CLASSES = {"drinking_distraction"}
MMRT_5CLASS_DEFAULT_CLASSES = [
    "normal_driving",
    "eye_closure",
    "yawning",
    "looking_away",
    "phone_use",
]


@dataclass(frozen=True)
class Activity7Prediction:
    label: str
    class_index: int
    confidence: float
    probabilities: list[dict]
    geom2_label: str
    geom2_confidence: float
    baseline_label: str
    baseline_confidence: float
    specialist_overrides: dict


@dataclass(frozen=True)
class Activity7ClipInference:
    prediction: Activity7Prediction
    start_seconds: float
    end_seconds: float
    audio_track_detected: bool
    sampled_frame_count: int
    pose_frame_count: int


def _normalize_activity7_outputs(classes, probabilities, final_label, final_confidence, geom2_label, geom2_confidence, baseline_label, baseline_confidence, specialist_overrides):
    filtered = [item for item in probabilities if item["class_name"] not in ACTIVITY7_SUPPRESSED_CLASSES]
    if not filtered:
        filtered = [{"class_name": "normal_driving", "confidence": 1.0}]
    total_conf = sum(float(item["confidence"]) for item in filtered)
    if total_conf > 0:
        filtered = [
            {"class_name": item["class_name"], "confidence": float(item["confidence"]) / total_conf}
            for item in filtered
        ]
    filtered.sort(key=lambda item: item["confidence"], reverse=True)
    fallback_label = filtered[0]["class_name"]
    confidence_by_label = {item["class_name"]: float(item["confidence"]) for item in filtered}

    if final_label in ACTIVITY7_SUPPRESSED_CLASSES:
        final_label = fallback_label
        final_confidence = confidence_by_label.get(final_label, 0.0)
    if geom2_label in ACTIVITY7_SUPPRESSED_CLASSES:
        geom2_label = fallback_label
        geom2_confidence = confidence_by_label.get(geom2_label, 0.0)
    if baseline_label in ACTIVITY7_SUPPRESSED_CLASSES:
        baseline_label = fallback_label
        baseline_confidence = confidence_by_label.get(baseline_label, 0.0)

    final_index = classes.index(final_label) if final_label in classes else classes.index("normal_driving")
    specialist_overrides = {
        class_name: prob
        for class_name, prob in specialist_overrides.items()
        if class_name not in ACTIVITY7_SUPPRESSED_CLASSES
    }
    return (
        final_index,
        final_label,
        float(final_confidence),
        filtered,
        geom2_label,
        float(geom2_confidence),
        baseline_label,
        float(baseline_confidence),
        specialist_overrides,
    )


class VideoBranch(nn.Module):
    def __init__(self, input_dim=576, hidden_dim=256, num_layers=2):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim))

    def forward(self, x):
        x = self.norm(x)
        _, h = self.gru(x)
        feat = h[-1]
        return self.head(feat)


class PoseBranch(nn.Module):
    def __init__(self, input_dim=51, hidden_dim=128):
        super().__init__()
        self.frame_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        x = self.frame_mlp(x)
        _, h = self.gru(x)
        return h[-1]


class AudioBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.net(x)
        return self.fc(x)


class Activity7CoreModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.video_branch = VideoBranch()
        self.pose_branch = PoseBranch()
        self.audio_branch = AudioBranch()
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, video_seq, pose_seq, audio_wave):
        video_feat = self.video_branch(video_seq)
        pose_feat = self.pose_branch(pose_seq)
        audio_feat = self.audio_branch(audio_wave)
        fused = torch.cat([video_feat, pose_feat, audio_feat], dim=1)
        logits = self.classifier(fused)
        return logits, fused


class Activity7GeomResidual(nn.Module):
    def __init__(self, meta_dim=270, geom_dim=256, num_classes=7):
        super().__init__()
        self.raw_scale = nn.Parameter(torch.tensor(1.0))
        self.net = nn.Sequential(
            nn.LayerNorm(meta_dim),
            nn.Linear(meta_dim, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(96, num_classes),
        )
        self.geom_dim = geom_dim

    def forward(self, meta):
        meta = meta * self.raw_scale
        return self.net(meta)


def _ensure_1d_float32(array):
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _pad_or_trim_2d(array, length, dim):
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, dim)
    if arr.shape[1] != dim:
        raise ValueError(f"Expected feature dim {dim}, got {arr.shape[1]}.")
    if arr.shape[0] == length:
        return arr
    if arr.shape[0] > length:
        return arr[:length]
    pad = np.zeros((length - arr.shape[0], dim), dtype=np.float32)
    return np.vstack([arr, pad])


def _chunk_stats(feature_seq, chunks=4):
    seq = np.asarray(feature_seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    splits = np.array_split(seq, chunks, axis=0)
    stats = []
    for split in splits:
        if split.size == 0:
            split = np.zeros((1, seq.shape[1]), dtype=np.float32)
        stats.extend([split.mean(axis=0), split.std(axis=0), split.max(axis=0)])
    return np.concatenate(stats, axis=0).astype(np.float32)


def _build_pose_geom_z(pose_seq):
    seq = np.asarray(pose_seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    if seq.shape[1] != 51:
        raise ValueError(f"Expected pose sequence with 51 dims, got {seq.shape[1]}.")
    mean = seq.mean(axis=0)
    std = seq.std(axis=0)
    min_ = seq.min(axis=0)
    max_ = seq.max(axis=0)
    last = seq[-1]
    motion = np.mean(np.abs(np.diff(seq, axis=0))) if seq.shape[0] > 1 else 0.0
    vec = np.concatenate([mean, std, min_, max_, last], axis=0)
    if vec.shape[0] < 255:
        vec = np.pad(vec, (0, 255 - vec.shape[0]))
    vec = np.concatenate([vec[:255], np.asarray([motion], dtype=np.float32)], axis=0)
    return vec.astype(np.float32)


def _load_npy(path):
    return np.load(path)


def _softmax_numpy(logits):
    x = np.asarray(logits, dtype=np.float32)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def _frame_embedding_model(device):
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        backbone = models.mobilenet_v3_small(weights=weights)
        preprocess = weights.transforms()
    except Exception:
        backbone = models.mobilenet_v3_small(weights=None)
        preprocess = lambda image: F.interpolate(  # noqa: E731
            torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    backbone.classifier = nn.Identity()
    backbone.eval().to(device)
    return backbone, preprocess


def _video_metadata(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = float(total_frames / fps) if fps > 0 else 0.0
    capture.release()
    return {"total_frames": total_frames, "fps": fps, "duration": duration}


def _window_frame_indices(total_frames, fps, sample_frames, start_seconds=0.0, duration_seconds=None):
    if total_frames <= 0:
        raise RuntimeError("Video has no readable frames.")
    if fps <= 0:
        fps = 30.0
    start_frame = max(0, int(round(start_seconds * fps)))
    if duration_seconds is None:
        end_frame = max(total_frames - 1, start_frame)
    else:
        end_frame = min(total_frames - 1, int(round((start_seconds + duration_seconds) * fps)) - 1)
    if end_frame < start_frame:
        end_frame = start_frame
    indices = np.linspace(start_frame, max(end_frame, start_frame), num=sample_frames, dtype=int)
    return np.unique(np.clip(indices, 0, max(total_frames - 1, 0)))


def _extract_video_embeddings(video_path, sample_frames=16, device="cpu", start_seconds=0.0, duration_seconds=None):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = _window_frame_indices(total_frames, fps, sample_frames, start_seconds=start_seconds, duration_seconds=duration_seconds)
    backbone, preprocess = _frame_embedding_model(device)
    embeddings = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(pil).detach().cpu().numpy()[0]
        embeddings.append(feat.astype(np.float32))
    capture.release()
    if not embeddings:
        return np.zeros((max(int(sample_frames), 1), 576), dtype=np.float32)
    return np.stack(embeddings, axis=0)


def _extract_audio_wave(video_path, target_sr=16000, duration_hint=None, start_seconds=0.0, clip_seconds=None):
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "audio.wav"
        cmd = ["ffmpeg", "-y"]
        if start_seconds > 0:
            cmd.extend(["-ss", f"{float(start_seconds):.3f}"])
        cmd.extend(["-i", str(video_path)])
        if clip_seconds is not None:
            cmd.extend(["-t", f"{float(clip_seconds):.3f}"])
        cmd.extend([
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-f",
            "wav",
            str(wav_path),
        ])
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            wave, sr = librosa.load(wav_path, sr=target_sr, mono=True)
            detected = True
        except Exception:
            duration = clip_seconds or duration_hint or 1.0
            wave = np.zeros(max(int(duration * target_sr), target_sr), dtype=np.float32)
            detected = False
            sr = target_sr
    if wave.size == 0:
        wave = np.zeros(target_sr, dtype=np.float32)
    if duration_hint:
        target_len = max(int(duration_hint * target_sr), target_sr)
        if wave.size > target_len:
            wave = wave[:target_len]
        elif wave.size < target_len:
            wave = np.pad(wave, (0, target_len - wave.size))
    return wave.astype(np.float32), detected


def _extract_pose_sequence(video_path, sample_frames=16, start_seconds=0.0, duration_seconds=None):
    import mediapipe as mp

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = _window_frame_indices(total_frames, fps, sample_frames, start_seconds=start_seconds, duration_seconds=duration_seconds)
    pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.3,
    )
    seq = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if not result.pose_landmarks:
            seq.append(np.zeros(51, dtype=np.float32))
            continue
        landmarks = result.pose_landmarks.landmark
        picks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        vec = []
        for idx in picks:
            if idx < len(landmarks):
                lm = landmarks[idx]
                vec.extend([lm.x, lm.y, lm.z])
            else:
                vec.extend([0.0, 0.0, 0.0])
        seq.append(np.asarray(vec[:51], dtype=np.float32))
    pose.close()
    capture.release()
    if not seq:
        return np.zeros((max(int(sample_frames), 1), 51), dtype=np.float32)
    return np.stack(seq, axis=0)


def _extract_roi_sequence(video_path, sample_frames=16, start_seconds=0.0, duration_seconds=None, device="cpu"):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    indices = _window_frame_indices(total_frames, fps, sample_frames, start_seconds=start_seconds, duration_seconds=duration_seconds)
    backbone, preprocess = _frame_embedding_model(device)
    embeddings = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        x1 = max(0, int(w * 0.15))
        y1 = max(0, int(h * 0.05))
        x2 = min(w, int(w * 0.85))
        y2 = min(h, int(h * 0.95))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            crop = frame
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(tensor).detach().cpu().numpy()[0]
        embeddings.append(feat.astype(np.float32))
    capture.release()
    if not embeddings:
        return np.zeros((max(int(sample_frames), 1), 576), dtype=np.float32)
    return np.stack(embeddings, axis=0)


class Activity7Package:
    def __init__(self, package_path, device="cpu", decision_mode="final"):
        self.package_path = Path(package_path)
        self.device = torch.device(device)
        self.decision_mode = str(decision_mode or "final").strip().lower()
        self._tempdir = None
        self.root = self._resolve_root()
        self.package_mode = self._detect_package_mode()
        self.core = self._load_core()
        self.geom2 = self._load_geom2()
        self.specialist = self._load_specialist()

    def close(self):
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def _resolve_root(self):
        if self.package_path.is_dir():
            return self.package_path
        if self.package_path.suffix.lower() == ".zip":
            self._tempdir = tempfile.TemporaryDirectory(prefix="activity7_pkg_")
            with zipfile.ZipFile(self.package_path) as zf:
                zf.extractall(self._tempdir.name)
            return Path(self._tempdir.name)
        raise FileNotFoundError(f"Unsupported package path: {self.package_path}")

    def _detect_package_mode(self):
        if (self.root / "outputs" / "checkpoints" / "activity7_fusion_feature_audio_v2_t4x2_best.pt").exists():
            return "activity7_package"
        if (self.root / "mmrt_multimodal_full_v2_t4x2" / "best_full_v2_t4x2.pt").exists():
            return "mmrt_bundle"
        if (self.root / "best_full_v2_t4x2.pt").exists():
            return "mmrt_checkpoint_dir"
        raise FileNotFoundError(f"Could not detect supported package contents under {self.root}")

    def _mmrt_ckpt_path(self):
        candidates = [
            self.root / "mmrt_multimodal_full_v2_t4x2" / "best_full_v2_t4x2.pt",
            self.root / "mmrt_multimodal_full_v1" / "best_full.pt",
            self.root / "mmrt_multimodal_smoke" / "best_smoke.pt",
            self.root / "best_full_v2_t4x2.pt",
            self.root / "best_full.pt",
            self.root / "best_smoke.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No supported MMRT checkpoint found under {self.root}")

    def _load_core(self):
        if self.package_mode == "activity7_package":
            ckpt = self.root / "outputs" / "checkpoints" / "activity7_fusion_feature_audio_v2_t4x2_best.pt"
        else:
            ckpt = self._mmrt_ckpt_path()
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        classes = obj.get("classes") or obj.get("class_order") or MMRT_5CLASS_DEFAULT_CLASSES
        model = Activity7CoreModel(num_classes=len(classes))
        try:
            model.load_state_dict(obj["model_state_dict"], strict=True)
        except RuntimeError as exc:
            if self.package_mode in {"mmrt_bundle", "mmrt_checkpoint_dir"}:
                raise RuntimeError(
                    "The MMRT raw checkpoint architecture does not match the current Activity7Package loader. "
                    "Use the integrated Activity7 package for app demos, or add the original MMRT model definition before wiring this checkpoint."
                ) from exc
            raise
        model.eval().to(self.device)
        return {"model": model, "classes": classes, "config": obj.get("config", {})}

    def _load_geom2(self):
        if self.package_mode != "activity7_package":
            return None
        ckpt = self.root / "outputs" / "checkpoints" / "activity7_geom2_residual_best.pt"
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = Activity7GeomResidual(meta_dim=obj["config"]["meta_dim"], geom_dim=obj["config"]["geom_dim"], num_classes=len(obj["classes"]))
        model.load_state_dict(obj["model_state_dict"], strict=True)
        model.eval().to(self.device)
        return {"model": model, "classes": obj["classes"], "config": obj["config"]}

    def _load_specialist(self):
        if self.package_mode != "activity7_package":
            return None
        path = self.root / "outputs" / "final_activity7" / "activity7_specialist1_final_decision_layer.joblib"
        return joblib.load(path)

    def predict_from_feature_pack(self, video_seq, pose_seq, audio_wave, pose_geom_vec=None, roi_seq=None):
        core_model = self.core["model"]
        classes = self.core["classes"]
        with torch.no_grad():
            video = torch.as_tensor(video_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            pose = torch.as_tensor(pose_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
            audio = torch.as_tensor(audio_wave, dtype=torch.float32, device=self.device).unsqueeze(0)
            if audio.dim() == 2:
                audio = audio.unsqueeze(1)
            logits = core_model(video, pose, audio)[0].detach().cpu().numpy()[0]
            probs = _softmax_numpy(logits)
        probabilities = [
            {"class_name": class_name, "confidence": float(prob)}
            for class_name, prob in zip(classes, probs)
        ]
        probabilities.sort(key=lambda item: item["confidence"], reverse=True)
        baseline_label = classes[int(np.argmax(probs))]
        baseline_confidence = float(np.max(probs))

        if self.geom2 is None:
            geom2_label = baseline_label
            geom2_confidence = baseline_confidence
            final_label = baseline_label
            final_confidence = baseline_confidence
            specialist_overrides = {}
            (
                final_index,
                final_label,
                final_confidence,
                probabilities,
                geom2_label,
                geom2_confidence,
                baseline_label,
                baseline_confidence,
                specialist_overrides,
            ) = _normalize_activity7_outputs(
                classes,
                probabilities,
                final_label,
                final_confidence,
                geom2_label,
                geom2_confidence,
                baseline_label,
                baseline_confidence,
                specialist_overrides,
            )
            return Activity7Prediction(
                label=final_label,
                class_index=final_index,
                confidence=float(final_confidence),
                probabilities=probabilities,
                geom2_label=geom2_label,
                geom2_confidence=geom2_confidence,
                baseline_label=baseline_label,
                baseline_confidence=baseline_confidence,
                specialist_overrides=specialist_overrides,
            )
        if pose_geom_vec is None:
            pose_geom_vec = _build_pose_geom_z(pose_seq)
        geom2_input = np.concatenate([logits, probs, pose_geom_vec[:256].astype(np.float32)], axis=0)
        geom2_model = self.geom2["model"]
        with torch.no_grad():
            geom2_logits = geom2_model(torch.as_tensor(geom2_input, dtype=torch.float32, device=self.device).unsqueeze(0)).detach().cpu().numpy()[0]
            geom2_probs = _softmax_numpy(geom2_logits)
        geom2_label_index = int(np.argmax(geom2_probs))
        geom2_label = classes[geom2_label_index]
        geom2_confidence = float(geom2_probs[geom2_label_index])
        final_label = geom2_label
        final_confidence = geom2_confidence
        specialist_overrides = {}

        if self.specialist is not None:
            if roi_seq is None:
                roi_seq = np.zeros((max(len(video_seq), 1), 576), dtype=np.float32)
            roi_stats = _chunk_stats(roi_seq, chunks=4)
            x_specialist = np.concatenate([geom2_logits, geom2_probs, pose_geom_vec[:256].astype(np.float32), roi_stats], axis=0)
            x_specialist = self.specialist["scaler"].transform([x_specialist])[0]
            for class_name, clf in self.specialist["specialists"].items():
                prob = float(clf.predict_proba([x_specialist])[0][1])
                specialist_overrides[class_name] = prob
                threshold = float(self.specialist["thresholds"][class_name])
                if prob >= threshold:
                    final_label = class_name
                    final_confidence = prob

        if self.decision_mode == "baseline":
            final_label = baseline_label
            final_confidence = baseline_confidence
        elif self.decision_mode == "geom2":
            final_label = geom2_label
            final_confidence = geom2_confidence
        (
            final_index,
            final_label,
            final_confidence,
            probabilities,
            geom2_label,
            geom2_confidence,
            baseline_label,
            baseline_confidence,
            specialist_overrides,
        ) = _normalize_activity7_outputs(
            classes,
            probabilities,
            final_label,
            final_confidence,
            geom2_label,
            geom2_confidence,
            baseline_label,
            baseline_confidence,
            specialist_overrides,
        )
        return Activity7Prediction(
            label=final_label,
            class_index=final_index,
            confidence=float(final_confidence),
            probabilities=probabilities,
            geom2_label=geom2_label,
            geom2_confidence=geom2_confidence,
            baseline_label=baseline_label,
            baseline_confidence=baseline_confidence,
            specialist_overrides=specialist_overrides,
        )

    def predict_clip(self, video_path, sample_frames=16, audio_sr=16000, start_seconds=0.0, duration_seconds=None):
        meta = _video_metadata(video_path)
        if duration_seconds is None:
            duration_seconds = meta["duration"] or max(sample_frames / 4.0, 1.0)
        video_seq = _extract_video_embeddings(
            video_path,
            sample_frames=sample_frames,
            device=self.device,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        pose_seq = _extract_pose_sequence(
            video_path,
            sample_frames=sample_frames,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        roi_seq = _extract_roi_sequence(
            video_path,
            sample_frames=sample_frames,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            device=self.device,
        )
        audio_wave, audio_detected = _extract_audio_wave(
            video_path,
            target_sr=audio_sr,
            duration_hint=duration_seconds or None,
            start_seconds=start_seconds,
            clip_seconds=duration_seconds,
        )
        prediction = self.predict_from_feature_pack(video_seq, pose_seq, audio_wave, roi_seq=roi_seq)
        pose_frame_count = int(np.count_nonzero(np.any(np.abs(pose_seq) > 0, axis=1)))
        return Activity7ClipInference(
            prediction=prediction,
            start_seconds=float(start_seconds),
            end_seconds=float(start_seconds + duration_seconds),
            audio_track_detected=bool(audio_detected),
            sampled_frame_count=int(video_seq.shape[0]),
            pose_frame_count=pose_frame_count,
        )

    def predict_from_video(self, video_path, sample_frames=16, audio_sr=16000):
        return self.predict_clip(
            video_path,
            sample_frames=sample_frames,
            audio_sr=audio_sr,
        ).prediction

    def predict_from_metadata_row(self, row, feature_root):
        feature_root = Path(feature_root)
        video_seq = _load_npy(feature_root / row["video_feature_path"])
        pose_seq = _load_npy(feature_root / row["pose_path"])
        audio_path = feature_root / row["audio_v2_path"]
        if audio_path.suffix.lower() == ".wav":
            audio_wave, _ = librosa.load(audio_path, sr=16000, mono=True)
        else:
            audio_wave = _extract_audio_wave(audio_path, target_sr=16000)
        roi_seq = _load_npy(feature_root / row["roi_fast_path"])
        pose_geom_vec = _load_npy(feature_root / row["pose_geom_path"]).reshape(-1).astype(np.float32)
        return self.predict_from_feature_pack(video_seq, pose_seq, audio_wave, pose_geom_vec=pose_geom_vec, roi_seq=roi_seq)


def prediction_to_dict(prediction: Activity7Prediction):
    return {
        "label": prediction.label,
        "class_index": prediction.class_index,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "geom2_label": prediction.geom2_label,
        "geom2_confidence": prediction.geom2_confidence,
        "baseline_label": prediction.baseline_label,
        "baseline_confidence": prediction.baseline_confidence,
        "specialist_overrides": prediction.specialist_overrides,
    }
