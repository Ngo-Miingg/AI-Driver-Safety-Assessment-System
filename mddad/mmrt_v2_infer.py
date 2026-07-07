from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import librosa
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


logger = logging.getLogger(__name__)

MMRT_V2_CLASSES = [
    "normal_driving",
    "eye_closure",
    "yawning",
    "looking_away",
    "phone_use",
]

MMRT_V2_FACE_POINTS = list(range(25))


@dataclass(frozen=True)
class MMRTV2ClipResult:
    start_seconds: float
    end_seconds: float
    label: str
    confidence: float
    probabilities: list[dict]
    face_frames_detected: int
    skeleton_frames_detected: int
    audio_track_detected: bool
    audio_rms: float
    frame_count: int


class MobileNetFrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_small(weights=None)
        self.features = backbone.features
        self.proj = nn.Linear(576, 128)

    def forward(self, frames):
        batch, steps, channels, height, width = frames.shape
        x = frames.reshape(batch * steps, channels, height, width)
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        x = self.proj(x)
        return x.reshape(batch, steps, 128)


class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_proj = nn.Sequential(
            nn.Linear(40, 96),
            nn.ReLU(inplace=True),
            nn.Linear(96, 128),
        )
        self.scalar_proj = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
        self.temporal = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, logmel, scalars, output_steps=16):
        x = self.frame_proj(logmel)
        scalar = self.scalar_proj(scalars).unsqueeze(1)
        x = x + scalar
        x, _ = self.temporal(x)
        if x.shape[1] != output_steps:
            x = F.interpolate(x.transpose(1, 2), size=output_steps, mode="linear", align_corners=False).transpose(1, 2)
        return x


class SkeletonEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(50),
            nn.Linear(50, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
        self.temporal = nn.GRU(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, seq):
        x = self.input_proj(seq)
        x, _ = self.temporal(x)
        return x


class MMRTV2Model(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.body_encoder = MobileNetFrameEncoder()
        self.face_encoder = MobileNetFrameEncoder()
        self.audio_encoder = AudioEncoder()
        self.skeleton_encoder = SkeletonEncoder()
        self.body_gate = nn.Linear(128, 128)
        self.face_gate = nn.Linear(128, 128)
        self.audio_gate = nn.Linear(128, 128)
        self.skeleton_gate = nn.Linear(128, 128)
        self.fusion_norm = nn.LayerNorm(128)
        self.temporal = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = nn.Linear(256, 1)
        self.evidence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, body_frames, face_frames, audio_logmel, audio_scalars, skeleton_seq):
        body_seq = self.body_encoder(body_frames)
        face_seq = self.face_encoder(face_frames)
        audio_seq = self.audio_encoder(audio_logmel, audio_scalars, output_steps=body_seq.shape[1])
        skeleton_seq = self.skeleton_encoder(skeleton_seq)

        body_gate = torch.sigmoid(self.body_gate(body_seq))
        face_gate = torch.sigmoid(self.face_gate(face_seq))
        audio_gate = torch.sigmoid(self.audio_gate(audio_seq))
        skeleton_gate = torch.sigmoid(self.skeleton_gate(skeleton_seq))

        numerator = (
            body_gate * body_seq
            + face_gate * face_seq
            + audio_gate * audio_seq
            + skeleton_gate * skeleton_seq
        )
        denominator = body_gate + face_gate + audio_gate + skeleton_gate + 1e-6
        fused_seq = self.fusion_norm(numerator / denominator)

        temporal_seq, _ = self.temporal(fused_seq)
        attn_logits = self.attn(temporal_seq).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(temporal_seq * attn, dim=1)
        evidence = self.evidence_head(pooled)
        logits = self.classifier(evidence)
        return logits


def _softmax(logits):
    x = np.asarray(logits, dtype=np.float32)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.maximum(exp.sum(), 1e-8)


def _video_meta(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration = float(total_frames / fps) if fps > 0 else 0.0
    capture.release()
    return {"total_frames": total_frames, "fps": fps if fps > 0 else 30.0, "duration": duration}


def _window_times(duration, clip_seconds, stride_seconds):
    if duration <= 0:
        return [(0.0, clip_seconds)]
    windows = []
    start = 0.0
    while start < duration:
        end = min(start + clip_seconds, duration)
        windows.append((start, end))
        if end >= duration:
            break
        start += stride_seconds
    return windows or [(0.0, clip_seconds)]


def _sample_indices(total_frames, fps, frame_count, start_seconds, end_seconds):
    start_frame = max(0, int(round(start_seconds * fps)))
    end_frame = max(start_frame, min(total_frames - 1, int(round(end_seconds * fps)) - 1))
    if total_frames <= 0:
        return np.zeros(frame_count, dtype=int)
    return np.linspace(start_frame, end_frame, num=frame_count, dtype=int)


def _image_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _face_crop(frame_bgr, detector):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    h, w = frame_bgr.shape[:2]
    if result.detections:
        bbox = result.detections[0].location_data.relative_bounding_box
        x1 = max(0, int((bbox.xmin - 0.08) * w))
        y1 = max(0, int((bbox.ymin - 0.08) * h))
        x2 = min(w, int((bbox.xmin + bbox.width + 0.08) * w))
        y2 = min(h, int((bbox.ymin + bbox.height + 0.08) * h))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size:
            return crop, True
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    y1 = 0
    y2 = int(h * 0.5)
    crop = frame_bgr[y1:y2, x1:x2]
    return crop if crop.size else frame_bgr, False


def _extract_body_face_frames(video_path: Path, frame_count: int, start_seconds: float, end_seconds: float):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    indices = _sample_indices(total_frames, fps, frame_count, start_seconds, end_seconds)
    tf = _image_transform()
    detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
    body_frames = []
    face_frames = []
    face_hits = 0
    try:
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = capture.read()
            if not ok or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            body_frames.append(tf(Image.fromarray(frame_rgb)))
            face_crop, detected = _face_crop(frame, detector)
            face_hits += int(detected)
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_frames.append(tf(Image.fromarray(face_rgb)))
    finally:
        detector.close()
        capture.release()
    return torch.stack(body_frames, dim=0), torch.stack(face_frames, dim=0), face_hits


def _extract_audio_features(video_path: Path, start_seconds: float, end_seconds: float):
    with tempfile.TemporaryDirectory(prefix="mmrt_v2_audio_") as tmpdir:
        wav_path = Path(tmpdir) / "clip.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{max(end_seconds - start_seconds, 0.2):.3f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(wav_path),
        ]
        audio_detected = True
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            wave, sr = librosa.load(wav_path, sr=16000, mono=True)
        except Exception:
            sr = 16000
            wave = np.zeros(sr * 3, dtype=np.float32)
            audio_detected = False

    return _audio_features_from_wave(wave, sr, audio_detected=audio_detected)


def _audio_features_from_wave(wave, sr, audio_detected=True):
    wave = np.asarray(wave if wave is not None else [], dtype=np.float32).reshape(-1)
    sr = int(sr or 16000)
    if wave.size == 0:
        wave = np.zeros(sr, dtype=np.float32)
        audio_detected = False

    mel = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=400, hop_length=160, n_mels=40, power=2.0)
    logmel = librosa.power_to_db(mel + 1e-6, ref=np.max).T.astype(np.float32)
    if logmel.shape[0] != 64:
        x = torch.from_numpy(logmel.T).unsqueeze(0)
        x = F.interpolate(x, size=64, mode="linear", align_corners=False)
        logmel = x.squeeze(0).T.numpy()
    if logmel.shape[1] != 40:
        if logmel.shape[1] == 0:
            logmel = np.zeros((64, 40), dtype=np.float32)
        else:
            x = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0)
            x = F.interpolate(x, size=(64, 40), mode="bilinear", align_corners=False)
            logmel = x.squeeze(0).squeeze(0).numpy()

    rms = librosa.feature.rms(y=wave, frame_length=400, hop_length=160).reshape(-1)
    zcr = librosa.feature.zero_crossing_rate(y=wave, frame_length=400, hop_length=160).reshape(-1)
    centroid = librosa.feature.spectral_centroid(y=wave, sr=sr, n_fft=400, hop_length=160).reshape(-1)
    bandwidth = librosa.feature.spectral_bandwidth(y=wave, sr=sr, n_fft=400, hop_length=160).reshape(-1)
    scalars = np.asarray(
        [
            float(rms.mean()) if rms.size else 0.0,
            float(zcr.mean()) if zcr.size else 0.0,
            float(centroid.mean() / max(sr, 1)) if centroid.size else 0.0,
            float(bandwidth.mean() / max(sr, 1)) if bandwidth.size else 0.0,
        ],
        dtype=np.float32,
    )
    audio_rms = float(np.sqrt(np.mean(np.square(wave))) if wave.size else 0.0)
    return logmel[:64, :40], scalars, bool(audio_detected), audio_rms


def _extract_skeleton(video_path: Path, frame_count: int, start_seconds: float, end_seconds: float):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    indices = _sample_indices(total_frames, fps, frame_count, start_seconds, end_seconds)
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.3)
    rows = []
    hits = 0
    try:
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = capture.read()
            if not ok or frame is None:
                rows.append(np.zeros(50, dtype=np.float32))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if not result.pose_landmarks:
                rows.append(np.zeros(50, dtype=np.float32))
                continue
            hits += 1
            pts = []
            for landmark_idx in MMRT_V2_FACE_POINTS:
                lm = result.pose_landmarks.landmark[landmark_idx]
                pts.extend([float(lm.x), float(lm.y)])
            rows.append(np.asarray(pts[:50], dtype=np.float32))
    finally:
        pose.close()
        capture.release()
    return np.stack(rows, axis=0), hits


class MMRTV2Package:
    def __init__(self, package_path: str | Path, device: str = "cpu"):
        self.package_path = Path(package_path)
        self.device = torch.device(device)
        self._tempdir = None
        self.root = self._resolve_root()
        self.ckpt_path = self._find_checkpoint()
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.class_order = list(ckpt.get("class_order") or MMRT_V2_CLASSES)
        self.model = MMRTV2Model(num_classes=len(self.class_order))
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval().to(self.device)
        self.best_val_metrics = ckpt.get("best_val_metrics", {})

    def close(self):
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None

    def _resolve_root(self):
        if self.package_path.is_dir():
            return self.package_path
        if self.package_path.suffix.lower() == ".zip":
            self._tempdir = tempfile.TemporaryDirectory(prefix="mmrt_v2_pkg_")
            with zipfile.ZipFile(self.package_path) as zf:
                zf.extractall(self._tempdir.name)
            return Path(self._tempdir.name)
        raise FileNotFoundError(f"Unsupported package path: {self.package_path}")

    def _find_checkpoint(self):
        candidates = [
            self.root / "mmrt_multimodal_full_v2_t4x2" / "best_full_v2_t4x2.pt",
            self.root / "best_full_v2_t4x2.pt",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"MMRT v2 checkpoint not found under {self.root}")

    def predict_clip(self, video_path: str | Path, frame_count=16, start_seconds=0.0, end_seconds=3.0):
        body_frames, face_frames, face_hits = _extract_body_face_frames(Path(video_path), frame_count, start_seconds, end_seconds)
        logmel, scalars, audio_detected, audio_rms = _extract_audio_features(Path(video_path), start_seconds, end_seconds)
        skeleton_seq, skeleton_hits = _extract_skeleton(Path(video_path), frame_count, start_seconds, end_seconds)
        return self.predict_from_modalities(
            body_frames=body_frames,
            face_frames=face_frames,
            skeleton_seq=skeleton_seq,
            audio_wave=None,
            audio_logmel=logmel,
            audio_scalars=scalars,
            audio_detected=audio_detected,
            audio_rms=audio_rms,
            face_hits=face_hits,
            skeleton_hits=skeleton_hits,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            frame_count=frame_count,
        )

    def predict_video(self, video_path: str | Path, clip_seconds=3.0, stride_seconds=1.5, frame_count=16):
        meta = _video_meta(Path(video_path))
        windows = _window_times(meta["duration"], clip_seconds, stride_seconds)
        clips = []
        for start, end in windows:
            clips.append(self.predict_clip(video_path, frame_count=frame_count, start_seconds=start, end_seconds=end))
        top = max(clips, key=lambda item: item.confidence) if clips else None
        return meta, clips, top

    def predict_from_modalities(
        self,
        *,
        body_frames,
        face_frames,
        skeleton_seq,
        audio_wave=None,
        audio_sample_rate=16000,
        audio_logmel=None,
        audio_scalars=None,
        audio_detected=True,
        audio_rms=None,
        face_hits=0,
        skeleton_hits=0,
        start_seconds=0.0,
        end_seconds=0.0,
        frame_count=16,
    ):
        if audio_logmel is None or audio_scalars is None:
            audio_logmel, audio_scalars, audio_detected, computed_rms = _audio_features_from_wave(
                audio_wave,
                audio_sample_rate,
                audio_detected=audio_detected,
            )
            if audio_rms is None:
                audio_rms = computed_rms
        if audio_rms is None:
            audio_rms = 0.0

        if isinstance(body_frames, np.ndarray):
            body_frames = torch.from_numpy(body_frames)
        if isinstance(face_frames, np.ndarray):
            face_frames = torch.from_numpy(face_frames)

        with torch.no_grad():
            logits = self.model(
                body_frames.unsqueeze(0).to(self.device, dtype=torch.float32),
                face_frames.unsqueeze(0).to(self.device, dtype=torch.float32),
                torch.from_numpy(np.asarray(audio_logmel, dtype=np.float32)).unsqueeze(0).to(self.device),
                torch.from_numpy(np.asarray(audio_scalars, dtype=np.float32)).unsqueeze(0).to(self.device),
                torch.from_numpy(np.asarray(skeleton_seq, dtype=np.float32)).unsqueeze(0).to(self.device),
            ).detach().cpu().numpy()[0]
        probs = _softmax(logits)
        ranked = [{"class_name": name, "confidence": float(prob)} for name, prob in zip(self.class_order, probs)]
        ranked.sort(key=lambda item: item["confidence"], reverse=True)
        return MMRTV2ClipResult(
            start_seconds=float(start_seconds),
            end_seconds=float(end_seconds),
            label=str(ranked[0]["class_name"]),
            confidence=float(ranked[0]["confidence"]),
            probabilities=ranked,
            face_frames_detected=int(face_hits),
            skeleton_frames_detected=int(skeleton_hits),
            audio_track_detected=bool(audio_detected),
            audio_rms=float(audio_rms),
            frame_count=int(frame_count),
        )


def save_mmrt_demo_run(output_dir: Path, payload: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    path = output_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_id, path


def clip_result_to_dict(result: MMRTV2ClipResult):
    return asdict(result)
