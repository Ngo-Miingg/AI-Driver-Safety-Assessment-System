from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


DEFAULT_BEHAVIOR_LABELS = [
    "safe driving",
    "texting right",
    "talking phone right",
    "texting left",
    "talking phone left",
    "operating radio",
    "drinking",
    "reaching behind",
    "hair and makeup",
    "talking to passenger",
]


@dataclass(frozen=True)
class BehaviorPrediction:
    class_id: int
    class_name: str
    confidence: float
    probabilities: list[dict]


class UnavailableBehaviorRecognizer:
    backend_name = "unavailable"

    def __init__(self, error: str):
        self.error = error

    @property
    def is_available(self):
        return False

    def predict_frame(self, frame, top_k=5):
        raise RuntimeError(self.error)


class ViTDDRecognizer:
    """Lightweight ViT-DD-style image/frame classifier adapter.

    ViT-DD is a Vision Transformer driver-distraction classifier. The original
    project trains a multi-task ViT, while this adapter keeps app integration
    practical by loading a ViT checkpoint with the same c0-c9 behavior classes.
    """

    backend_name = "vitdd"

    def __init__(
        self,
        model_path,
        model_name="vit_base_patch16_224",
        labels=None,
        image_size=224,
        device="auto",
    ):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.labels = labels or DEFAULT_BEHAVIOR_LABELS
        self.image_size = int(image_size)
        self.error = None

        try:
            import torch
            import timm
            from torchvision import transforms
        except Exception as exc:  # pragma: no cover - depends on optional deps.
            raise RuntimeError(
                "Missing optional behavior-recognition dependencies. Install torch, torchvision, and timm."
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"Behavior model file not found at {self.model_path}.")

        self.torch = torch
        self.transforms = transforms
        self.device = self._resolve_device(device)
        self.model = timm.create_model(
            self.model_name,
            pretrained=False,
            num_classes=len(self.labels),
        )
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @property
    def is_available(self):
        return self.error is None

    def predict_frame(self, frame, top_k=5):
        if frame is None:
            raise RuntimeError("Empty frame cannot be classified.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self.transforms.ToPILImage()(rgb)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            logits = self.model(tensor)
            scores = self.torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        ranked = np.argsort(scores)[::-1][: max(1, min(int(top_k), len(self.labels)))]
        probabilities = [
            {
                "class_id": int(index),
                "class_name": self.labels[int(index)],
                "confidence": float(scores[int(index)]),
            }
            for index in ranked
        ]
        best = probabilities[0]
        return BehaviorPrediction(
            class_id=best["class_id"],
            class_name=best["class_name"],
            confidence=best["confidence"],
            probabilities=probabilities,
        )

    def _resolve_device(self, device):
        if device and device != "auto":
            return self.torch.device(device)
        return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

    def _load_checkpoint(self):
        checkpoint = self.torch.load(self.model_path, map_location="cpu")
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict", "net"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break

        cleaned = {}
        for key, value in state_dict.items():
            clean_key = key
            for prefix in ("model.", "module.", "net.", "student."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
            cleaned[clean_key] = value

        self.model.load_state_dict(cleaned, strict=False)


class BehaviorRecognizerFactory:
    @staticmethod
    def create(model_path, model_name="vit_base_patch16_224", labels=None, image_size=224, device="auto", strict=False):
        try:
            return ViTDDRecognizer(
                model_path=model_path,
                model_name=model_name,
                labels=labels,
                image_size=image_size,
                device=device,
            )
        except Exception as exc:
            if strict:
                raise
            return UnavailableBehaviorRecognizer(str(exc))


def parse_behavior_labels(raw_value):
    if not raw_value:
        return DEFAULT_BEHAVIOR_LABELS
    labels = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return labels or DEFAULT_BEHAVIOR_LABELS
