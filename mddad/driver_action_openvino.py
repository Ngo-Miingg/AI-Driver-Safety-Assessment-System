from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Keep Windows DLL directory handles alive for the entire process.  If the
# handle returned by os.add_dll_directory is discarded, Windows can remove
# the directory from the DLL search path before OpenVINO loads its plugins.
_OPENVINO_DLL_DIRECTORY_HANDLES = []
logger = logging.getLogger(__name__)


ACTION_LABELS = (
    "normal_driving",
    "phone_texting",
    "phone_texting",
    "phone_calling",
    "phone_calling",
    "operating_radio",
    "drinking_eating",
    "reaching_behind",
    "hair_and_makeup",
)

# Public product contract: left/right variants are merged before any output is
# sent to the UI, event layer, or session metadata.
OPENVINO_LABELS = tuple(dict.fromkeys(ACTION_LABELS))

RISK_EVENT_BY_LABEL = {
    "phone_texting": "phone_usage",
    "phone_calling": "phone_call",
    "operating_radio": "in_vehicle_display_interaction",
    "drinking_eating": "eating_distraction",
    "reaching_behind": "visual_posture_distraction",
    "hair_and_makeup": "visual_posture_distraction",
}


def _prepare_openvino_windows_runtime() -> None:
    if os.name != "nt":
        return
    spec = importlib.util.find_spec("openvino")
    if spec is None or spec.origin is None:
        return
    libs_dir = Path(spec.origin).resolve().parent / "libs"
    if not libs_dir.exists():
        return
    libs_path = str(libs_dir)
    current_path = os.environ.get("PATH", "")
    if libs_path.lower() not in current_path.lower():
        os.environ["PATH"] = libs_path + os.pathsep + current_path
    add_dir = getattr(os, "add_dll_directory", None)
    if add_dir is not None:
        try:
            if not any(str(handle.path).lower() == libs_path.lower()
                       for handle in _OPENVINO_DLL_DIRECTORY_HANDLES):
                _OPENVINO_DLL_DIRECTORY_HANDLES.append(add_dir(libs_path))
        except Exception:
            pass


@dataclass(frozen=True)
class DriverActionObservation:
    ready: bool
    buffer_size: int
    buffer_target: int
    prediction_updated: bool
    label: str | None = None
    raw_label: str | None = None
    class_index: int | None = None
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    risk_event_type: str | None = None
    error: str | None = None
    instant_label: str | None = None
    instant_raw_label: str | None = None
    instant_confidence: float | None = None
    stable_ready: bool = False
    decision_threshold: float | None = None


def action_model_readiness(app) -> dict[str, Any]:
    paths = {
        "action_encoder": Path(app.config["DRIVER_ACTION_ENCODER_PATH"]),
        "action_decoder": Path(app.config["DRIVER_ACTION_DECODER_PATH"]),
    }
    available = {name: path.exists() and path.is_file() for name, path in paths.items()}
    try:
        _prepare_openvino_windows_runtime()
        import openvino as ov  # noqa: F401

        runtime_available = True
        runtime_error = None
    except Exception as exc:
        runtime_available = False
        runtime_error = str(exc)
    return {
        "backend_key": "driver_safety_v1.action_recognition",
        "available": all(available.values()) and runtime_available,
        "runtime_available": runtime_available,
        "runtime_error": runtime_error,
        "python_executable": os.sys.executable,
        "openvino_path": getattr(ov, "__file__", None) if runtime_available else None,
        "models": {
            name: {"available": available[name]}
            for name in ("action_encoder", "action_decoder")
        },
        "window_size": int(app.config.get("DRIVER_ACTION_WINDOW_SIZE", 16)),
        "sample_stride_frames": int(app.config.get("DRIVER_ACTION_SAMPLE_STRIDE_FRAMES", 2)),
        "inference_stride": int(app.config.get("DRIVER_ACTION_INFERENCE_STRIDE", 1)),
        "label_smoothing_frames": int(app.config.get("DRIVER_ACTION_LABEL_SMOOTHING_FRAMES", 15)),
        "classes": list(ACTION_LABELS),
        "openvino_classes": list(OPENVINO_LABELS),
    }


def resolve_openvino_device(configured_device: str | None, available_devices: list[str] | None) -> str:
    requested = str(configured_device or "auto").strip()
    available = [str(device) for device in (available_devices or [])]
    if not available:
        return requested or "CPU"
    if requested and requested.lower() not in {"auto", "auto_gpu"}:
        return requested
    for candidate in ("GPU.1", "GPU.0", "GPU", "AUTO", "CPU"):
        if candidate == "AUTO" or candidate in available:
            return candidate
    return "CPU"


class DriverActionOpenVINORuntime:
    def __init__(self, app):
        self.app = app
        self.window_size = int(app.config.get("DRIVER_ACTION_WINDOW_SIZE", 16))
        self.sample_stride_frames = max(int(app.config.get("DRIVER_ACTION_SAMPLE_STRIDE_FRAMES", 2)), 1)
        self.inference_stride = max(int(app.config.get("DRIVER_ACTION_INFERENCE_STRIDE", 1)), 1)
        self.label_smoothing_frames = max(
            int(app.config.get("DRIVER_ACTION_LABEL_SMOOTHING_FRAMES", 15)),
            1,
        )
        self.confidence_threshold = float(app.config.get("DRIVER_ACTION_CONFIDENCE_THRESHOLD", 0.45))
        self.drinking_eating_threshold = float(
            app.config.get("DRIVER_ACTION_DRINKING_EATING_THRESHOLD", 0.70)
        )
        self._embeddings: deque[np.ndarray] = deque(maxlen=self.window_size)
        self._label_history: deque[int] = deque(maxlen=self.label_smoothing_frames)
        self._stable_class_index: int | None = None
        self._seen_frames = 0
        self._encoded_count = 0
        self._last_inference_count = 0
        self._encoder = None
        self._decoder = None
        self._encoder_input = None
        self._encoder_output = None
        self._decoder_input = None
        self._decoder_output = None
        self._resolved_device = None
        self._last_observation = DriverActionObservation(
            ready=False,
            buffer_size=0,
            buffer_target=self.window_size,
            prediction_updated=False,
            error="not_loaded",
        )
        self._load_error: str | None = None
        self._load_runtime()

    @property
    def ready(self) -> bool:
        return self._load_error is None and self._encoder is not None and self._decoder is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def resolved_device(self) -> str | None:
        return self._resolved_device

    def _load_runtime(self):
        readiness = action_model_readiness(self.app)
        if not readiness["models"]["action_encoder"]["available"]:
            self._load_error = "Missing OpenVINO action encoder XML/BIN."
            return
        if not readiness["models"]["action_decoder"]["available"]:
            self._load_error = "Missing OpenVINO action decoder XML/BIN."
            return
        if not readiness["runtime_available"]:
            self._load_error = f"OpenVINO runtime unavailable: {readiness['runtime_error']}"
            return
        try:
            _prepare_openvino_windows_runtime()
            import openvino as ov

            core = ov.Core()
            device = resolve_openvino_device(
                self.app.config.get("DRIVER_ACTION_DEVICE", "auto"),
                list(core.available_devices),
            )
            self._encoder = core.compile_model(self.app.config["DRIVER_ACTION_ENCODER_PATH"], device)
            self._decoder = core.compile_model(self.app.config["DRIVER_ACTION_DECODER_PATH"], device)
            self._encoder_input = self._encoder.input(0)
            self._encoder_output = self._encoder.output(0)
            self._decoder_input = self._decoder.input(0)
            self._decoder_output = self._decoder.output(0)
            self._resolved_device = device
            logger.info(
                "OpenVINO action runtime ready executable=%s openvino=%s devices=%s device=%s",
                os.sys.executable,
                getattr(ov, "__file__", None),
                list(core.available_devices),
                device,
            )
        except Exception as exc:
            self._load_error = f"OpenVINO action model could not load: {exc}"
            logger.exception(
                "OpenVINO action runtime failed executable=%s openvino=%s encoder=%s decoder=%s",
                os.sys.executable,
                readiness.get("openvino_path"),
                self.app.config.get("DRIVER_ACTION_ENCODER_PATH"),
                self.app.config.get("DRIVER_ACTION_DECODER_PATH"),
            )

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values.astype(np.float32).reshape(-1)
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        total = np.sum(exp)
        if not np.isfinite(total) or total <= 0:
            return np.zeros_like(values, dtype=np.float32)
        return exp / total

    @staticmethod
    def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("OpenVINO action model received an empty frame.")

        # Match the Open Model Zoo action-recognition demo: resize the short
        # side to 224, preserve aspect ratio, then take a 224x224 center crop.
        scale = 224.0 / min(height, width)
        resized_width = max(224, int(width * scale))
        resized_height = max(224, int(height * scale))
        resized = cv2.resize(
            frame,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        x0 = max(int(round((resized_width - 224) / 2.0)), 0)
        y0 = max(int(round((resized_height - 224) / 2.0)), 0)
        cropped = resized[y0:y0 + 224, x0:x0 + 224]
        blob = cropped.astype(np.float32).transpose(2, 0, 1)
        return blob.reshape(1, 3, 224, 224)

    def _encode_frame(self, frame: np.ndarray) -> np.ndarray:
        assert self._encoder is not None
        output = self._encoder({self._encoder_input: self._preprocess_frame(frame)})[self._encoder_output]
        return np.asarray(output, dtype=np.float32).reshape(512)

    def _decode_window(self) -> DriverActionObservation:
        assert self._decoder is not None
        sequence = np.asarray(list(self._embeddings), dtype=np.float32).reshape(1, self.window_size, 512)
        logits = np.asarray(self._decoder({self._decoder_input: sequence})[self._decoder_output]).reshape(-1)
        probabilities = self._softmax(logits)
        instant_class_index = int(np.argmax(probabilities))
        instant_confidence = float(probabilities[instant_class_index])
        self._label_history.append(instant_class_index)

        history_is_stable = (
            len(self._label_history) == self.label_smoothing_frames
            and len(set(self._label_history)) == 1
        )
        if history_is_stable:
            self._stable_class_index = instant_class_index

        stable_class_index = self._stable_class_index
        stable_confidence = (
            float(probabilities[stable_class_index])
            if stable_class_index is not None
            else None
        )
        decision_threshold = self.drinking_eating_threshold if (
            stable_class_index is not None
            and ACTION_LABELS[stable_class_index] == "drinking_eating"
        ) else self.confidence_threshold
        accepted = (
            stable_class_index is not None
            and stable_confidence is not None
            and stable_confidence >= decision_threshold
        )
        class_index = stable_class_index if accepted else None
        label = ACTION_LABELS[class_index] if class_index is not None else None
        confidence = stable_confidence if accepted else None
        risk_event_type = None
        if label is not None:
            risk_event_type = RISK_EVENT_BY_LABEL.get(label)
        return DriverActionObservation(
            ready=True,
            buffer_size=len(self._embeddings),
            buffer_target=self.window_size,
            prediction_updated=True,
            label=label,
            raw_label=ACTION_LABELS[class_index] if class_index is not None else None,
            class_index=class_index,
            confidence=confidence,
            probabilities={
                label_name: float(sum(
                    probabilities[index]
                    for index, mapped_label in enumerate(ACTION_LABELS)
                    if mapped_label == label_name
                ))
                for label_name in OPENVINO_LABELS
            },
            risk_event_type=risk_event_type,
            instant_label=ACTION_LABELS[instant_class_index],
            instant_raw_label=ACTION_LABELS[instant_class_index],
            instant_confidence=instant_confidence,
            stable_ready=stable_class_index is not None,
            decision_threshold=decision_threshold,
        )

    def observe(self, frame: np.ndarray) -> DriverActionObservation:
        if not self.ready:
            return DriverActionObservation(
                ready=False,
                buffer_size=len(self._embeddings),
                buffer_target=self.window_size,
                prediction_updated=False,
                error=self._load_error,
            )
        try:
            self._seen_frames += 1
            if (self._seen_frames - 1) % self.sample_stride_frames != 0:
                return DriverActionObservation(
                    ready=True,
                    buffer_size=len(self._embeddings),
                    buffer_target=self.window_size,
                    prediction_updated=False,
                    label=self._last_observation.label,
                    raw_label=self._last_observation.raw_label,
                    class_index=self._last_observation.class_index,
                    confidence=self._last_observation.confidence,
                    probabilities=self._last_observation.probabilities,
                    risk_event_type=self._last_observation.risk_event_type,
                    instant_label=self._last_observation.instant_label,
                    instant_raw_label=self._last_observation.instant_raw_label,
                    instant_confidence=self._last_observation.instant_confidence,
                    stable_ready=self._last_observation.stable_ready,
                    decision_threshold=self._last_observation.decision_threshold,
                )
            self._embeddings.append(self._encode_frame(frame))
            self._encoded_count += 1
            if len(self._embeddings) < self.window_size:
                return DriverActionObservation(
                    ready=True,
                    buffer_size=len(self._embeddings),
                    buffer_target=self.window_size,
                    prediction_updated=False,
                    label=self._last_observation.label,
                    raw_label=self._last_observation.raw_label,
                    class_index=self._last_observation.class_index,
                    confidence=self._last_observation.confidence,
                    probabilities=self._last_observation.probabilities,
                    risk_event_type=self._last_observation.risk_event_type,
                    instant_label=self._last_observation.instant_label,
                    instant_raw_label=self._last_observation.instant_raw_label,
                    instant_confidence=self._last_observation.instant_confidence,
                    stable_ready=self._last_observation.stable_ready,
                    decision_threshold=self._last_observation.decision_threshold,
                )
            if self._encoded_count - self._last_inference_count < self.inference_stride:
                return DriverActionObservation(
                    ready=True,
                    buffer_size=len(self._embeddings),
                    buffer_target=self.window_size,
                    prediction_updated=False,
                    label=self._last_observation.label,
                    raw_label=self._last_observation.raw_label,
                    class_index=self._last_observation.class_index,
                    confidence=self._last_observation.confidence,
                    probabilities=self._last_observation.probabilities,
                    risk_event_type=self._last_observation.risk_event_type,
                    instant_label=self._last_observation.instant_label,
                    instant_raw_label=self._last_observation.instant_raw_label,
                    instant_confidence=self._last_observation.instant_confidence,
                    stable_ready=self._last_observation.stable_ready,
                    decision_threshold=self._last_observation.decision_threshold,
                )
            self._last_inference_count = self._encoded_count
            self._last_observation = self._decode_window()
            return self._last_observation
        except Exception as exc:
            self._load_error = f"OpenVINO action inference failed: {exc}"
            return DriverActionObservation(
                ready=False,
                buffer_size=len(self._embeddings),
                buffer_target=self.window_size,
                prediction_updated=False,
                error=self._load_error,
            )
