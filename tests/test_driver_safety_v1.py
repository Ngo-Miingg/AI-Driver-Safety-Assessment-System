from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np

from mddad.driver_safety_v1 import (
    FACEMESH_LEFT_EYE,
    FEATURE_ORDER,
    DriverSafetyV1Runtime,
)
from mddad.face_identity import embedding_from_json, embedding_to_json


@dataclass
class Landmark:
    x: float = 0.0
    y: float = 0.0


def landmarks(count=468):
    return [Landmark() for _ in range(count)]


def test_left_ear_matches_training_repository_point_mapping():
    points = landmarks()
    values = {
        FACEMESH_LEFT_EYE[8]: (0.0, 0.0),
        FACEMESH_LEFT_EYE[13]: (1.0, 1.0),
        FACEMESH_LEFT_EYE[11]: (3.0, 1.0),
        FACEMESH_LEFT_EYE[0]: (4.0, 0.0),
        FACEMESH_LEFT_EYE[3]: (3.0, -1.0),
        FACEMESH_LEFT_EYE[4]: (1.0, -1.0),
    }
    for index, (x, y) in values.items():
        points[index] = Landmark(x, y)

    assert np.isclose(DriverSafetyV1Runtime._calculate_ear(points, FACEMESH_LEFT_EYE), 0.5)


def test_mar_matches_training_repository_formula():
    points = landmarks()
    points[61], points[291] = Landmark(0, 0), Landmark(4, 0)
    points[0], points[17] = Landmark(1, 1), Landmark(1, -1)
    points[40], points[37] = Landmark(3, 2), Landmark(3, -2)

    assert np.isclose(DriverSafetyV1Runtime._calculate_mar(points), 0.75)


def test_manifest_and_scaler_preserve_six_feature_contract():
    root = Path(__file__).resolve().parents[1]
    model_dir = root / "models" / "driver_safety_v1"
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    scaler = joblib.load(model_dir / "scaler.joblib")

    assert tuple(manifest["feature_contract"]["feature_order"]) == FEATURE_ORDER
    assert manifest["feature_contract"]["sequence_length"] == 15
    assert manifest["feature_contract"]["training_window_step_frames"] == 2
    assert manifest["decision_contract"] == {
        "model_positive_threshold": 0.5,
        "application_alert_threshold": 0.7,
        "application_confirmation_seconds": 3.0,
    }
    assert scaler.n_features_in_ == 6
    assert np.allclose(scaler.transform(np.asarray([scaler.mean_])), 0.0, atol=1e-6)


def test_face_identity_embedding_json_roundtrip_normalizes_128d_vector():
    raw = np.ones(128, dtype=np.float32)
    parsed = embedding_from_json(embedding_to_json(raw))

    assert parsed.shape == (128,)
    assert np.isclose(np.linalg.norm(parsed), 1.0, atol=1e-5)


def test_openvino_driver_action_runtime_runs_16_frame_window():
    from config import Config
    from mddad.driver_action_openvino import DriverActionOpenVINORuntime

    class App:
        config = Config.__dict__

    runtime = DriverActionOpenVINORuntime(App())
    assert runtime.ready, runtime.load_error

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    observation = None
    for _ in range(30):
        observation = runtime.observe(frame)
        assert not observation.prediction_updated

    observation = runtime.observe(frame)
    assert observation is not None
    assert observation.ready
    assert observation.prediction_updated
    assert observation.buffer_size == 16
    assert observation.instant_confidence is not None
    assert 0.0 <= observation.instant_confidence <= 1.0
    assert observation.probabilities is not None
    assert len(observation.probabilities) == 9
