from __future__ import annotations

"""Verify the Driver Safety v1 artifact and inference contract."""

import argparse
import hashlib
import h5py
import json
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "driver_safety_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_dir: Path, load_detector: bool = False) -> dict:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: dict[str, dict] = {}
    failures: list[str] = []

    for filename, specification in manifest["artifacts"].items():
        path = model_dir / filename
        present = path.is_file()
        actual_hash = sha256(path) if present else None
        hash_ok = present and actual_hash == specification["sha256"]
        checks[filename] = {
            "present": present,
            "sha256_ok": hash_ok,
            "role": specification["runtime_role"],
        }
        if not present:
            failures.append(f"missing artifact: {filename}")
        elif not hash_ok:
            failures.append(f"hash mismatch: {filename}")

    if failures:
        return {"ok": False, "checks": checks, "failures": failures}

    with h5py.File(model_dir / "face_verification.h5", "r") as handle:
        verifier_config = json.loads(handle.attrs["model_config"])["config"]
    verifier_input = verifier_config["layers"][0]["config"].get("batch_input_shape")
    verifier_dense = [
        layer["config"].get("units")
        for layer in verifier_config["layers"]
        if layer["class_name"] == "Dense"
    ]
    verifier_output = verifier_config["layers"][-1]
    verifier_ok = (
        verifier_input == [None, 112, 112, 3]
        and verifier_dense[-1:] == [128]
        and verifier_output["class_name"] == "TFOpLambda"
        and verifier_output["config"].get("function") == "math.l2_normalize"
    )
    checks["face_verifier_contract"] = {
        "ok": verifier_ok,
        "input_shape": verifier_input,
        "embedding_dimensions": verifier_dense[-1] if verifier_dense else None,
        "l2_normalized": verifier_output["config"].get("function") == "math.l2_normalize",
        "reported_euclidean_threshold": manifest.get("identity_contract", {}).get("reported_threshold"),
        "runtime_status": manifest.get("identity_contract", {}).get("runtime_status"),
    }
    if not verifier_ok:
        failures.append("face verification architecture does not match the report contract")

    import joblib

    contract = manifest["feature_contract"]
    expected_order = [
        "left_ear",
        "right_ear",
        "mar",
        "pitch_degrees",
        "yaw_degrees",
        "roll_degrees",
    ]
    feature_order_ok = contract.get("feature_order") == expected_order
    checks["feature_order_contract"] = {
        "ok": feature_order_ok,
        "feature_order": contract.get("feature_order"),
        "source_commit": manifest.get("provenance", {}).get("verified_source_commit"),
    }
    if not feature_order_ok:
        failures.append("feature order does not match the released scaler/model contract")
    scaler = joblib.load(model_dir / "scaler.joblib")
    scaler_features = int(getattr(scaler, "n_features_in_", -1))
    scaler_ok = scaler_features == int(contract["feature_count"])
    checks["scaler_contract"] = {
        "ok": scaler_ok,
        "n_features_in": scaler_features,
        "n_samples_seen": int(np.asarray(getattr(scaler, "n_samples_seen_", 0)).max()),
    }
    if not scaler_ok:
        failures.append("scaler feature count does not match manifest")

    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras

    @keras.saving.register_keras_serializable()
    class CompatLSTM(keras.layers.LSTM):
        def __init__(self, *args, **kwargs):
            kwargs.pop("time_major", None)
            super().__init__(*args, **kwargs)

    model = keras.saving.load_model(
        model_dir / "best_model.h5",
        compile=False,
        custom_objects={
            "LSTM": CompatLSTM,
            "CompatLSTM": CompatLSTM,
            "Bidirectional": keras.layers.Bidirectional,
            "Dropout": keras.layers.Dropout,
            "Dense": keras.layers.Dense,
            "InputLayer": keras.layers.InputLayer,
        },
    )
    expected_shape = (
        None,
        int(contract["sequence_length"]),
        int(contract["feature_count"]),
    )
    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    shape_ok = input_shape == expected_shape and output_shape[-1] == 1
    checks["drowsiness_model_contract"] = {
        "ok": shape_ok,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }
    if not shape_ok:
        failures.append("drowsiness model shape does not match manifest")

    raw_sequence = np.tile(np.asarray(scaler.mean_, dtype=np.float32), (contract["sequence_length"], 1))
    scaled = scaler.transform(raw_sequence).reshape(1, contract["sequence_length"], contract["feature_count"])
    probability = float(np.asarray(model.predict(scaled, verbose=0)).reshape(-1)[0])
    inference_ok = np.isfinite(probability) and 0.0 <= probability <= 1.0
    checks["drowsiness_smoke_inference"] = {"ok": bool(inference_ok), "probability": probability}
    if not inference_ok:
        failures.append("drowsiness smoke inference returned an invalid probability")

    if load_detector:
        from ultralytics import YOLO

        detector = YOLO(model_dir / "face_detection.pt")
        result = detector(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False)[0]
        detector_contract_ok = detector.names == {0: "face"}
        checks["face_detector_load"] = {
            "ok": result is not None and detector_contract_ok,
            "classes": detector.names,
            "training_imgsz": detector.overrides.get("imgsz"),
            "note": "Load/inference smoke test only; not an accuracy test.",
        }
        if not detector_contract_ok:
            failures.append("face detector is not the expected one-class face model")

    return {"ok": not failures, "checks": checks, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--load-detector", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.model_dir.resolve(), load_detector=args.load_detector)
    except Exception as exc:
        result = {"ok": False, "checks": {}, "failures": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=list))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
