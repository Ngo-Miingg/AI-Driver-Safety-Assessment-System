from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .database import get_db


def json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def get_model_artifact_id(backend_key: str) -> int | None:
    row = get_db().execute(
        "SELECT id FROM model_artifacts WHERE backend_key = ?",
        (backend_key,),
    ).fetchone()
    return int(row["id"]) if row else None


def create_media_asset(
    *,
    user_id: int,
    storage_path: str | Path,
    asset_type: str,
    driver_id: int | None = None,
    session_id: int | None = None,
    original_filename: str | None = None,
    mime_type: str | None = None,
    file_size_bytes: int | None = None,
    width: int | None = None,
    height: int | None = None,
    duration_seconds: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> int:
    cursor = get_db().execute(
        """
        INSERT INTO media_assets (
            user_id, driver_id, session_id, asset_type, storage_path,
            original_filename, mime_type, file_size_bytes, width, height,
            duration_seconds, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            driver_id,
            session_id,
            asset_type,
            str(storage_path),
            original_filename,
            mime_type,
            file_size_bytes,
            width,
            height,
            duration_seconds,
            json_dumps(metadata),
        ),
    )
    return int(cursor.lastrowid)


def upsert_driver_biometric(
    *,
    user_id: int,
    driver_id: int,
    biometric_type: str,
    model_backend: str,
    embedding_json: str,
    enrollment_asset_id: int | None = None,
    threshold: float | None = None,
) -> int:
    db = get_db()
    db.execute(
        """
        INSERT INTO driver_biometrics (
            user_id, driver_id, biometric_type, model_backend, embedding_json,
            enrollment_asset_id, threshold, status, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, 'active', CURRENT_TIMESTAMP)
        ON CONFLICT(driver_id, biometric_type, model_backend) DO UPDATE SET
            user_id = excluded.user_id,
            embedding_json = excluded.embedding_json,
            enrollment_asset_id = excluded.enrollment_asset_id,
            threshold = excluded.threshold,
            status = 'active',
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            user_id,
            driver_id,
            biometric_type,
            model_backend,
            embedding_json,
            enrollment_asset_id,
            threshold,
        ),
    )
    row = db.execute(
        """
        SELECT id
        FROM driver_biometrics
        WHERE driver_id = ? AND biometric_type = ? AND model_backend = ?
        """,
        (driver_id, biometric_type, model_backend),
    ).fetchone()
    return int(row["id"])


def get_active_driver_biometric(driver_id: int, biometric_type: str, model_backend: str):
    return get_db().execute(
        """
        SELECT *
        FROM driver_biometrics
        WHERE driver_id = ?
          AND biometric_type = ?
          AND model_backend = ?
          AND status = 'active'
        """,
        (driver_id, biometric_type, model_backend),
    ).fetchone()


def deactivate_driver_biometric(driver_id: int, biometric_type: str, model_backend: str):
    get_db().execute(
        """
        UPDATE driver_biometrics
        SET status = 'inactive', updated_at = CURRENT_TIMESTAMP
        WHERE driver_id = ? AND biometric_type = ? AND model_backend = ?
        """,
        (driver_id, biometric_type, model_backend),
    )


def create_inference_run(
    *,
    user_id: int,
    run_type: str,
    driver_id: int | None = None,
    session_id: int | None = None,
    model_backend: str | None = None,
    model_artifact_id: int | None = None,
    input_asset_id: int | None = None,
    status: str = "running",
) -> int:
    cursor = get_db().execute(
        """
        INSERT INTO inference_runs (
            user_id, driver_id, session_id, run_type, status, model_backend,
            model_artifact_id, input_asset_id, started_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            user_id,
            driver_id,
            session_id,
            run_type,
            status,
            model_backend,
            model_artifact_id,
            input_asset_id,
        ),
    )
    return int(cursor.lastrowid)


def finish_inference_run(run_id: int, *, status: str, started_monotonic: float, summary: dict[str, Any] | None = None, error: str | None = None):
    latency_ms = max((time.monotonic() - started_monotonic) * 1000.0, 0.0)
    get_db().execute(
        """
        UPDATE inference_runs
        SET status = ?, ended_at = CURRENT_TIMESTAMP, latency_ms = ?,
            summary_json = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (status, latency_ms, json_dumps(summary), error, run_id),
    )


def create_session_model(
    *,
    session_id: int,
    model_backend: str,
    model_role: str,
    model_artifact_id: int | None = None,
    threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
):
    get_db().execute(
        """
        INSERT INTO session_models (
            session_id, model_artifact_id, model_backend, model_role, threshold, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, model_artifact_id, model_backend, model_role, threshold, json_dumps(metadata)),
    )


def create_model_prediction(
    *,
    model_backend: str,
    run_id: int | None = None,
    session_id: int | None = None,
    timestamp_seconds: float | None = None,
    frame_index: int | None = None,
    label: str | None = None,
    confidence: float | None = None,
    probabilities: dict[str, Any] | None = None,
    features: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
):
    get_db().execute(
        """
        INSERT INTO model_predictions (
            run_id, session_id, timestamp_seconds, frame_index, model_backend,
            label, confidence, probabilities_json, features_json, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            session_id,
            timestamp_seconds,
            frame_index,
            model_backend,
            label,
            confidence,
            json_dumps(probabilities),
            json_dumps(features),
            json_dumps(metadata),
        ),
    )


def create_audit_log(
    *,
    action: str,
    user_id: int | None = None,
    driver_id: int | None = None,
    session_id: int | None = None,
    entity_type: str | None = None,
    entity_id: str | int | None = None,
    metadata: dict[str, Any] | None = None,
):
    get_db().execute(
        """
        INSERT INTO app_audit_logs (
            user_id, driver_id, session_id, action, entity_type, entity_id, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, driver_id, session_id, action, entity_type, str(entity_id) if entity_id is not None else None, json_dumps(metadata)),
    )
