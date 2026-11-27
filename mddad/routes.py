from pathlib import Path

from flask import flash, redirect, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from .database import get_db
from .driver_safety_v1 import model_readiness
from .i18n import translate


def register_routes(app):
    @app.errorhandler(RequestEntityTooLarge)
    def request_too_large(error):
        flash(
            translate(
                "flash.face_enrollment_too_large",
                "Face enrollment upload is too large. Reduce the number of camera samples or lower capture resolution.",
            ),
            "error",
        )
        target = request.referrer or url_for("dashboard")
        return redirect(target)

    @app.get("/")
    def home():
        return redirect(url_for("dashboard"))

    @app.get("/foundation")
    def foundation():
        db = get_db()
        db.execute("SELECT 1").fetchone()
        return render_template(
            "home.html",
            app_title="AI Driver Safety Assessment System",
            database_path=app.config["DATABASE_PATH"],
        )

    @app.get("/health")
    def health():
        db = get_db()
        db.execute("SELECT 1").fetchone()
        readiness = model_readiness(app)
        return {
            "status": "ok",
            "database": "connected",
            "socketio": "initialized",
            "pipeline": readiness["pipeline"],
            "drowsiness_artifacts_ready": readiness["drowsiness_artifacts_ready"],
            "identity_ready": readiness["identity_ready"],
            "legacy_research_routes": bool(app.config.get("ENABLE_LEGACY_RESEARCH_ROUTES", False)),
        }

    @app.get("/health/model")
    def model_health():
        readiness = model_readiness(app)
        phone_backend = str(app.config.get("REALTIME_PHONE_DETECTOR_BACKEND") or "disabled")
        phone_model_path = app.config.get("RTDETR_MODEL_PATH") if phone_backend == "rtdetr_finetuned" else None
        # Do not expose local absolute paths through a health endpoint.
        return {
            "pipeline": readiness["pipeline"],
            "models": {
                name: {"available": item["available"]}
                for name, item in readiness["models"].items()
            },
            "action_recognition": {
                "backend": "driver_safety_v1.action_recognition",
                "available": readiness["action_recognition_ready"],
                "runtime_available": readiness["action_recognition_runtime_available"],
                "classes": readiness["action_recognition_classes"],
            },
            "phone_detector": {
                "backend": phone_backend,
                "available": phone_backend == "disabled" or bool(phone_model_path and Path(phone_model_path).is_file()),
                "classes": ["Mobile-phone"] if phone_backend == "rtdetr_finetuned" else [],
            },
            "drowsiness_artifacts_ready": readiness["drowsiness_artifacts_ready"],
            "identity_artifact_available": readiness["identity_artifact_available"],
            "identity_ready": readiness["identity_ready"],
            "identity_status": readiness["identity_status"],
        }

    @app.get("/health/db")
    def database_health():
        db = get_db()
        table_names = [
            "users",
            "drivers",
            "sessions",
            "events",
            "risk_logs",
            "reports",
            "app_settings",
            "model_artifacts",
            "media_assets",
            "driver_biometrics",
            "inference_runs",
            "model_predictions",
            "session_models",
            "app_audit_logs",
        ]
        counts = {}
        for table_name in table_names:
            row = db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table_name,),
            ).fetchone()
            counts[table_name] = None if row is None else db.execute(
                f"SELECT COUNT(*) AS count FROM {table_name}"
            ).fetchone()["count"]
        models = db.execute(
            """
            SELECT backend_key, display_name, model_type, status
            FROM model_artifacts
            ORDER BY backend_key
            """
        ).fetchall()
        return {
            "status": "ok",
            "database": "connected",
            "tables": counts,
            "registered_models": [dict(row) for row in models],
        }
