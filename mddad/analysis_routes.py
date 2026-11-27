import json
import time

from flask import current_app, flash, g, redirect, url_for

from .analysis_models import Event
from .auth import login_required
from .database import get_db
from .mediapipe_analysis import analyze_video_with_mediapipe
from .multimodal_analysis import analyze_video_with_multimodal_model, config_from_app
from .report_generator import generate_report
from .risk_engine import contribution_for_event, summarize_session
from .settings import get_analysis_settings
from .yolo_phone_analysis import analyze_video_for_phone
from .ai.detectors import DetectorFactory
from .i18n import translate
from .repositories import (
    create_inference_run,
    create_session_model,
    finish_inference_run,
    get_model_artifact_id,
)


def _severity_for_duration(event_type, duration):
    if event_type == "phone_usage":
        if duration > 10:
            return "critical"
        if duration > 5:
            return "high"
        if duration > 2.5:
            return "medium"
        return "low"
    if event_type == "eye_closure":
        if duration >= 6:
            return "critical"
        if duration >= 3.5:
            return "high"
        if duration >= 2:
            return "medium"
        return "low"
    if duration >= 8:
        return "critical"
    if duration >= 5:
        return "high"
    if duration >= 3:
        return "medium"
    return "low"


def _bounded_event(session_id, event_type, start, duration, total_duration, confidence, explanation):
    if total_duration <= 0:
        return None
    start_time = max(0.0, min(float(start), max(total_duration - 1.0, 0.0)))
    end_time = min(start_time + duration, total_duration)
    actual_duration = max(end_time - start_time, 0.1)
    event = Event(
        session_id=session_id,
        event_type=event_type,
        start_time_seconds=round(start_time, 2),
        end_time_seconds=round(end_time, 2),
        duration_seconds=round(actual_duration, 2),
        severity=_severity_for_duration(event_type, actual_duration),
        confidence=confidence,
        explanation=explanation,
    )
    return Event(
        **{
            **event.__dict__,
            "risk_contribution": contribution_for_event(event),
        }
    )


def generate_simulated_events(session_id, duration_seconds):
    duration = float(duration_seconds or 0)
    events = []
    if duration > 20:
        events.append(_bounded_event(
            session_id,
            "yawn",
            duration * 0.25,
            3.0,
            duration,
            0.82,
            "Simulated yawn event generated because video duration is longer than 20 seconds.",
        ))
    if duration > 30:
        events.append(_bounded_event(
            session_id,
            "looking_away",
            duration * 0.45,
            5.0,
            duration,
            0.86,
            "Simulated looking-away event generated because video duration is longer than 30 seconds.",
        ))
    if duration > 40:
        events.append(_bounded_event(
            session_id,
            "phone_usage",
            duration * 0.60,
            6.0,
            duration,
            0.90,
            "Simulated phone usage event generated because video duration is longer than 40 seconds.",
        ))
    if duration > 60:
        events.append(_bounded_event(
            session_id,
            "eye_closure",
            duration * 0.78,
            4.0,
            duration,
            0.88,
            "Simulated prolonged eye closure event generated because video duration is longer than 60 seconds.",
        ))
    return [event for event in events if event is not None]


def _load_session(db, session_id):
    return db.execute(
        """
        SELECT
            s.*,
            d.display_name AS driver_name
        FROM sessions s
        JOIN drivers d ON d.id = s.driver_id
        WHERE s.id = ? AND s.user_id = ?
        """,
        (session_id, g.user["id"]),
    ).fetchone()


def _clear_previous_results(db, session_id):
    db.execute("DELETE FROM reports WHERE session_id = ?", (session_id,))
    db.execute("DELETE FROM risk_logs WHERE session_id = ?", (session_id,))
    db.execute("DELETE FROM events WHERE session_id = ?", (session_id,))


def _store_events(db, events):
    for event in events:
        db.execute(
            """
            INSERT INTO events (
                session_id,
                event_type,
                severity,
                start_time_seconds,
                end_time_seconds,
                duration_seconds,
                confidence,
                risk_contribution,
                explanation,
                evidence_image_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.session_id,
                event.event_type,
                event.severity,
                event.start_time_seconds,
                event.end_time_seconds,
                event.duration_seconds,
                event.confidence,
                event.risk_contribution,
                event.explanation,
                event.evidence_image_path,
            ),
        )


def _store_risk_logs(db, session_id, logs):
    for log in logs:
        db.execute(
            """
            INSERT INTO risk_logs (
                session_id,
                timestamp_seconds,
                risk_score,
                estimated_safety_score,
                alert_level,
                drowsiness_contribution,
                distraction_contribution,
                phone_contribution,
                face_missing_contribution,
                active_event_types,
                explanation_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                log.timestamp_seconds,
                log.risk_score,
                log.estimated_safety_score,
                log.alert_level,
                log.category_contributions.get("drowsiness", 0),
                log.category_contributions.get("distraction", 0),
                log.category_contributions.get("phone_usage", 0),
                log.category_contributions.get("visibility", 0),
                ",".join(log.active_event_types),
                json.dumps(log.category_contributions),
            ),
        )


def _store_report(db, session_id, report):
    db.execute(
        """
        INSERT INTO reports (
            session_id,
            status,
            title,
            summary,
            safety_score_explanation,
            key_risk_explanation,
            fatigue_forecast_explanation,
            recommendations,
            limitations,
            dominant_risk_breakdown,
            timeline_summary,
            generated_at
        )
        VALUES (?, 'available', ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (
            session_id,
            report.title,
            report.summary,
            report.safety_score_explanation,
            report.key_risk_explanation,
            report.fatigue_forecast_explanation,
            report.recommendations,
            report.limitations,
            report.dominant_risk_breakdown,
            report.timeline_summary,
        ),
    )


def _persist_analysis_results(db, session, events, metrics, summary, logs, report, analysis_mode, analysis_error=None):
    session_id = session["id"]
    _clear_previous_results(db, session_id)
    _store_events(db, events)
    _store_risk_logs(db, session_id, logs)
    _store_report(db, session_id, report)
    db.execute(
        """
        UPDATE sessions
        SET status = 'analyzed',
            analysis_mode = ?,
            multimodal_model_backend = ?,
            multimodal_model_path = ?,
            multimodal_clip_count = ?,
            audio_track_detected = ?,
            skeleton_frame_count = ?,
            final_risk_score = ?,
            final_safety_score = ?,
            final_alert_level = ?,
            fatigue_trend_label = ?,
            processed_frame_count = ?,
            average_processing_fps = ?,
            face_detected_frame_count = ?,
            face_missing_frame_count = ?,
            average_ear = ?,
            average_mar = ?,
            min_ear = ?,
            max_mar = ?,
            phone_detected_frame_count = ?,
            max_phone_confidence = ?,
            phone_usage_event_count = ?,
            phone_detection_error = ?,
            analysis_error = ?,
            failure_reason = NULL,
            ended_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND user_id = ?
        """,
        (
            analysis_mode,
            metrics.get("multimodal_model_backend"),
            metrics.get("multimodal_model_path"),
            metrics.get("multimodal_clip_count", 0),
            metrics.get("audio_track_detected", 0),
            metrics.get("skeleton_frame_count", 0),
            summary.final_risk_score,
            summary.final_safety_score,
            summary.alert_level,
            summary.fatigue_trend,
            metrics.get("processed_frame_count", 0),
            metrics.get("average_processing_fps"),
            metrics.get("face_detected_frame_count", 0),
            metrics.get("face_missing_frame_count", 0),
            metrics.get("average_ear"),
            metrics.get("average_mar"),
            metrics.get("min_ear"),
            metrics.get("max_mar"),
            metrics.get("phone_detected_frame_count", 0),
            metrics.get("max_phone_confidence"),
            metrics.get("phone_usage_event_count", 0),
            metrics.get("phone_detection_error"),
            analysis_error,
            session_id,
            g.user["id"],
        ),
    )


def register_analysis_routes(app):
    @app.post("/sessions/<int:session_id>/multimodal-analysis")
    @login_required
    def run_multimodal_analysis(session_id):
        db = get_db()
        session = _load_session(db, session_id)
        if session is None:
            flash(translate("flash.session_unavailable", "The requested session does not exist or is not available."), "error")
            return redirect(url_for("history"))
        if not session["saved_video_path"]:
            flash(translate("flash.multimodal_no_path", "Multimodal analysis failed because the session has no saved video path."), "error")
            return redirect(url_for("session_detail", session_id=session_id))

        from pathlib import Path

        video_path = Path(current_app.root_path).parent / session["saved_video_path"]
        model_config = config_from_app(current_app)
        input_asset = db.execute(
            """
            SELECT id
            FROM media_assets
            WHERE session_id = ? AND asset_type = 'uploaded_video'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        model_artifact_id = get_model_artifact_id(model_config.backend) or get_model_artifact_id("driver_safety_v1.drowsiness")
        run_started = time.monotonic()
        inference_run_id = create_inference_run(
            user_id=g.user["id"],
            driver_id=session["driver_id"],
            session_id=session_id,
            run_type="uploaded_video_analysis",
            model_backend=model_config.backend,
            model_artifact_id=model_artifact_id,
            input_asset_id=input_asset["id"] if input_asset else None,
        )
        db.execute(
            """
            UPDATE sessions
            SET status = 'analyzing',
                analysis_mode = 'multimodal',
                multimodal_model_backend = ?,
                multimodal_model_path = ?,
                analysis_error = NULL,
                phone_detection_error = NULL,
                failure_reason = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (
                model_config.backend,
                str(model_config.model_path),
                session_id,
                g.user["id"],
            ),
        )
        db.commit()

        try:
            result = analyze_video_with_multimodal_model(
                session_id=session_id,
                video_path=video_path,
                model_config=model_config,
            )
        except Exception as exc:
            error = f"{translate('flash.multimodal_failed', 'Multimodal analysis failed')}: {exc}"
            db.execute(
                """
                UPDATE sessions
                SET status = 'failed',
                    analysis_mode = 'multimodal',
                    multimodal_model_backend = ?,
                    multimodal_model_path = ?,
                    analysis_error = ?,
                    failure_reason = ?,
                    ended_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (
                    model_config.backend,
                    str(model_config.model_path),
                    error,
                    error,
                    session_id,
                    g.user["id"],
                ),
            )
            db.commit()
            finish_inference_run(
                inference_run_id,
                status="failed",
                started_monotonic=run_started,
                error=error,
            )
            db.commit()
            flash(error, "error")
            return redirect(url_for("session_detail", session_id=session_id))

        duration = session["video_duration_seconds"] or session["duration_seconds"] or 0
        metrics = {
            **result.metrics,
            "multimodal_model_backend": model_config.backend,
            "multimodal_model_path": str(model_config.model_path),
        }
        summary, logs = summarize_session(session_id, result.events, duration)
        report = generate_report(session, result.events, summary, analysis_mode="multimodal")
        _persist_analysis_results(db, session, result.events, metrics, summary, logs, report, "multimodal")
        create_session_model(
            session_id=session_id,
            model_backend=model_config.backend,
            model_role="uploaded_behavior_analysis",
            model_artifact_id=model_artifact_id,
            metadata={"clip_count": metrics.get("multimodal_clip_count", 0)},
        )
        finish_inference_run(
            inference_run_id,
            status="completed",
            started_monotonic=run_started,
            summary={
                "events": len(result.events),
                "processed_frame_count": metrics.get("processed_frame_count", 0),
                "average_processing_fps": metrics.get("average_processing_fps"),
            },
        )
        db.commit()
        flash(translate("flash.multimodal_done", "Multimodal video-audio-skeleton analysis completed."), "success")
        return redirect(url_for("session_detail", session_id=session_id))

    @app.post("/sessions/<int:session_id>/simulate-analysis")
    @login_required
    def run_simulated_analysis(session_id):
        flash(translate("flash.legacy_simulated_disabled", "Legacy simulated analysis has been disabled. Use multimodal video-audio-skeleton analysis."), "error")
        return redirect(url_for("session_detail", session_id=session_id))

        db = get_db()
        session = _load_session(db, session_id)
        if session is None:
            flash(translate("flash.session_unavailable", "The requested session does not exist or is not available."), "error")
            return redirect(url_for("history"))

        duration = session["video_duration_seconds"] or session["duration_seconds"] or 0
        events = generate_simulated_events(session_id, duration)
        summary, logs = summarize_session(session_id, events, duration)
        report = generate_report(session, events, summary, analysis_mode="simulated")

        _clear_previous_results(db, session_id)
        _store_events(db, events)
        _store_risk_logs(db, session_id, logs)
        _store_report(db, session_id, report)
        db.execute(
            """
            UPDATE sessions
            SET status = 'analyzed',
                analysis_mode = 'simulated',
                face_detected_frame_count = 0,
                face_missing_frame_count = 0,
                average_ear = NULL,
                average_mar = NULL,
                min_ear = NULL,
                max_mar = NULL,
                phone_detected_frame_count = 0,
                max_phone_confidence = NULL,
                phone_usage_event_count = 0,
                phone_detection_error = NULL,
                final_risk_score = ?,
                final_safety_score = ?,
                final_alert_level = ?,
                fatigue_trend_label = ?,
                analysis_error = NULL,
                failure_reason = NULL,
                ended_at = COALESCE(ended_at, CURRENT_TIMESTAMP),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (
                summary.final_risk_score,
                summary.final_safety_score,
                summary.alert_level,
                summary.fatigue_trend,
                session_id,
                g.user["id"],
            ),
        )
        db.commit()
        flash(translate("flash.simulated_done", "Simulated analysis completed. Results are deterministic and do not use AI inference yet."), "success")
        return redirect(url_for("session_detail", session_id=session_id))

    @app.post("/sessions/<int:session_id>/mediapipe-analysis")
    @login_required
    def run_mediapipe_analysis(session_id):
        flash(translate("flash.legacy_mediapipe_disabled", "Legacy MediaPipe threshold analysis has been disabled. Use multimodal video-audio-skeleton analysis."), "error")
        return redirect(url_for("session_detail", session_id=session_id))

        db = get_db()
        session = _load_session(db, session_id)
        if session is None:
            flash(translate("flash.session_unavailable", "The requested session does not exist or is not available."), "error")
            return redirect(url_for("history"))

        if not session["saved_video_path"]:
            flash(translate("flash.mediapipe_no_path", "MediaPipe analysis failed because the session has no saved video path."), "error")
            return redirect(url_for("session_detail", session_id=session_id))

        from pathlib import Path
        from flask import current_app

        video_path = Path(current_app.root_path).parent / session["saved_video_path"]
        if not video_path.exists():
            db.execute(
                """
                UPDATE sessions
                SET status = 'failed',
                    analysis_mode = 'mediapipe',
                    analysis_error = 'Video file is missing.',
                    failure_reason = 'Video file is missing.',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (session_id, g.user["id"]),
            )
            db.commit()
            flash(translate("flash.mediapipe_missing_file", "MediaPipe analysis failed because the video file is missing."), "error")
            return redirect(url_for("session_detail", session_id=session_id))

        db.execute(
            """
            UPDATE sessions
            SET status = 'analyzing',
                analysis_mode = 'mediapipe',
                analysis_error = NULL,
                failure_reason = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (session_id, g.user["id"]),
        )
        db.commit()

        settings = get_analysis_settings(db)
        try:
            _frame_results, events, metrics = analyze_video_with_mediapipe(
                session_id=session_id,
                video_path=video_path,
                sample_fps=settings["uploaded_sample_fps"],
                settings=settings,
            )
        except Exception as exc:
            error = f"{translate('flash.mediapipe_failed', 'MediaPipe analysis failed')}: {exc}"
            db.execute(
                """
                UPDATE sessions
                SET status = 'failed',
                    analysis_mode = 'mediapipe',
                    analysis_error = ?,
                    failure_reason = ?,
                    ended_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (error, error, session_id, g.user["id"]),
            )
            db.commit()
            flash(error, "error")
            return redirect(url_for("session_detail", session_id=session_id))

        duration = session["video_duration_seconds"] or session["duration_seconds"] or 0
        metrics["average_processing_fps"] = (
            metrics.get("processed_frame_count", 0) / duration if duration else None
        )
        summary, logs = summarize_session(session_id, events, duration)
        report = generate_report(session, events, summary, analysis_mode="mediapipe")

        no_face_error = None
        if metrics["processed_frame_count"] > 0 and metrics["face_detected_frame_count"] == 0:
            no_face_error = "MediaPipe processed the video but did not detect a face in any sampled frame."
        metrics.update({
            "phone_detected_frame_count": 0,
            "max_phone_confidence": None,
            "phone_usage_event_count": 0,
            "phone_detection_error": None,
        })
        _persist_analysis_results(db, session, events, metrics, summary, logs, report, "mediapipe", no_face_error)
        db.commit()

        if no_face_error:
            flash(translate("flash.mediapipe_no_face", "MediaPipe analysis completed, but no face was detected in sampled frames."), "error")
        else:
            flash(translate("flash.mediapipe_done", "MediaPipe analysis completed using real face-landmark heuristics."), "success")
        return redirect(url_for("session_detail", session_id=session_id))

    @app.post("/sessions/<int:session_id>/full-ai-analysis")
    @login_required
    def run_full_ai_analysis(session_id):
        flash(translate("flash.legacy_full_ai_disabled", "Legacy MediaPipe plus standalone phone detector analysis has been disabled. Use multimodal video-audio-skeleton analysis."), "error")
        return redirect(url_for("session_detail", session_id=session_id))

        db = get_db()
        session = _load_session(db, session_id)
        if session is None:
            flash(translate("flash.session_unavailable", "The requested session does not exist or is not available."), "error")
            return redirect(url_for("history"))
        if not session["saved_video_path"]:
            flash(translate("flash.full_ai_no_path", "Full AI analysis failed because the session has no saved video path."), "error")
            return redirect(url_for("session_detail", session_id=session_id))

        from pathlib import Path

        video_path = Path(current_app.root_path).parent / session["saved_video_path"]
        if not video_path.exists():
            db.execute(
                """
                UPDATE sessions
                SET status = 'failed',
                    analysis_mode = 'full_ai',
                    analysis_error = 'Video file is missing.',
                    failure_reason = 'Video file is missing.',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (session_id, g.user["id"]),
            )
            db.commit()
            flash(translate("flash.full_ai_missing_file", "Full AI analysis failed because the video file is missing."), "error")
            return redirect(url_for("session_detail", session_id=session_id))

        db.execute(
            """
            UPDATE sessions
            SET status = 'analyzing',
                analysis_mode = 'full_ai',
                analysis_error = NULL,
                phone_detection_error = NULL,
                failure_reason = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (session_id, g.user["id"]),
        )
        db.commit()

        settings = get_analysis_settings(db)
        try:
            _frame_results, media_events, metrics = analyze_video_with_mediapipe(
                session_id=session_id,
                video_path=video_path,
                sample_fps=settings["uploaded_sample_fps"],
                settings=settings,
            )
        except Exception as exc:
            error = f"{translate('flash.mediapipe_failed', 'MediaPipe analysis failed')}: {exc}"
            db.execute(
                """
                UPDATE sessions
                SET status = 'failed',
                    analysis_mode = 'full_ai',
                    analysis_error = ?,
                    failure_reason = ?,
                    ended_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (error, error, session_id, g.user["id"]),
            )
            db.commit()
            flash(error, "error")
            return redirect(url_for("session_detail", session_id=session_id))

        phone_detection_error = None
        phone_events = []
        phone_metrics = {
            "phone_detected_frame_count": 0,
            "max_phone_confidence": None,
            "phone_usage_event_count": 0,
            "phone_detection_error": None,
        }
        try:
            phone_backend = current_app.config["PHONE_DETECTOR_BACKEND"]
            phone_events, phone_metrics = analyze_video_for_phone(
                session_id=session_id,
                video_path=video_path,
                model_path=DetectorFactory.model_path_for_backend(current_app, phone_backend),
                confidence_threshold=settings["yolo_confidence_threshold"],
                sample_fps=settings["uploaded_sample_fps"],
                detector_backend=phone_backend,
                driver_roi=current_app.config["PHONE_DRIVER_ROI"],
                temporal_min_detections=current_app.config["PHONE_TEMPORAL_MIN_DETECTIONS"],
                temporal_merge_gap_seconds=current_app.config["PHONE_TEMPORAL_MERGE_GAP_SECONDS"],
            )
        except Exception as exc:
            phone_detection_error = str(exc)
            phone_metrics["phone_detection_error"] = phone_detection_error

        events = sorted([*media_events, *phone_events], key=lambda event: event.start_time_seconds)
        metrics.update(phone_metrics)
        no_face_error = None
        if metrics["processed_frame_count"] > 0 and metrics["face_detected_frame_count"] == 0:
            no_face_error = "MediaPipe processed the video but did not detect a face in any sampled frame."
        warnings = [warning for warning in (no_face_error, phone_detection_error) if warning]
        analysis_warning = " | ".join(warnings) if warnings else None

        duration = session["video_duration_seconds"] or session["duration_seconds"] or 0
        metrics["average_processing_fps"] = (
            metrics.get("processed_frame_count", 0) / duration if duration else None
        )
        summary, logs = summarize_session(session_id, events, duration)
        report = generate_report(session, events, summary, analysis_mode="full_ai")
        _persist_analysis_results(db, session, events, metrics, summary, logs, report, "full_ai", analysis_warning)
        db.commit()

        if phone_detection_error:
            flash(
                f"{translate('flash.full_ai_phone_failed', 'Full AI analysis completed with MediaPipe results, but phone detection failed')}: {phone_detection_error}",
                "error",
            )
        else:
            flash(translate("flash.full_ai_done", "Full AI analysis completed with MediaPipe and phone detection."), "success")
        return redirect(url_for("session_detail", session_id=session_id))
