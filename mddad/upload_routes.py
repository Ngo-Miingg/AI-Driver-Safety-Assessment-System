from pathlib import Path
from uuid import uuid4

from flask import abort, current_app, flash, g, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.utils import secure_filename

from .auth import login_required
from .database import get_db
from .evaluation import EVENT_TYPES
from .i18n import translate
from .settings import get_analysis_settings
from .video_processing import count_sampled_frames, extract_video_metadata


def _allowed_video(filename, allowed_extensions):
    if "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in allowed_extensions


def _get_active_driver():
    driver_id = session.get("active_driver_id")
    if not driver_id:
        return None
    return get_db().execute(
        """
        SELECT id, display_name
        FROM drivers
        WHERE id = ? AND user_id = ? AND status = 'active'
        """,
        (driver_id, g.user["id"]),
    ).fetchone()


def register_upload_routes(app):
    @app.route("/upload", methods=("GET", "POST"))
    @login_required
    def upload_video():
        active_driver = _get_active_driver()
        if active_driver is None:
            return render_template("upload.html", active_driver=None)

        if request.method == "POST":
            uploaded_file = request.files.get("video")
            if uploaded_file is None or uploaded_file.filename == "":
                flash(translate("flash.choose_video", "Choose a video file before uploading."), "error")
                return render_template("upload.html", active_driver=active_driver)

            original_filename = uploaded_file.filename
            if not _allowed_video(original_filename, app.config["ALLOWED_VIDEO_EXTENSIONS"]):
                flash(translate("flash.unsupported_video", "Unsupported video type. Use mp4, avi, mov, or mkv."), "error")
                return render_template("upload.html", active_driver=active_driver)

            safe_name = secure_filename(original_filename)
            if not safe_name:
                flash(translate("flash.invalid_filename", "The uploaded file name is not valid."), "error")
                return render_template("upload.html", active_driver=active_driver)

            upload_dir = Path(app.config["UPLOAD_FOLDER"])
            upload_dir.mkdir(parents=True, exist_ok=True)
            stored_name = f"{uuid4().hex}_{safe_name}"
            saved_path = upload_dir / stored_name
            uploaded_file.save(saved_path)

            file_size = saved_path.stat().st_size if saved_path.exists() else 0
            if file_size == 0:
                saved_path.unlink(missing_ok=True)
                flash(translate("flash.empty_file", "The uploaded file is empty."), "error")
                return render_template("upload.html", active_driver=active_driver)

            settings = get_analysis_settings(get_db())
            metadata = extract_video_metadata(saved_path)
            if not metadata.readable:
                saved_path.unlink(missing_ok=True)
                flash(
                    translate(
                        "flash.video_analysis_failed",
                        "Video analysis failed",
                    )
                    + f": {metadata.error}",
                    "error",
                )
                return render_template("upload.html", active_driver=active_driver)

            try:
                sampled_count = count_sampled_frames(
                    saved_path,
                    sample_fps=settings["uploaded_sample_fps"],
                )
            except ValueError as exc:
                saved_path.unlink(missing_ok=True)
                flash(
                    translate("flash.video_analysis_failed", "Video analysis failed")
                    + f": {exc}",
                    "error",
                )
                return render_template("upload.html", active_driver=active_driver)

            relative_path = str(Path("uploads") / stored_name)
            cursor = get_db().execute(
                """
                INSERT INTO sessions (
                    user_id,
                    driver_id,
                    session_type,
                    status,
                    started_at,
                    original_video_filename,
                    saved_video_path,
                    file_size_bytes,
                    video_fps,
                    video_frame_count,
                    video_duration_seconds,
                    video_width,
                    video_height,
                    processed_frame_count,
                    average_processing_fps
                )
                VALUES (?, ?, 'uploaded_video', 'pending_analysis', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    g.user["id"],
                    active_driver["id"],
                    original_filename,
                    relative_path,
                    file_size,
                    metadata.fps,
                    metadata.frame_count,
                    metadata.duration_seconds,
                    metadata.width,
                    metadata.height,
                    sampled_count,
                    sampled_count / metadata.duration_seconds if metadata.duration_seconds else None,
                ),
            )
            get_db().commit()
            flash(
                translate(
                    "flash.video_uploaded",
                    "Video uploaded successfully. Metadata is ready, and the main workflow is multimodal video-audio-skeleton analysis from the session page.",
                ),
                "success",
            )
            return redirect(url_for("session_detail", session_id=cursor.lastrowid))

        return render_template("upload.html", active_driver=active_driver)

    @app.get("/sessions/<int:session_id>")
    @login_required
    def session_detail(session_id):
        row = get_db().execute(
            """
            SELECT
                s.id,
                s.session_type,
                s.status,
                s.original_video_filename,
                s.saved_video_path,
                s.file_size_bytes,
                s.video_fps,
                s.video_frame_count,
                s.video_duration_seconds,
                s.duration_seconds,
                s.video_width,
                s.video_height,
                s.processed_frame_count,
                s.average_processing_fps,
                s.analysis_mode,
                s.multimodal_model_backend,
                s.multimodal_model_path,
                s.multimodal_clip_count,
                s.audio_track_detected,
                s.skeleton_frame_count,
                s.face_detected_frame_count,
                s.face_missing_frame_count,
                s.average_ear,
                s.average_mar,
                s.min_ear,
                s.max_mar,
                s.phone_detected_frame_count,
                s.max_phone_confidence,
                s.phone_usage_event_count,
                s.phone_detection_error,
                s.analysis_error,
                s.failure_reason,
                s.created_at,
                s.started_at,
                s.ended_at,
                s.final_safety_score,
                s.final_risk_score,
                s.final_alert_level,
                s.fatigue_trend_label,
                d.display_name AS driver_name
            FROM sessions s
            JOIN drivers d ON d.id = s.driver_id
            WHERE s.id = ? AND s.user_id = ?
            """,
            (session_id, g.user["id"]),
        ).fetchone()
        if row is None:
            return render_template(
                "placeholder.html",
                page_title="Session Not Found",
                message="The requested session does not exist or is not available for this account.",
            ), 404
        db = get_db()
        events = db.execute(
            """
            SELECT event_type, severity, start_time_seconds, end_time_seconds,
                   duration_seconds, confidence, risk_contribution, explanation,
                   evidence_image_path
            FROM events
            WHERE session_id = ?
            ORDER BY start_time_seconds ASC
            """,
            (session_id,),
        ).fetchall()
        risk_logs = db.execute(
            """
            SELECT timestamp_seconds, risk_score, estimated_safety_score, alert_level,
                   drowsiness_contribution, distraction_contribution, phone_contribution,
                   face_missing_contribution
            FROM risk_logs
            WHERE session_id = ?
            ORDER BY timestamp_seconds ASC
            """,
            (session_id,),
        ).fetchall()
        report = db.execute(
            """
            SELECT status, title, summary, safety_score_explanation, key_risk_explanation,
                   fatigue_forecast_explanation, recommendations, limitations,
                   dominant_risk_breakdown, timeline_summary, generated_at
            FROM reports
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        category_totals = {
            "drowsiness": sum((log["drowsiness_contribution"] or 0) for log in risk_logs),
            "distraction": sum((log["distraction_contribution"] or 0) for log in risk_logs),
            "phone_usage": sum((log["phone_contribution"] or 0) for log in risk_logs),
            "visibility": sum((log["face_missing_contribution"] or 0) for log in risk_logs),
        }
        total_contribution = sum(category_totals.values())
        risk_breakdown = {
            key: round((value / total_contribution) * 100, 1) if total_contribution else 0
            for key, value in category_totals.items()
        }
        event_counts = {event_type: 0 for event_type in EVENT_TYPES}
        for event in events:
            if event["event_type"] in event_counts:
                event_counts[event["event_type"]] += 1
        dominant_risk_factor = None
        if any(risk_breakdown.values()):
            dominant_risk_factor = max(risk_breakdown, key=risk_breakdown.get)
        duration = row["video_duration_seconds"] if row["video_duration_seconds"] is not None else row["duration_seconds"]
        processed = row["processed_frame_count"] or 0
        face_detected = row["face_detected_frame_count"] or 0
        evaluation_summary = {
            "duration": duration,
            "approximate_processing_fps": row["average_processing_fps"] or ((processed / duration) if duration and processed else None),
            "face_detection_rate": (face_detected / processed) if processed else None,
            "event_counts": event_counts,
            "dominant_risk_factor": dominant_risk_factor,
            "warnings": [
                warning
                for warning in (row["analysis_error"], row["phone_detection_error"], row["failure_reason"])
                if warning
            ],
        }
        return render_template(
            "session_detail.html",
            session=row,
            events=events,
            risk_logs=risk_logs,
            report=report,
            risk_breakdown=risk_breakdown,
            evaluation_summary=evaluation_summary,
        )

    @app.get("/evidence/<path:filename>")
    @login_required
    def evidence_image(filename):
        evidence_dir = Path(current_app.config["UPLOAD_FOLDER"]) / "evidence"
        requested = (evidence_dir / filename).resolve()
        try:
            requested.relative_to(evidence_dir.resolve())
        except ValueError:
            abort(404)
        if not requested.exists():
            abort(404)
        return send_from_directory(evidence_dir, filename)

    @app.post("/sessions/<int:session_id>/analyze")
    @login_required
    def analyze_uploaded_video(session_id):
        db = get_db()
        row = db.execute(
            """
            SELECT id, saved_video_path, session_type
            FROM sessions
            WHERE id = ? AND user_id = ?
            """,
            (session_id, g.user["id"]),
        ).fetchone()
        if row is None:
            flash(translate("flash.session_unavailable", "The requested session does not exist or is not available."), "error")
            return redirect(url_for("history"))

        if row["session_type"] != "uploaded_video":
            flash(translate("flash.upload_only", "Only uploaded video sessions can be analyzed here."), "error")
            return redirect(url_for("session_detail", session_id=session_id))

        if not row["saved_video_path"]:
            db.execute(
                """
                UPDATE sessions
                SET status = 'failed',
                    analysis_error = 'Session has no saved video path.',
                    failure_reason = 'Session has no saved video path.',
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (session_id, g.user["id"]),
            )
            db.commit()
            flash(translate("flash.upload_no_path", "Analysis failed because the session has no saved video path."), "error")
            return redirect(url_for("session_detail", session_id=session_id))

        video_path = Path(app.root_path).parent / row["saved_video_path"]
        db.execute(
            """
            UPDATE sessions
            SET status = 'analyzing',
                analysis_error = NULL,
                failure_reason = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (session_id, g.user["id"]),
        )
        db.commit()

        metadata = extract_video_metadata(video_path)
        if not metadata.readable:
            db.execute(
                """
                UPDATE sessions
                SET status = 'failed',
                    video_fps = ?,
                    video_frame_count = ?,
                    video_duration_seconds = ?,
                    video_width = ?,
                    video_height = ?,
                    processed_frame_count = 0,
                    analysis_error = ?,
                    failure_reason = ?,
                    ended_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (
                    metadata.fps,
                    metadata.frame_count,
                    metadata.duration_seconds,
                    metadata.width,
                    metadata.height,
                    metadata.error,
                    metadata.error,
                    session_id,
                    g.user["id"],
                ),
            )
            db.commit()
            flash(translate("flash.video_analysis_failed", "Video analysis failed") + f": {metadata.error}", "error")
            return redirect(url_for("session_detail", session_id=session_id))

        try:
            settings = get_analysis_settings(db)
            sampled_count = count_sampled_frames(video_path, sample_fps=settings["uploaded_sample_fps"])
        except ValueError as exc:
            error = str(exc)
            db.execute(
                """
                UPDATE sessions
                SET status = 'failed',
                    analysis_error = ?,
                    failure_reason = ?,
                    ended_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (error, error, session_id, g.user["id"]),
            )
            db.commit()
            flash(translate("flash.video_analysis_failed", "Video analysis failed") + f": {error}", "error")
            return redirect(url_for("session_detail", session_id=session_id))

        db.execute(
            """
            UPDATE sessions
            SET status = 'analyzed',
                video_fps = ?,
                video_frame_count = ?,
                video_duration_seconds = ?,
                video_width = ?,
                video_height = ?,
                processed_frame_count = ?,
                average_processing_fps = ?,
                analysis_error = NULL,
                failure_reason = NULL,
                ended_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (
                metadata.fps,
                metadata.frame_count,
                metadata.duration_seconds,
                metadata.width,
                metadata.height,
                sampled_count,
                sampled_count / metadata.duration_seconds if metadata.duration_seconds else None,
                session_id,
                g.user["id"],
            ),
        )
        db.commit()
        flash(translate("flash.video_metadata_extracted", "Video metadata extracted successfully. The multimodal analysis will use video, audio, and skeleton cues from this file."), "success")
        return redirect(url_for("session_detail", session_id=session_id))
