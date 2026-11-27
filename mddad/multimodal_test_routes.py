from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import current_app, flash, g, render_template, request, session
from werkzeug.utils import secure_filename

from .auth import login_required
from .i18n import translate, translate_value
from .activity7_infer import Activity7Package
from .mmrt_v2_infer import MMRTV2Package, clip_result_to_dict
from .model_options import behavior_model_options
from .multimodal_analysis import _clip_windows
from .video_processing import extract_video_metadata
from .database import get_db


def _allowed_video(filename, allowed_extensions):
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in allowed_extensions


def _is_mmrt_backend(backend):
    return str(backend or "").strip().lower().startswith("mmrt")


def _infer_model(package_path, video_path, clip_seconds, stride_seconds, frame_count, device, decision_mode="final"):
    backend = current_app.config.get("MULTIMODAL_MODEL_BACKEND", "activity7_package")
    if _is_mmrt_backend(backend):
        package = MMRTV2Package(package_path, device=device)
        try:
            metadata = extract_video_metadata(video_path)
            _, clips, top = package.predict_video(
                video_path,
                clip_seconds=clip_seconds,
                stride_seconds=stride_seconds,
                frame_count=frame_count,
            )
            predictions = []
            for clip in clips:
                clip_dict = clip_result_to_dict(clip)
                predictions.append(
                    {
                        "start_seconds": clip_dict["start_seconds"],
                        "end_seconds": clip_dict["end_seconds"],
                        "label": clip_dict["label"],
                        "confidence": clip_dict["confidence"],
                        "audio_track_detected": clip_dict["audio_track_detected"],
                        "sampled_frame_count": clip_dict["frame_count"],
                        "pose_frame_count": clip_dict["skeleton_frames_detected"],
                        "top3": clip_dict["probabilities"][:3],
                        "probabilities": clip_dict["probabilities"],
                        "baseline_label": clip_dict["label"],
                        "baseline_confidence": clip_dict["confidence"],
                        "geom2_label": clip_dict["label"],
                        "geom2_confidence": clip_dict["confidence"],
                    }
                )
            top_prediction = max(predictions, key=lambda item: item["confidence"]) if predictions else None
            return metadata, predictions, top_prediction
        finally:
            package.close()

    package = Activity7Package(package_path, device=device, decision_mode=decision_mode)
    try:
        metadata = extract_video_metadata(video_path)
        duration_seconds = metadata.duration_seconds or 0.0
        windows = _clip_windows(duration_seconds, clip_seconds, stride_seconds)
        predictions = []
        for start_seconds, end_seconds in windows:
            clip = package.predict_clip(
                video_path,
                sample_frames=frame_count,
                start_seconds=start_seconds,
                duration_seconds=max(end_seconds - start_seconds, 0.5),
            )
            pred = clip.prediction
            predictions.append(
                {
                    "start_seconds": clip.start_seconds,
                    "end_seconds": clip.end_seconds,
                    "label": pred.label,
                    "confidence": pred.confidence,
                    "audio_track_detected": clip.audio_track_detected,
                    "sampled_frame_count": clip.sampled_frame_count,
                    "pose_frame_count": clip.pose_frame_count,
                    "top3": pred.probabilities[:3],
                    "probabilities": pred.probabilities,
                    "baseline_label": pred.baseline_label,
                    "baseline_confidence": pred.baseline_confidence,
                    "geom2_label": pred.geom2_label,
                    "geom2_confidence": pred.geom2_confidence,
                }
            )
        top_prediction = max(predictions, key=lambda item: item["confidence"]) if predictions else None
        return metadata, predictions, top_prediction
    finally:
        package.close()


def register_multimodal_test_routes(app):
    @app.route("/multimodal-test", methods=("GET", "POST"))
    @app.route("/model-test-simple", methods=("GET", "POST"))
    @app.route("/model-test-simple-cam", methods=("GET", "POST"))
    @app.route("/mmrt-model-lab", methods=("GET", "POST"))
    @login_required
    def multimodal_test():
        if request.path == "/mmrt-model-lab":
            template_name = "mmrt_model_lab.html"
        elif request.path == "/model-test-simple-cam":
            template_name = "model_test_simple_cam.html"
        elif request.path == "/model-test-simple":
            template_name = "model_test_simple.html"
        else:
            template_name = "multimodal_test.html"
        result = None
        uploaded_name = None
        options = behavior_model_options()
        allowed_backends = {option["value"] for option in options}
        default_backend = app.config.get("MULTIMODAL_MODEL_BACKEND", "mmrt_multimodal_full_v2_t4x2")
        backend = request.form.get("behavior_model_backend") or request.args.get("behavior_model_backend") or default_backend
        backend = str(backend).strip().lower()
        if backend not in allowed_backends:
            backend = default_backend if default_backend in allowed_backends else "mmrt_multimodal_full_v2_t4x2"
        if _is_mmrt_backend(backend):
            model_path = Path(app.config["MMRT_V2_MODEL_PATH"])
        else:
            model_path = Path(app.config["ACTIVITY7_MODEL_PACKAGE_PATH"])
        active_driver = None
        user = getattr(g, "user", None)
        if user is not None:
            active_driver_id = session.get("active_driver_id") or request.cookies.get("active_driver_id")
            if active_driver_id:
                active_driver = get_db().execute(
                    """
                    SELECT id, display_name, description, status
                    FROM drivers
                    WHERE id = ? AND user_id = ?
                    """,
                    (active_driver_id, user["id"]),
                ).fetchone()
        if request.method == "POST":
            uploaded_file = request.files.get("video")
            if uploaded_file is None or uploaded_file.filename == "":
                flash(translate("flash.choose_video", "Choose a video file before uploading."), "error")
            elif not _allowed_video(uploaded_file.filename, app.config["ALLOWED_VIDEO_EXTENSIONS"]):
                flash(translate("flash.unsupported_video", "Unsupported video type. Use mp4, avi, mov, or mkv."), "error")
            else:
                safe_name = secure_filename(uploaded_file.filename)
                if not safe_name:
                    flash(translate("flash.invalid_filename", "The uploaded file name is not valid."), "error")
                else:
                    test_dir = Path(app.config["UPLOAD_FOLDER"]) / "multimodal_tests"
                    test_dir.mkdir(parents=True, exist_ok=True)
                    stored_name = f"{uuid4().hex}_{safe_name}"
                    saved_path = test_dir / stored_name
                    uploaded_file.save(saved_path)
                    try:
                        if _is_mmrt_backend(backend):
                            clip_seconds = float(app.config.get("MMRT_V2_CLIP_SECONDS", 3.0))
                            stride_seconds = float(app.config.get("MMRT_V2_STRIDE_SECONDS", 1.5))
                            frame_count = int(app.config.get("MMRT_V2_FRAME_COUNT", 16))
                        else:
                            clip_seconds = float(app.config.get("MULTIMODAL_CLIP_SECONDS", 4.0))
                            stride_seconds = float(app.config.get("MULTIMODAL_CLIP_STRIDE_SECONDS", 2.0))
                            frame_count = int(app.config.get("MULTIMODAL_FRAME_COUNT", 16))
                        metadata, predictions, top_prediction = _infer_model(
                            model_path,
                            saved_path,
                            clip_seconds,
                            stride_seconds,
                            frame_count,
                            app.config.get("MMRT_V2_DEVICE", app.config.get("MULTIMODAL_DEVICE", "cpu")),
                            app.config.get("ACTIVITY7_DECISION_MODE", "final"),
                        )
                        result = {
                            "video_path": str(saved_path),
                            "original_filename": uploaded_file.filename,
                            "model_path": str(model_path),
                            "metadata": metadata,
                            "predictions": predictions,
                            "top_prediction": top_prediction or {
                                "label": "normal_driving",
                                "confidence": 0.0,
                                "audio_track_detected": False,
                                "sampled_frame_count": 0,
                                "pose_frame_count": 0,
                                "top3": [],
                                "baseline_label": "normal_driving",
                                "baseline_confidence": 0.0,
                                "geom2_label": "normal_driving",
                                "geom2_confidence": 0.0,
                            },
                            "modalities": {
                                "video": True,
                                "audio": bool(top_prediction["audio_track_detected"]) if top_prediction else False,
                                "skeleton_frames": int(top_prediction["pose_frame_count"]) if top_prediction else 0,
                                "clip_count": len(predictions),
                            },
                        }
                        uploaded_name = uploaded_file.filename
                        flash(translate("flash.multimodal_done", "Multimodal video-audio-skeleton analysis completed."), "success")
                    except Exception as exc:
                        flash(f"{translate('flash.multimodal_failed', 'Multimodal analysis failed')}: {exc}", "error")

        return render_template(
            template_name,
            result=result,
            uploaded_name=uploaded_name,
            model_path=model_path,
            demo_model_name=app.config.get("MODEL_DEMO_NAME", ""),
            decision_mode=app.config.get("ACTIVITY7_DECISION_MODE", "final"),
            behavior_backend=backend,
            behavior_model_options=options,
            current_language=g.get("language", "en"),
            active_driver=active_driver,
        )
