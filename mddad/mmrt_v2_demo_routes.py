from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

from flask import current_app, flash, g, render_template, request
from werkzeug.utils import secure_filename

from .auth import login_required
from .i18n import translate
from .mmrt_v2_infer import MMRTV2Package, clip_result_to_dict, save_mmrt_demo_run
from .video_processing import extract_video_metadata


logger = logging.getLogger(__name__)


def _allowed_video(filename, allowed_extensions):
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in allowed_extensions


def _current_user_id():
    user = getattr(g, "user", None)
    if user is None:
        return None
    try:
        return user["id"]
    except Exception:
        return getattr(user, "id", None)


def register_mmrt_v2_demo_routes(app):
    @app.route("/mmrt-v2-demo", methods=("GET", "POST"))
    @login_required
    def mmrt_v2_demo():
        result = None
        uploaded_name = None
        package_path = Path(app.config["MMRT_V2_MODEL_PATH"])
        log_path = Path(app.config["MMRT_V2_RUN_LOG_DIR"])

        if request.method == "POST":
            uploaded_file = request.files.get("video")
            if uploaded_file is None or uploaded_file.filename == "":
                flash(translate("flash.choose_video", "Choose a video file before uploading."), "error")
            elif not _allowed_video(uploaded_file.filename, app.config["ALLOWED_VIDEO_EXTENSIONS"]):
                flash(translate("flash.unsupported_video", "Unsupported video type. Use mp4, avi, mov, or mkv."), "error")
            else:
                safe_name = secure_filename(uploaded_file.filename)
                test_dir = Path(app.config["UPLOAD_FOLDER"]) / "mmrt_v2_tests"
                test_dir.mkdir(parents=True, exist_ok=True)
                stored_name = f"{uuid4().hex}_{safe_name}"
                saved_path = test_dir / stored_name
                uploaded_file.save(saved_path)
                uploaded_name = uploaded_file.filename
                logger.info("MMRT v2 demo upload saved path=%s user=%s", saved_path, _current_user_id())
                try:
                    package = MMRTV2Package(package_path, device=app.config.get("MMRT_V2_DEVICE", "cpu"))
                    try:
                        meta, clips, top = package.predict_video(
                            saved_path,
                            clip_seconds=float(app.config.get("MMRT_V2_CLIP_SECONDS", 3.0)),
                            stride_seconds=float(app.config.get("MMRT_V2_STRIDE_SECONDS", 1.5)),
                            frame_count=int(app.config.get("MMRT_V2_FRAME_COUNT", 16)),
                        )
                    finally:
                        package.close()

                    source_meta = extract_video_metadata(saved_path)
                    run_payload = {
                        "video_path": str(saved_path),
                        "original_filename": uploaded_file.filename,
                        "package_path": str(package_path),
                        "source_metadata": {
                            "fps": getattr(source_meta, "fps", 0.0),
                            "frame_count": getattr(source_meta, "frame_count", 0),
                            "duration_seconds": getattr(source_meta, "duration_seconds", 0.0),
                            "width": getattr(source_meta, "width", 0),
                            "height": getattr(source_meta, "height", 0),
                        },
                        "model_metadata": meta,
                        "top_prediction": clip_result_to_dict(top) if top else None,
                        "clip_predictions": [clip_result_to_dict(item) for item in clips],
                    }
                    run_id, run_file = save_mmrt_demo_run(log_path, run_payload)
                    logger.info("MMRT v2 demo completed run_id=%s log=%s top=%s", run_id, run_file, top.label if top else None)
                    result = {
                        "run_id": run_id,
                        "run_file": str(run_file),
                        "video_path": str(saved_path),
                        "top_prediction": clip_result_to_dict(top) if top else None,
                        "clips": [clip_result_to_dict(item) for item in clips],
                        "model_metadata": meta,
                    }
                    flash("MMRT v2 demo inference completed.", "success")
                except Exception as exc:
                    logger.exception("MMRT v2 demo failed for file=%s", saved_path)
                    flash(f"MMRT v2 demo failed: {exc}", "error")

        return render_template(
            "mmrt_v2_demo.html",
            result=result,
            uploaded_name=uploaded_name,
            model_path=package_path,
            current_language=g.get("language", "en"),
        )
