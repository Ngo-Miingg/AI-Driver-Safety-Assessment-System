from pathlib import Path
import base64
import json
import time

import cv2
from flask import abort, current_app, flash, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

from .auth import login_required
from .database import get_db
from .driver_safety_v1 import model_readiness
from .face_identity import average_embeddings, embedding_consistency, embedding_to_json, get_face_identity_verifier
from .i18n import translate
from .repositories import (
    create_audit_log,
    create_media_asset,
    deactivate_driver_biometric,
    upsert_driver_biometric,
)


ALLOWED_FACE_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FACE_ENROLLMENT_SAMPLES = 12
_FACE_DETECTOR_CACHE = {}


def _current_user_id():
    from flask import g

    return g.user["id"]


def _get_owned_driver(driver_id):
    driver = get_db().execute(
        """
        SELECT id, user_id, display_name, description, notes, status,
               face_embedding_json, face_enrolled_at, face_enrollment_image_path,
               created_at, updated_at
        FROM drivers
        WHERE id = ? AND user_id = ?
        """,
        (driver_id, _current_user_id()),
    ).fetchone()
    if driver is None:
        abort(404)
    return driver


def _allowed_face_image(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_FACE_IMAGE_EXTENSIONS


def _decode_captured_face_samples(payload):
    if not payload:
        return []
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    samples = []
    for index, item in enumerate(parsed[:MAX_FACE_ENROLLMENT_SAMPLES], start=1):
        if not isinstance(item, dict):
            continue
        data_url = str(item.get("data_url") or "")
        if "," in data_url:
            _, encoded = data_url.split(",", 1)
        else:
            encoded = data_url
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except Exception:
            continue
        if not image_bytes:
            continue
        samples.append(
            {
                "filename": f"camera_sample_{index:02d}.jpg",
                "mime_type": "image/jpeg",
                "bytes": image_bytes,
                "source": "camera_capture",
                "pose_hint": item.get("pose_hint"),
            }
        )
    return samples


def _read_frame_from_saved_image(saved_path):
    return cv2.imread(str(saved_path))


def _detect_largest_face(frame):
    from ultralytics import YOLO
    from .driver_safety_v1 import DriverSafetyV1Runtime

    model_path = str(Path(current_app.config["DRIVER_SAFETY_FACE_DETECTOR_PATH"]).resolve())
    detector = _FACE_DETECTOR_CACHE.get(model_path)
    if detector is None:
        detector = YOLO(model_path)
        _FACE_DETECTOR_CACHE[model_path] = detector
    result = detector(
        frame,
        verbose=False,
        conf=float(current_app.config["DRIVER_SAFETY_FACE_CONFIDENCE"]),
        imgsz=int(current_app.config["DRIVER_SAFETY_FACE_IMGSZ"]),
    )[0]
    return DriverSafetyV1Runtime._largest_face(result)


def _face_box_quality(face_box, frame_shape):
    if face_box is None or frame_shape is None:
        return {"face_area_ratio": 0.0}
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(value) for value in face_box]
    face_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    frame_area = max(width * height, 1)
    return {"face_area_ratio": round(float(face_area / frame_area), 6)}


def register_driver_routes(app):
    @app.get("/drivers")
    @login_required
    def drivers():
        rows = get_db().execute(
            """
            SELECT id, display_name, description, status, face_enrolled_at, created_at
            FROM drivers
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (_current_user_id(),),
        ).fetchall()
        return render_template("drivers.html", drivers=rows)

    @app.route("/drivers/new", methods=("GET", "POST"))
    @login_required
    def new_driver():
        if request.method == "POST":
            display_name = request.form.get("display_name", "").strip()
            description = request.form.get("description", "").strip()
            notes = request.form.get("notes", "").strip()

            if not display_name:
                flash(translate("flash.driver_name_required", "Driver name is required."), "error")
                return render_template(
                    "driver_form.html",
                    form_title="Create Driver",
                    driver=None,
                )

            cursor = get_db().execute(
                """
                INSERT INTO drivers (user_id, display_name, description, notes)
                VALUES (?, ?, ?, ?)
                """,
                (_current_user_id(), display_name, description or None, notes or None),
            )
            get_db().commit()
            flash(translate("flash.driver_created", "Driver profile created."), "success")
            return redirect(url_for("driver_detail", driver_id=cursor.lastrowid))

        return render_template("driver_form.html", form_title="Create Driver", driver=None)

    @app.get("/drivers/<int:driver_id>")
    @login_required
    def driver_detail(driver_id):
        driver = _get_owned_driver(driver_id)
        summary = get_db().execute(
            """
            SELECT
                COUNT(*) AS total_sessions,
                AVG(final_safety_score) AS average_safety
            FROM sessions
            WHERE driver_id = ? AND user_id = ?
            """,
            (driver_id, _current_user_id()),
        ).fetchone()
        readiness = model_readiness(current_app)
        verifier_status = {
            "ready": readiness["identity_ready"],
            "threshold": readiness["identity_threshold"],
            "error": readiness["identity_runtime_error"],
        }
        return render_template(
            "driver_detail.html",
            driver=driver,
            summary=summary,
            verifier_status=verifier_status,
        )

    @app.route("/drivers/<int:driver_id>/edit", methods=("GET", "POST"))
    @login_required
    def edit_driver(driver_id):
        driver = _get_owned_driver(driver_id)

        if request.method == "POST":
            display_name = request.form.get("display_name", "").strip()
            description = request.form.get("description", "").strip()
            notes = request.form.get("notes", "").strip()
            status = request.form.get("status", "active")

            if not display_name:
                flash(translate("flash.driver_name_required", "Driver name is required."), "error")
                return render_template(
                    "driver_form.html",
                    form_title="Edit Driver",
                    driver=driver,
                )

            if status not in {"active", "inactive"}:
                status = "active"

            get_db().execute(
                """
                UPDATE drivers
                SET display_name = ?, description = ?, notes = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (
                    display_name,
                    description or None,
                    notes or None,
                    status,
                    driver_id,
                    _current_user_id(),
                ),
            )
            get_db().commit()
            if session.get("active_driver_id") == driver_id:
                session["active_driver_name"] = display_name
            flash(translate("flash.driver_updated", "Driver profile updated."), "success")
            return redirect(url_for("driver_detail", driver_id=driver_id))

        return render_template("driver_form.html", form_title="Edit Driver", driver=driver)

    @app.post("/drivers/<int:driver_id>/select")
    @login_required
    def select_driver(driver_id):
        driver = _get_owned_driver(driver_id)
        session["active_driver_id"] = driver["id"]
        session["active_driver_name"] = driver["display_name"]
        flash(f"{translate('flash.selected_driver', 'Selected driver')}: {driver['display_name']}", "success")
        return redirect(request.referrer or url_for("dashboard"))

    @app.post("/drivers/<int:driver_id>/face-enrollment")
    @login_required
    def enroll_driver_face(driver_id):
        _get_owned_driver(driver_id)
        camera_images = [
            image
            for image in request.files.getlist("face_samples")
            if image is not None and image.filename
        ]
        images = [
            image
            for image in request.files.getlist("face_images")
            if image is not None and image.filename
        ]
        if camera_images:
            images = [*camera_images, *images]
        if not images:
            legacy_image = request.files.get("face_image")
            if legacy_image is not None and legacy_image.filename:
                images = [legacy_image]
        captured_samples = _decode_captured_face_samples(request.form.get("face_samples_json"))
        if not images and not captured_samples:
            flash(translate("flash.face_image_required", "Choose a face image before enrolling."), "error")
            return redirect(url_for("driver_detail", driver_id=driver_id))
        invalid_names = [image.filename for image in images if not _allowed_face_image(image.filename)]
        if invalid_names:
            flash(translate("flash.face_image_invalid", "Use JPG, PNG, or WEBP face images."), "error")
            return redirect(url_for("driver_detail", driver_id=driver_id))

        verifier = get_face_identity_verifier(current_app)
        if not verifier.ready:
            flash(verifier.error or translate("flash.face_verifier_unavailable", "Face verifier is unavailable."), "error")
            return redirect(url_for("driver_detail", driver_id=driver_id))

        enrollment_dir = Path(current_app.config["DRIVER_SAFETY_FACE_ENROLLMENT_DIR"])
        enrollment_dir.mkdir(parents=True, exist_ok=True)
        embeddings = []
        sample_assets = []
        rejected = []
        first_valid_path = None
        timestamp_ms = int(time.time() * 1000)

        upload_sources = [
            {
                "filename": image.filename,
                "mime_type": image.mimetype,
                "file": image,
                "source": "file_upload",
                "pose_hint": None,
            }
            for image in images
        ]
        sample_sources = [*captured_samples, *upload_sources][:MAX_FACE_ENROLLMENT_SAMPLES]

        for index, sample in enumerate(sample_sources, start=1):
            safe_name = secure_filename(sample["filename"])
            stored_name = f"driver_{driver_id}_{timestamp_ms}_{index}_{safe_name}"
            saved_path = enrollment_dir / stored_name
            if "bytes" in sample:
                saved_path.write_bytes(sample["bytes"])
            else:
                sample["file"].save(saved_path)

            frame = _read_frame_from_saved_image(saved_path)
            if frame is None:
                saved_path.unlink(missing_ok=True)
                rejected.append({"filename": sample["filename"], "reason": "unreadable"})
                continue
            try:
                face_box = _detect_largest_face(frame)
                if face_box is None:
                    raise ValueError("no_face")
                embedding = verifier.embed_frame(frame, face_box)
            except Exception as exc:
                saved_path.unlink(missing_ok=True)
                rejected.append({"filename": sample["filename"], "reason": str(exc)})
                continue

            media_asset_id = create_media_asset(
                user_id=_current_user_id(),
                driver_id=driver_id,
                asset_type="face_enrollment_image",
                storage_path=saved_path,
                original_filename=sample["filename"],
                mime_type=sample["mime_type"],
                file_size_bytes=saved_path.stat().st_size if saved_path.exists() else None,
                width=int(frame.shape[1]),
                height=int(frame.shape[0]),
                metadata={
                    "face_box": list(face_box),
                    "source": sample["source"],
                    "sample_index": index,
                    "pose_hint": sample.get("pose_hint"),
                    **_face_box_quality(face_box, frame.shape),
                },
            )
            sample_assets.append(media_asset_id)
            embeddings.append(embedding)
            if first_valid_path is None:
                first_valid_path = saved_path

        if not embeddings:
            get_db().rollback()
            flash(translate("flash.face_enrollment_no_valid_samples", "No uploaded image produced a valid face embedding."), "error")
            return redirect(url_for("driver_detail", driver_id=driver_id))

        embedding = average_embeddings(embeddings)
        quality_summary = embedding_consistency(embeddings)
        get_db().execute(
            """
            UPDATE drivers
            SET face_embedding_json = ?,
                face_enrolled_at = CURRENT_TIMESTAMP,
                face_enrollment_image_path = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (embedding_to_json(embedding), str(first_valid_path), driver_id, _current_user_id()),
        )
        biometric_id = upsert_driver_biometric(
            user_id=_current_user_id(),
            driver_id=driver_id,
            biometric_type="face_embedding",
            model_backend="driver_safety_v1.face_verification",
            embedding_json=embedding_to_json(embedding),
            enrollment_asset_id=sample_assets[0],
            threshold=float(current_app.config.get("DRIVER_SAFETY_FACE_VERIFY_THRESHOLD", 0.94)),
        )
        create_audit_log(
            user_id=_current_user_id(),
            driver_id=driver_id,
            action="driver.face_enrolled",
            entity_type="driver_biometric",
            entity_id=biometric_id,
            metadata={
                "model_backend": "driver_safety_v1.face_verification",
                "sample_asset_ids": sample_assets,
                "accepted_samples": len(embeddings),
                "rejected_samples": rejected,
                **quality_summary,
            },
        )
        get_db().commit()
        message = translate("flash.face_enrolled", "Face enrollment saved for this driver.")
        if len(embeddings) > 1:
            message = f"{message} {translate('flash.face_enrolled_samples', 'Valid samples averaged')}: {len(embeddings)}."
        if rejected:
            message = f"{message} {translate('flash.face_enrolled_skipped', 'Skipped images')}: {len(rejected)}."
        flash(message, "success")
        return redirect(url_for("driver_detail", driver_id=driver_id))

    @app.post("/drivers/<int:driver_id>/face-enrollment/delete")
    @login_required
    def delete_driver_face_enrollment(driver_id):
        driver = _get_owned_driver(driver_id)
        get_db().execute(
            """
            UPDATE drivers
            SET face_embedding_json = NULL,
                face_enrolled_at = NULL,
                face_enrollment_image_path = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
            """,
            (driver_id, _current_user_id()),
        )
        deactivate_driver_biometric(
            driver_id,
            "face_embedding",
            "driver_safety_v1.face_verification",
        )
        create_audit_log(
            user_id=_current_user_id(),
            driver_id=driver_id,
            action="driver.face_enrollment_deleted",
            entity_type="driver",
            entity_id=driver_id,
        )
        get_db().commit()
        image_path = driver["face_enrollment_image_path"]
        if image_path:
            try:
                Path(image_path).unlink(missing_ok=True)
            except OSError:
                pass
        flash(translate("flash.face_enrollment_deleted", "Face enrollment removed."), "success")
        return redirect(url_for("driver_detail", driver_id=driver_id))
