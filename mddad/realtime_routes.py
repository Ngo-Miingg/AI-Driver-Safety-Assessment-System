from __future__ import annotations

import time

import cv2
from flask import Response, current_app, g, jsonify, render_template, session, stream_with_context

from .auth import login_required
from .database import get_db
from .model_options import behavior_model_options, phone_detector_options
from .realtime_monitor import realtime_manager


def _get_active_driver():
    active_driver_id = session.get("active_driver_id")
    if not active_driver_id:
        return None

    return get_db().execute(
        """
        SELECT id, display_name, description, status
        FROM drivers
        WHERE id = ? AND user_id = ?
        """,
        (active_driver_id, g.user["id"]),
    ).fetchone()


def _generate_mjpeg_frames(camera_index: int, user_id: int):
    cap = None
    try:
        while True:
            if realtime_manager.has_active_session_for_user(user_id):
                if cap is not None:
                    cap.release()
                    cap = None
                payload = realtime_manager.get_latest_preview_frame(user_id)
                if payload:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                    )
                time.sleep(0.05)
                continue

            if cap is None:
                cap = cv2.VideoCapture(int(camera_index), getattr(cv2, "CAP_DSHOW", 0))
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    time.sleep(0.2)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue

            payload = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
            )
            time.sleep(0.03)
    finally:
        if cap is not None:
            cap.release()


def _debug_camera_once(camera_index):
    capture = cv2.VideoCapture(int(camera_index), getattr(cv2, "CAP_DSHOW", 0))
    opened = capture.isOpened()
    frame_read = False
    frame_shape = None
    if opened:
        frame_read, frame = capture.read()
        if frame_read and frame is not None:
            frame_shape = list(frame.shape)
    capture.release()
    return {
        "index": int(camera_index),
        "opened": opened,
        "frame_read": frame_read,
        "frame_shape": frame_shape,
    }


def register_realtime_routes(app):
    @app.get("/realtime")
    @login_required
    def realtime():
        active_driver = _get_active_driver()
        return render_template(
            "realtime.html",
            active_driver=active_driver,
            realtime_camera_index=app.config.get("REALTIME_CAMERA_INDEX", 0),
            behavior_model_options=behavior_model_options(),
            phone_detector_options=phone_detector_options(app, include_disabled=True),
            selected_behavior_backend=app.config.get("REALTIME_BEHAVIOR_MODEL_BACKEND", "mmrt_multimodal_full_v2_t4x2"),
            selected_phone_backend="disabled",
        )

    @app.get("/realtime/feed")
    @login_required
    def realtime_feed():
        return Response(
            stream_with_context(_generate_mjpeg_frames(app.config.get("REALTIME_CAMERA_INDEX", 0), g.user["id"])),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/debug/camera")
    @login_required
    def debug_camera():
        configured_index = int(app.config.get("REALTIME_CAMERA_INDEX", 0))
        results = [_debug_camera_once(configured_index)]
        if not results[0]["opened"]:
            for fallback in (0, 1, 2):
                if fallback == configured_index:
                    continue
                results.append(_debug_camera_once(fallback))
        return jsonify({"configured_index": configured_index, "results": results})
