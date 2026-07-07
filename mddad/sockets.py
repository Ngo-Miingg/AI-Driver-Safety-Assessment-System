from __future__ import annotations

from flask import current_app, request, session

from .database import get_db
from .model_options import supported_behavior_backends, supported_phone_backends
from .realtime_monitor import realtime_manager


def register_socket_events(socketio):
    @socketio.on("connect")
    def handle_connect():
        socketio.emit("server_status", {"status": "connected"}, to=request.sid)

    @socketio.on("ping_server")
    def handle_ping(message=None):
        socketio.emit("server_status", {"status": "ok", "echo": message}, to=request.sid)

    @socketio.on("set_realtime_language")
    def handle_set_realtime_language(payload=None):
        payload = payload or {}
        language = str(payload.get("language") or session.get("language") or "en").strip().lower()
        if language not in {"en", "vi"}:
            language = "en"
        realtime_manager.set_language(request.sid, language)

    @socketio.on("realtime_audio_chunk")
    def handle_realtime_audio_chunk(payload=None):
        payload = payload or {}
        samples = payload.get("samples")
        sample_rate = payload.get("sample_rate")
        if samples is None:
            return
        realtime_manager.push_audio_chunk(request.sid, samples, sample_rate)

    @socketio.on("disconnect")
    def handle_disconnect():
        realtime_manager.discard(request.sid)

    @socketio.on("start_realtime_session")
    def handle_start_realtime_session(payload=None):
        payload = payload or {}
        user_id = session.get("user_id")
        if not user_id:
            socketio.emit(
                "realtime_error",
                {"message": "Bạn cần đăng nhập trước khi giám sát realtime."},
                to=request.sid,
            )
            return

        driver_id = session.get("active_driver_id")
        if not driver_id:
            socketio.emit(
                "realtime_error",
                {"message": "Hãy chọn một tài xế đang hoạt động trước khi giám sát."},
                to=request.sid,
            )
            return

        db = get_db()
        driver = db.execute(
            """
            SELECT id
            FROM drivers
            WHERE id = ? AND user_id = ? AND status = 'active'
            """,
            (driver_id, user_id),
        ).fetchone()
        if driver is None:
            socketio.emit(
                "realtime_error",
                {"message": "Tài xế đã chọn không còn khả dụng."},
                to=request.sid,
            )
            return

        behavior_backend = str(payload.get("behavior_model_backend") or "mmrt_multimodal_full_v2_t4x2").strip().lower()
        phone_backend = str(payload.get("phone_detector_backend") or "disabled").strip().lower()
        if behavior_backend.startswith("mmrt"):
            phone_backend = "disabled"
        language = str(payload.get("language") or session.get("language") or "en").strip().lower()
        if language not in {"en", "vi"}:
            language = "en"

        if behavior_backend not in supported_behavior_backends():
            socketio.emit(
                "realtime_error",
                {"message": f"Backend mô hình hành vi không được hỗ trợ: {behavior_backend}."},
                to=request.sid,
            )
            return

        if phone_backend not in supported_phone_backends(include_disabled=True):
            socketio.emit(
                "realtime_error",
                {"message": f"Backend phát hiện điện thoại không được hỗ trợ: {phone_backend}."},
                to=request.sid,
            )
            return

        started, error = realtime_manager.start(
            current_app._get_current_object(),
            socketio,
            request.sid,
            user_id,
            driver_id,
            {
                "behavior_model_backend": behavior_backend,
                "phone_detector_backend": phone_backend,
                "language": language,
            },
        )
        if not started:
            socketio.emit(
                "realtime_error",
                {"message": error or "Không thể bắt đầu giám sát realtime."},
                to=request.sid,
            )
            return

        socketio.emit("realtime_status", {"status": "starting"}, to=request.sid)

    @socketio.on("stop_realtime_session")
    def handle_stop_realtime_session():
        if realtime_manager.stop(request.sid):
            socketio.emit("realtime_status", {"status": "stopping"}, to=request.sid)
        else:
            socketio.emit(
                "realtime_error",
                {"message": "Không tìm thấy phiên realtime đang hoạt động cho trình duyệt này."},
                to=request.sid,
            )
