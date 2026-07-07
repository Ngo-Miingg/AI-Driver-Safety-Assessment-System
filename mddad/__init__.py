from flask import Flask
from flask_socketio import SocketIO

from config import get_config


socketio = SocketIO(cors_allowed_origins="*", async_mode="threading")


def create_app(config_object=None):
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_object(config_object or get_config())

    from .logging_config import configure_logging

    configure_logging(app)

    from .analysis_routes import register_analysis_routes
    from .auth import register_auth
    from .database import init_app as init_database
    from .dashboard_routes import register_dashboard_routes
    from .driver_routes import register_driver_routes
    from .history_routes import register_history_routes
    from .i18n import register_i18n
    from .multimodal_test_routes import register_multimodal_test_routes
    from .mmrt_v2_demo_routes import register_mmrt_v2_demo_routes
    from .routes import register_routes
    from .realtime_routes import register_realtime_routes
    from .sockets import register_socket_events
    from .settings_routes import register_evaluation_routes
    from .upload_routes import register_upload_routes

    register_i18n(app)
    init_database(app)
    register_auth(app)
    register_routes(app)
    register_dashboard_routes(app)
    register_driver_routes(app)
    register_upload_routes(app)
    register_multimodal_test_routes(app)
    register_mmrt_v2_demo_routes(app)
    register_realtime_routes(app)
    register_history_routes(app)
    register_analysis_routes(app)
    register_evaluation_routes(app)
    socketio.init_app(app, async_mode=app.config["SOCKETIO_ASYNC_MODE"])
    register_socket_events(socketio)

    return app
