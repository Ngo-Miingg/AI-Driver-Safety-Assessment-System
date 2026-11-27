"""Start the rebuilt Driver Safety v1 app on a non-conflicting local port."""

from app import app
from mddad import socketio


if __name__ == "__main__":
    socketio.run(
        app,
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
