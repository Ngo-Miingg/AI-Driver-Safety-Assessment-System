from __future__ import annotations

import os
import sys
from pathlib import Path


def _maybe_reexec_into_workspace_venv():
    current_exe = Path(sys.executable).resolve()
    if current_exe.parent.name.lower() == "scripts" and current_exe.parent.parent.name.lower() == ".venv":
        return
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".venv" / "Scripts" / "python.exe"
        if not candidate.exists():
            continue
        candidate = candidate.resolve()
        if candidate == current_exe:
            return
        os.execv(str(candidate), [str(candidate), __file__, *sys.argv[1:]])


_maybe_reexec_into_workspace_venv()

import cv2

from mddad import create_app, socketio
from config import get_config


class Demo7ClassConfig(get_config()):
    ACTIVITY7_MODEL_PACKAGE_PATH = str(Path(__file__).resolve().parent / "models" / "activity7_final_best_model_package.zip")
    MULTIMODAL_MODEL_PATH = ACTIVITY7_MODEL_PACKAGE_PATH
    ACTIVITY7_DECISION_MODE = "final"
    REALTIME_BEHAVIOR_MODEL_BACKEND = "activity7_multimodal_live"
    REALTIME_PHONE_DETECTOR_BACKEND = "disabled"
    MODEL_DEMO_NAME = "Activity7 SPECIALIST1 final demo"


app = create_app(Demo7ClassConfig)

print(
    "[demo-7class] python=",
    sys.executable,
    "cv2=",
    getattr(cv2, "__file__", None),
    "has_videocapture=",
    hasattr(cv2, "VideoCapture"),
    flush=True,
)


if __name__ == "__main__":
    socketio.run(
        app,
        host="127.0.0.1",
        port=5003,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
