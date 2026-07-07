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

from mddad import create_app, socketio
from config import get_config


class MMRTV2DemoConfig(get_config()):
    MMRT_V2_MODEL_PATH = str(Path("D:/Downloads/mmrt_artifacts_bundle.zip"))
    MODEL_DEMO_NAME = "MMRT v2 t4x2 research demo"


app = create_app(MMRTV2DemoConfig)


if __name__ == "__main__":
    socketio.run(
        app,
        host="127.0.0.1",
        port=5004,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
