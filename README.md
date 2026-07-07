# AI Driver Safety Assessment System

Flask-based driver monitoring and multimodal behavior analysis system for cabin videos and webcam sessions. The project combines:

- realtime webcam monitoring with Flask-SocketIO
- uploaded video analysis
- driver/session management with SQLite
- phone detection backends (YOLO / RT-DETR)
- multimodal behavior inference demos, including MMRT v2 and Activity7 comparison runners

This repository is the cleaned source-code version of the project. Large local artifacts such as datasets, trained weights, logs, uploads, and virtual environments are intentionally excluded from Git.

## Main capabilities

- User login, local registration, logout, and English/Vietnamese UI switching
- Driver management with active-driver selection
- Dashboard, history, reports, evaluation, CSV export, and settings pages
- Uploaded-video analysis pipeline with:
  - OpenCV metadata extraction
  - rule-based / landmark-based behavior analysis
  - optional phone detection with fine-tuned YOLO or RT-DETR backends
  - multimodal clip-based model testing
- Realtime webcam monitoring with:
  - Flask-SocketIO updates
  - OpenCV webcam capture
  - MediaPipe face / pose cues
  - live risk, safety, alert, event timeline, and snapshot preview
  - optional phone detector backend selection per session
- Dedicated demo pages for:
  - Activity7 comparison demos
  - MMRT v2 upload demo with per-run JSON logging

## Repository layout

```text
MDDAD/
|-- app.py
|-- config.py
|-- requirements.txt
|-- mddad/                     # Flask app modules, inference helpers, realtime logic
|-- templates/                 # Jinja templates
|-- static/                    # CSS / JS / vendor assets
|-- tools/                     # helper scripts
|-- docs/                      # handoff notes, reports, slide sources
|-- kaggle/                    # training notes / Kaggle workflow docs
|-- tests/                     # test files
|-- run_model_test_simple.py   # general model test runner
|-- run_mddad_demo_geom2.py    # Activity7 GEOM2 demo on port 5002
|-- run_mddad_demo_7class.py   # Activity7 final demo on port 5003
|-- run_mmrt_v2_demo.py        # MMRT v2 upload demo on port 5004
```

## Supported model flows

### 1. Main app

The main app runs on port `5000` and exposes the full driver-safety system:

- dashboard
- drivers
- upload analysis
- realtime monitoring
- history / reports / evaluation
- settings

Current default config points the multimodal upload / realtime behavior backend to:

- `mmrt_multimodal_full_v2_t4x2` for multimodal behavior inference
- fine-tuned RT-DETR / YOLO options for phone detection

### 2. Activity7 comparison demos

These runners are for comparing the older Activity7 package in two decision modes:

- `run_mddad_demo_geom2.py` -> port `5002`
- `run_mddad_demo_7class.py` -> port `5003`

Both expose:

- `http://127.0.0.1:<port>/model-test-simple-cam`

### 3. MMRT v2 demo

`run_mmrt_v2_demo.py` starts a dedicated upload demo for:

- `mmrt_multimodal_full_v2_t4x2`

URL:

- `http://127.0.0.1:5004/mmrt-v2-demo`

This page is focused on clean upload-based testing and writes a JSON log for every inference run.

## Model artifacts expected locally

This repository does **not** commit large trained weights. Keep them locally and point config values to the correct paths.

Common local artifacts used during development:

```text
models/activity7_final_best_model_package.zip
models/phone_rtdetr_l.pt
models/phone_rtdetr_l_light_compare.pt
models/phone_yolo26n.pt
D:/Downloads/mmrt_artifacts_bundle.zip
```

The MMRT demo expects the `mmrt_multimodal_full_v2_t4x2` checkpoint bundle to be available inside:

```text
D:/Downloads/mmrt_artifacts_bundle.zip
```

If you want a different location, override:

```text
MMRT_V2_MODEL_PATH
```

## Tech stack

- Python 3.11
- Flask
- Flask-SocketIO
- SQLite
- OpenCV
- MediaPipe
- PyTorch
- torchvision
- librosa
- Ultralytics
- HTML / CSS / JavaScript

## Requirements

### Python

- Python 3.11 recommended

### OS

- Windows was the primary development environment

### External tools

- `ffmpeg` must be available in `PATH` for MMRT v2 audio extraction
- a working webcam is required for realtime testing

## Installation

```powershell
cd D:\Work\HocKy3\ComputerVision\MDDAD
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment and config

You can copy settings from:

```text
.env.example
```

Important config values live in [config.py](config.py) and include:

- `MULTIMODAL_MODEL_BACKEND`
- `MMRT_V2_MODEL_PATH`
- `ACTIVITY7_MODEL_PACKAGE_PATH`
- `PHONE_DETECTOR_BACKEND`
- `REALTIME_BEHAVIOR_MODEL_BACKEND`
- `REALTIME_PHONE_DETECTOR_BACKEND`
- `REALTIME_TARGET_FPS`
- `REALTIME_SNAPSHOT_FPS`

## Run the full application

```powershell
cd D:\Work\HocKy3\ComputerVision\MDDAD
.\.venv\Scripts\python.exe app.py
```

Open:

```text
http://127.0.0.1:5000
```

Demo account:

```text
username: admin
password: admin123
```

## Run the dedicated demos

### Main model test page

```powershell
.\.venv\Scripts\python.exe run_model_test_simple.py
```

Open:

```text
http://127.0.0.1:5001/model-test-simple
```

### Activity7 GEOM2 demo

```powershell
.\.venv\Scripts\python.exe run_mddad_demo_geom2.py
```

Open:

```text
http://127.0.0.1:5002/model-test-simple-cam
```

### Activity7 final demo

```powershell
.\.venv\Scripts\python.exe run_mddad_demo_7class.py
```

Open:

```text
http://127.0.0.1:5003/model-test-simple-cam
```

### MMRT v2 upload demo

```powershell
.\.venv\Scripts\python.exe run_mmrt_v2_demo.py
```

Open:

```text
http://127.0.0.1:5004/mmrt-v2-demo
```

## Main user flows

### Realtime monitoring

1. Log in
2. Select an active driver
3. Open `/realtime`
4. Choose behavior model and phone detector backend
5. Start monitoring
6. Review live risk, safety, warning level, state, event timeline, and snapshot preview
7. Stop monitoring to save the session and generated report

### Uploaded video analysis

1. Log in and select a driver
2. Open `/upload`
3. Upload a cabin video (`mp4`, `avi`, `mov`, `mkv`)
4. Review metadata
5. Run available analysis modes
6. Inspect session detail, events, report, and evaluation metrics

### MMRT v2 demo logging

Each MMRT v2 demo run stores:

- input file path
- original filename
- source video metadata
- clip-by-clip predictions
- top prediction
- audio / face / skeleton coverage signals

Run logs are written under:

```text
uploads/mmrt_v2_demo_runs/
```

when using the local demo environment.

## Important routes

- `/dashboard`
- `/drivers`
- `/upload`
- `/realtime`
- `/history`
- `/reports`
- `/evaluation`
- `/evaluation.csv`
- `/multimodal-test`
- `/model-test-simple`
- `/model-test-simple-cam`
- `/mmrt-v2-demo`
- `/health`

## Notes on the current architecture

- Realtime monitoring is implemented as a SocketIO-backed background workflow
- Uploaded-video analysis and dedicated demos run synchronously from request handlers
- The project contains both legacy Activity7 demo flows and the newer MMRT v2 upload demo
- Some legacy config names still contain `YOLO` for compatibility with older settings / schema

## Known limitations

- Large model files, datasets, logs, and uploads are not included in this repository
- MMRT v2 upload inference depends on local artifact availability and `ffmpeg`
- Realtime behavior quality depends heavily on camera angle, lighting, audio availability, and model selection
- Some older modules and route names are still kept for comparison / demo continuity

## Suggested next cleanup items

- split legacy comparison demos from the main app into a dedicated `demos/` package
- add automated smoke tests for the key routes and model-demo pages
- add a production-oriented dependency / deployment guide
- normalize remaining mixed-language strings across templates and JS

## Related documentation

- [docs/CHAT_HANDOFF_NEW_CHAT.md](docs/CHAT_HANDOFF_NEW_CHAT.md)
- [models/phone_detection/README.md](models/phone_detection/README.md)
- [docs/requirements.md](docs/requirements.md)

