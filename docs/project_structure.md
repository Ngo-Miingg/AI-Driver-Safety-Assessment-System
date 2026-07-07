# Project Structure

```text
D:\Work\HocKy3\ComputerVision\MDDAD
|-- app.py
|-- config.py
|-- .env.example
|-- requirements.txt
|-- mddad
|   |-- __init__.py
|   |-- analysis_models.py
|   |-- analysis_routes.py
|   |-- auth.py
|   |-- dashboard_routes.py
|   |-- database.py
|   |-- driver_routes.py
|   |-- history_routes.py
|   |-- logging_config.py
|   |-- mediapipe_analysis.py
|   |-- realtime_monitor.py
|   |-- realtime_routes.py
|   |-- report_generator.py
|   |-- risk_engine.py
|   |-- evaluation.py
|   |-- settings.py
|   |-- settings_routes.py
|   |-- sockets.py
|   |-- upload_routes.py
|   |-- video_processing.py
|   |-- yolo_phone_analysis.py
|   `-- ai
|       |-- face_analyzer.py
|       |-- yolo_detector.py
|       `-- detectors
|           |-- base.py
|           |-- factory.py
|           `-- ultralytics_detector.py
|-- static
|   |-- css
|   |   `-- main.css
|   `-- js
|       |-- main.js
|       `-- realtime.js
|-- templates
|   |-- base.html
|   |-- dashboard.html
|   |-- history.html
|   |-- realtime.html
|   |-- reports.html
|   |-- evaluation.html
|   |-- settings.html
|   |-- session_detail.html
|   `-- ...
|-- models
|-- uploads
|   `-- evidence
|-- logs
|-- instance
`-- docs
```

## Current Responsibilities

- `mddad/realtime_routes.py`: protected realtime monitoring page and active-driver lookup.
- `mddad/realtime_monitor.py`: OpenCV webcam loop, MediaPipe frame analysis, optional low-rate realtime YOLO phone detection, SocketIO emissions, and final realtime session persistence.
- `mddad/sockets.py`: SocketIO connection, ping, realtime start/stop/disconnect handlers.
- `static/js/realtime.js`: realtime controls, button states, live metrics, behavior indicators, timeline updates, saved-session link, and optional low-rate JPEG snapshot rendering.
- `templates/realtime.html`: demo-ready realtime monitoring UI with optional low-rate snapshot preview.
- `mddad/dashboard_routes.py`: dashboard metrics, recent sessions, active-driver data, and reports listing.
- `templates/dashboard.html`: real SQLite-backed dashboard cards and recent-session table.
- `mddad/history_routes.py`: session history query with scores, events, status, driver, and report status.
- `templates/history.html`: demo-friendly history table for uploaded and realtime sessions.
- `mddad/upload_routes.py`: upload workflow, session detail data query, and OpenCV metadata analysis.
- `templates/session_detail.html`: presentation-ready session summary, metrics, event timeline, risk breakdown, and report display.
- `uploads/evidence/`: small optional JPEG event evidence snapshots from realtime monitoring.
- `templates/reports.html`: basic reports page with optional driver filter.
- `mddad/settings.py`: default/SQLite-backed calibration settings and validation.
- `mddad/settings_routes.py`: protected Settings, Evaluation, and evaluation CSV export routes.
- `mddad/evaluation.py`: session-level evaluation query, derived metrics, and CSV row helpers.
- `templates/settings.html`: editable calibration/settings UI.
- `templates/evaluation.html`: session evaluation table and CSV export entry point.
- `mddad/analysis_routes.py`: uploaded-video simulated, MediaPipe, and Full AI analysis.
- `mddad/ai/detectors/`: detector plug-in scaffold for `mediapipe_only`, `yolo_pretrained`, `yolo_finetuned`, and `rtdetr_finetuned`.
- `mddad/logging_config.py`: rotating file logging setup under `logs/app.log`.
- `.env.example`: centralized local/offline configuration template.

## Implemented

- Authentication and driver management.
- Uploaded video analysis with OpenCV metadata extraction.
- Deterministic simulated analysis fallback.
- Uploaded-video MediaPipe analysis.
- Uploaded-video Full AI Analysis with MediaPipe plus optional YOLO phone detection.
- Plug-and-play phone detector backend scaffold for pretrained YOLO, fine-tuned YOLO, and fine-tuned RT-DETR.
- Realtime webcam monitoring with OpenCV and MediaPipe.
- Optional realtime YOLO phone detection at low sample FPS, disabled by default.
- SocketIO live updates for status, elapsed time, risk, safety estimate, alert level, active states, and events.
- Realtime session persistence with events, risk logs, final scores, fatigue trend, and report.
- Dashboard backed by SQLite data.
- History, session detail, and reports pages suitable for defense demo.
- Protected Settings page for threshold calibration.
- Protected Evaluation page with session-level metrics.
- Evaluation CSV export.
- Optional low-rate realtime JPEG snapshot preview through SocketIO.
- Optional small event evidence snapshots for confirmed realtime events.

## Not Implemented Yet

- Realtime YOLO training or custom realtime model optimization.
- YOLO training or fine-tuning.
- RT-DETR training or validation with a real RT-DETR phone model file.
- Uploaded-video event evidence snapshots.
- Full video streaming.
- Production-grade 3D head pose.
- Model-trained drowsiness classification.
