# System Overview

The Adaptive Multimodal AI Driver Safety Assessment System is a Flask application for local driver safety demos and evaluation.

It supports:

- Authenticated local users.
- Multiple driver profiles.
- Uploaded-video analysis.
- Realtime webcam monitoring.
- MediaPipe drowsiness/distraction signals.
- Optional YOLO/RT-DETR phone detector backends.
- Temporal event tracking.
- Risk and safety scoring.
- Explainable reports.
- History, reports, settings, evaluation, and CSV export.

The system is designed to run without trained phone models and to enable them later by copying model files into `models/` and selecting the detector backend.
