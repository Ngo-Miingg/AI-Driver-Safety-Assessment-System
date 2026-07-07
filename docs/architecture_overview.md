# Architecture Overview

## Runtime

```text
Flask app
  -> routes/templates/static assets
  -> SQLite persistence
  -> Flask-SocketIO realtime events
  -> OpenCV video/camera capture
  -> MediaPipe face landmarks
  -> optional Ultralytics detector backend
  -> temporal event generation
  -> risk/safety/report pipeline
```

## Detector Plug-In Layer

Package:

```text
mddad/ai/detectors/
```

Backends:

- `mediapipe_only`: no phone detector.
- `yolo_pretrained`: general pretrained YOLO model.
- `yolo_finetuned`: future phone YOLO model.
- `rtdetr_finetuned`: future RT-DETR phone model.

Missing model files produce a warning path instead of forcing the app to stop.

## Data Flow

Uploaded video:

```text
Upload -> OpenCV metadata -> MediaPipe -> optional phone detector -> temporal events -> risk logs -> report -> session detail
```

Realtime:

```text
Start SocketIO event -> OpenCV webcam -> MediaPipe loop -> optional low-rate phone detector -> live updates/snapshots -> stop/finalize -> saved session detail
```
