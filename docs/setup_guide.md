# Setup Guide

## Start

```powershell
cd D:\Work\HocKy3\ComputerVision\MDDAD
python app.py
```

Open `http://127.0.0.1:5000`.

Demo login:

```text
username: admin
password: admin123
```

## Realtime Monitoring Test

1. Log in.
2. Create and select a driver.
3. Open `Realtime Monitoring`.
4. Click `Start Monitoring`.
5. Allow camera access if prompted.
6. Confirm the status panel changes to active.
7. Confirm elapsed time, live risk score, estimated safety score, alert level, and behavior indicators update.
8. Watch `Latest Timeline Events` for eye closure, yawn candidate, looking away, head down, or face missing events.
9. Click `Stop Monitoring`.
10. Click `Open Saved Session` and confirm the session detail page shows the realtime summary, timeline, risk breakdown, and report.

Config values:

```powershell
$env:REALTIME_CAMERA_INDEX="0"
$env:REALTIME_TARGET_FPS="8"
$env:REALTIME_ENABLE_YOLO="0"
$env:REALTIME_YOLO_SAMPLE_FPS="1"
$env:REALTIME_YOLO_CONFIDENCE_THRESHOLD="0.35"
$env:REALTIME_YOLO_MODEL_PATH="D:\Work\HocKy3\ComputerVision\MDDAD\models\yolo11n.pt"
$env:REALTIME_ENABLE_SNAPSHOT_PREVIEW="1"
$env:REALTIME_SNAPSHOT_FPS="2"
$env:REALTIME_SNAPSHOT_WIDTH="360"
$env:REALTIME_ENABLE_EVENT_EVIDENCE="1"
```

Realtime YOLO phone detection is optional and remains disabled by default for performance safety.

## Realtime Snapshot Preview Test

1. Log in and select a driver.
2. Open `Realtime Monitoring`.
3. Confirm `Snapshot Preview` shows the fallback message before monitoring starts.
4. Click `Start Monitoring`.
5. Confirm the preview updates at low rate, about 2 FPS by default.
6. Confirm the image has simple overlay text for alert level, risk score, safety score, and active behavior labels.
7. Confirm normal live metrics and event timeline still update.
8. Click `Stop Monitoring` and open the saved session.

The preview is low-rate JPEG snapshots over SocketIO. It is not full video streaming and should not send every processed frame.

If JPEG encoding fails, the page shows a preview warning and realtime monitoring continues.

## Optional Phone Detector Backend

The app works without trained phone models. Advanced phone detection can be selected through `.env`:

```text
PHONE_DETECTOR_BACKEND=yolo_pretrained
PHONE_DETECTOR_BACKEND=yolo_finetuned
PHONE_DETECTOR_BACKEND=rtdetr_finetuned
```

Model paths:

```text
models/yolo11n.pt
models/phone_yolo11n.pt
models/phone_rtdetr_r18.pt
```

The detector scaffold is plug-and-play; model training is still a separate future step.

## Optional Realtime YOLO / Phone Detector Test

1. Place a lightweight YOLO model at `models\yolo11n.pt`.
2. Open `Settings`.
3. Set `Realtime YOLO enabled` to `1`.
4. Keep `Realtime YOLO sample FPS` low, for example `1`.
5. Start realtime monitoring.
6. Confirm the phone detection status says realtime YOLO is enabled.
7. Hold a phone in view long enough for repeated low-rate detections.
8. Confirm phone state, risk score, event timeline, report, history, and session detail phone metrics update when `phone_usage` events are created.

If the model is missing or Ultralytics is unavailable, realtime monitoring continues with MediaPipe only and the UI shows a phone detection warning.

## Event Evidence Snapshot Test

1. Enable event evidence with `REALTIME_ENABLE_EVENT_EVIDENCE="1"` or the Settings page value `Event evidence enabled = 1`.
2. Run realtime monitoring long enough to trigger a temporal event.
3. Stop monitoring.
4. Open the saved session detail page.
5. In `Event Timeline`, check the `Evidence` column.

Evidence images are small JPEG files stored under:

```text
uploads\evidence
```

If no event evidence was created, the timeline remains usable and shows `--` in the evidence column.

## Dashboard Data Test

1. Run at least one upload analysis or realtime session.
2. Open `Dashboard`.
3. Confirm the metric cards show real values from SQLite:
   - total sessions
   - average safety score
   - latest safety score
   - drowsiness events
   - distraction events
   - phone usage events
4. Confirm `Recent Sessions` links to session detail pages.
5. Confirm the active-driver card reflects the selected driver.

## Upload Full AI Test

1. Place a YOLO model at `models\yolo11n.pt` if phone detection is needed.
2. Create and select a driver.
3. Open `Video Upload`.
4. Upload a video.
5. Click `Analyze Video` to extract OpenCV metadata.
6. Click `Run Simulated Analysis`, `Run MediaPipe Analysis`, or `Run Full AI Analysis`.

If the YOLO model is missing, Full AI Analysis still returns MediaPipe results and stores a YOLO warning.

## Reports Test

1. Generate a report by running simulated, MediaPipe, Full AI, or realtime analysis.
2. Open `Reports`.
3. Confirm reports are listed with driver, session, generated time, safety score, and alert level.
4. Use the driver filter if there are multiple drivers.

## Settings / Calibration Test

1. Log in.
2. Open `Settings`.
3. Review the current threshold values:
   - EAR threshold
   - MAR threshold
   - event minimum durations
   - YOLO confidence threshold
   - uploaded-video sample FPS
   - realtime target FPS
   - snapshot preview enabled
   - snapshot FPS
   - snapshot width
   - event evidence enabled
   - realtime YOLO enabled
   - realtime YOLO sample FPS
   - realtime YOLO confidence threshold
4. Change a value and click `Save Settings`.
5. Run a new uploaded or realtime analysis.
6. Return to `Settings` and click `Reset To Defaults` when finished.

Settings are stored in SQLite in `app_settings`. They apply to new analysis runs. Existing sessions keep their stored results until re-analyzed.

## Evaluation Test

1. Run at least one realtime session or uploaded-video analysis.
2. Open `Evaluation`.
3. Confirm rows show session type, driver, duration, safety score, risk score, alert level, event counts, processed frames, face detection rate, and phone count.
4. Open a session detail page and confirm `Evaluation Summary` is present.
5. Click `Export CSV` on the Evaluation page.

The CSV endpoint is:

```text
http://127.0.0.1:5000/evaluation.csv
```

## SocketIO Events

Client to server:

- `start_realtime_session`
- `stop_realtime_session`

Server to client:

- `realtime_status`
- `realtime_update`
- `realtime_event`
- `realtime_yolo_status`
- `realtime_snapshot`
- `realtime_snapshot_warning`
- `realtime_error`
- `realtime_completed`

## Current Limitations

- Realtime monitoring uses MediaPipe by default.
- Realtime YOLO phone detection is optional and disabled by default.
- Realtime preview uses low-rate JPEG snapshots only; no full video stream is sent.
- Head pose remains heuristic.
- Settings are calibration thresholds only; no model training is performed.
