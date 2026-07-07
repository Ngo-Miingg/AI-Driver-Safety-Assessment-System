# Test Plan

## Functional Tests

- Login/logout with demo account.
- Driver create/select/view flows.
- Dashboard metrics after creating sessions.
- Upload validation for allowed/disallowed extensions.
- OpenCV metadata extraction.
- Simulated analysis.
- MediaPipe analysis.
- Full AI analysis with missing model.
- Full AI analysis with available phone detector.
- Realtime start/stop/finalize.
- Snapshot preview.
- Event evidence snapshots.
- History/session detail/report pages.
- Settings save/reset.
- Evaluation page and CSV export.

## Failure Tests

- Missing camera.
- Camera already in use.
- Missing YOLO/RT-DETR model.
- Missing Ultralytics package.
- Invalid upload file.
- Oversized upload file.
- Snapshot write failure.

## Performance Observations

Record:

- Realtime target FPS.
- Snapshot FPS.
- Processed frame count.
- Average processing FPS.
- Face detection rate.
- Whether realtime phone detector was enabled.

## Acceptance

The app passes the defense baseline when `python app.py` starts, realtime MediaPipe works, uploads work, Full AI degrades gracefully without model files, reports/evaluation are readable, and CSV export succeeds.
