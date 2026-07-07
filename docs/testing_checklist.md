# Testing Checklist

Use this checklist before a defense demo.

## Environment Setup

- [ ] Python environment is activated.
- [ ] `pip install -r requirements.txt` completed.
- [ ] `python app.py` starts.
- [ ] `http://127.0.0.1:5000/health` returns OK.
- [ ] Camera permission is available.
- [ ] Optional YOLO model exists at `models/yolo11n.pt`.

## Database Initialization

- [ ] `instance/mddad.sqlite3` is created automatically.
- [ ] Demo user exists.
- [ ] Tables include users, drivers, sessions, events, risk logs, reports, and app settings.
- [ ] Existing data is either useful for demo or intentionally cleared.

## Login

- [ ] Login page loads.
- [ ] `admin` / `admin123` works.
- [ ] Logout works.
- [ ] Protected pages redirect when logged out.

## Driver CRUD

- [ ] Create driver.
- [ ] View driver detail.
- [ ] Edit driver information.
- [ ] Select active driver.
- [ ] Dashboard and realtime pages show active driver.

## Upload Validation

- [ ] Upload accepts `.mp4`, `.avi`, `.mov`, `.mkv`.
- [ ] Empty upload is rejected.
- [ ] Unsupported file type is rejected.
- [ ] Upload requires selected driver.

## OpenCV Video Metadata

- [ ] `Analyze Video` extracts FPS.
- [ ] Frame count appears.
- [ ] Duration appears.
- [ ] Resolution appears.
- [ ] Processed frame count appears.

## Simulated Analysis

- [ ] `Run Simulated Analysis` completes.
- [ ] Event timeline is generated for long enough videos.
- [ ] Risk/safety/alert cards update.
- [ ] Report is generated.

## MediaPipe Analysis

- [ ] `Run MediaPipe Analysis` completes.
- [ ] Face detection counts appear.
- [ ] EAR/MAR metrics appear when a face is detected.
- [ ] No-face warning appears when appropriate.
- [ ] Report is generated.

## Full AI Analysis With Missing YOLO

- [ ] Temporarily remove or rename YOLO model.
- [ ] Run `Full AI Analysis`.
- [ ] App does not crash.
- [ ] MediaPipe results are still saved.
- [ ] YOLO warning is shown in session detail.

## Full AI Analysis With YOLO Model

- [ ] Place model at `models/yolo11n.pt`.
- [ ] Run `Full AI Analysis`.
- [ ] Phone metrics appear.
- [ ] Phone usage events appear when video contains phone usage.
- [ ] Report recommendations mention phone usage when detected.

## Realtime MediaPipe

- [ ] Realtime page loads.
- [ ] Start button disables after start.
- [ ] Stop button enables after start.
- [ ] Elapsed time updates.
- [ ] Risk/safety/alert values update.
- [ ] Behavior indicators update.
- [ ] Stop saves a session.

## Realtime YOLO Disabled

- [ ] `Realtime YOLO enabled` is `0`.
- [ ] Realtime monitoring still works.
- [ ] UI states phone detection is disabled.
- [ ] No YOLO model is required.

## Realtime YOLO Enabled

- [ ] Place model at `models/yolo11n.pt`.
- [ ] Set `Realtime YOLO enabled` to `1`.
- [ ] Keep sample FPS near `1`.
- [ ] Realtime monitoring starts.
- [ ] Phone state updates when phone is visible.
- [ ] Repeated detections can create `phone_usage` events.
- [ ] Missing model produces warning and does not crash.

## Snapshot Preview

- [ ] Snapshot preview is enabled.
- [ ] Preview area shows fallback before first frame.
- [ ] Low-rate JPEG preview appears during realtime monitoring.
- [ ] Overlay shows alert, risk, safety, and active states.
- [ ] Full video streaming is not used.

## Evidence Snapshots

- [ ] Event evidence is enabled.
- [ ] Realtime event creates a small JPEG under `uploads/evidence/`.
- [ ] Session detail timeline shows thumbnail.
- [ ] Missing evidence does not break timeline.

## Dashboard Metrics

- [ ] Total sessions updates.
- [ ] Average safety score updates.
- [ ] Latest safety score updates.
- [ ] Drowsiness/distraction/phone counts update.
- [ ] Recent sessions link to detail pages.

## History

- [ ] Uploaded sessions appear.
- [ ] Realtime sessions appear.
- [ ] Driver, type, status, scores, alert, duration, events, and links are visible.

## Reports

- [ ] Reports page lists generated reports.
- [ ] Driver filter works.
- [ ] Report links open session detail.

## Settings

- [ ] Settings page loads.
- [ ] Threshold values save.
- [ ] Reset to defaults works.
- [ ] Snapshot and realtime YOLO controls are visible.

## Evaluation CSV

- [ ] Evaluation page loads.
- [ ] Rows show session-level metrics.
- [ ] CSV export downloads.
- [ ] CSV includes expected columns.

## Failure Cases

- [ ] Camera unavailable shows realtime error without crash.
- [ ] Missing YOLO model shows warning without crash.
- [ ] Invalid video upload is rejected.
- [ ] Missing uploaded video file marks session failed.
- [ ] JPEG snapshot/evidence failures do not stop realtime monitoring.
