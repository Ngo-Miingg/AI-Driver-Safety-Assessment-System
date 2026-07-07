# CODEX Handoff

## Current Project Phase

Implementation phase continues. The app is now functionally demo-ready and has a final defense documentation package.

## Latest ViT-DD Behavior Recognition Pass

Goal:

- Add the second AI branch for the planned multimodal system: driver behavior recognition.
- Keep the current phone-detection branch intact.
- Do not force heavy PyTorch dependencies into Flask startup.

What was implemented:

- Added optional ViT-DD-style behavior recognition adapter at `mddad/ai/behavior_recognition.py`.
- Added protected `/behavior-test` page for testing driver behavior recognition from an image or video.
- Added model configuration keys:
  - `VITDD_MODEL_PATH`
  - `VITDD_MODEL_NAME`
  - `VITDD_IMAGE_SIZE`
  - `VITDD_SAMPLE_FPS`
  - `VITDD_LABELS`
- Added `kaggle/vitdd_driver_behavior_kaggle_train.ipynb` to train/fine-tune a Vision Transformer behavior classifier on `c0..c9` driver-behavior datasets.
- Added `kaggle/README_KAGGLE_VITDD_BEHAVIOR_TRAIN.md`.
- Added `docs/vitdd_behavior_integration.md`.
- Added `Behavior Test` to the main navigation.

Files changed:

- `.env.example`
- `config.py`
- `mddad/__init__.py`
- `mddad/ai/behavior_recognition.py`
- `mddad/behavior_test_routes.py`
- `templates/base.html`
- `templates/behavior_test.html`
- `tools/prepare_vitdd_behavior_training.py`
- `kaggle/vitdd_driver_behavior_kaggle_train.ipynb`
- `kaggle/README_KAGGLE_VITDD_BEHAVIOR_TRAIN.md`
- `docs/vitdd_behavior_integration.md`
- `docs/CODEX_HANDOFF.md`

How to train:

1. Prepare a behavior dataset with class folders `c0..c9`, preferably State Farm, AUCDD, or another distracted-driver dataset.
2. Upload the dataset to Kaggle.
3. Import and run `kaggle/vitdd_driver_behavior_kaggle_train.ipynb`.
4. Download `/kaggle/working/vitdd_behavior_training_results.zip`.
5. Copy `vitdd_behavior.pt` to `models/vitdd_behavior.pt`.
6. Run the Flask app and open `/behavior-test`.

How to test in the app:

1. Run `python app.py`.
2. Log in with `admin` / `admin123`.
3. Open `Behavior Test`.
4. Upload a driver image or short video.
5. Use model path `D:\Work\HocKy3\ComputerVision\MDDAD\models\vitdd_behavior.pt`.
6. Confirm the page shows top behavior class, confidence, probabilities, and annotated output.

Current limitation:

- The behavior branch is isolated on `/behavior-test` first. It is not yet fused into realtime monitoring, uploaded-video Full AI Analysis, risk scoring, reports, or evaluation CSV.
- The app will show a clear error if `torch`, `torchvision`, `timm`, or the checkpoint file is missing, but Flask startup remains usable.
- This adapter is a practical ViT-DD-style ViT classifier integration. The original Purdue ViT-DD source/checkpoints can still be used as reference, but the local app expects a `timm` ViT classifier checkpoint for the first integration pass.

Recommended next step:

- Train/fine-tune `vitdd_behavior.pt`, validate `/behavior-test`, then add a controlled fusion step so behavior predictions can contribute to session events and risk score.

## Latest Production Candidate Pass

Architecture summary:

- Flask remains the application runtime.
- Flask-SocketIO remains the realtime transport.
- SQLite remains the local persistence layer.
- OpenCV handles uploaded-video metadata and webcam/video frames.
- MediaPipe handles face landmarks for drowsiness and distraction signals.
- Phone detection is now routed through a plug-in detector layer under `mddad/ai/detectors/`.
- Supported detector backends are `mediapipe_only`, `yolo_pretrained`, `yolo_finetuned`, and `rtdetr_finetuned`.
- Missing model files do not make the full app unusable; Full AI and realtime phone detection continue to degrade through warning/error paths.

Files changed or added in this pass:

- `.env.example`
- `config.py`
- `mddad/__init__.py`
- `mddad/logging_config.py`
- `mddad/ai/yolo_detector.py`
- `mddad/ai/detectors/__init__.py`
- `mddad/ai/detectors/base.py`
- `mddad/ai/detectors/factory.py`
- `mddad/ai/detectors/ultralytics_detector.py`
- `mddad/yolo_phone_analysis.py`
- `mddad/analysis_routes.py`
- `mddad/realtime_monitor.py`
- `docs/production_audit.md`
- `docs/database_schema.md`
- `docs/logging_guide.md`
- `docs/offline_deployment.md`
- `docs/test_plan.md`
- `docs/manual_test_checklist.md`
- `docs/demo_mode.md`
- `docs/system_overview.md`
- `docs/architecture_overview.md`
- `docs/ai_pipeline.md`
- `docs/deployment_guide.md`
- `docs/troubleshooting.md`
- `docs/CODEX_HANDOFF.md`

What was implemented:

- Added `.env` loading via `python-dotenv`.
- Added centralized detector backend config for pretrained YOLO, fine-tuned YOLO, and fine-tuned RT-DETR paths.
- Added detector factory scaffold for plug-and-play phone detector backends.
- Kept legacy `YOLODetector` as a compatibility wrapper.
- Added configurable phone driver ROI and temporal confirmation settings.
- Full AI uploaded-video phone detection now receives detector backend, ROI, and temporal confirmation config.
- Optional realtime phone detection now uses the detector factory.
- Added rotating file logging under `logs/app.log`.
- Created production audit, schema, logging, offline deployment, QA, demo, architecture, AI pipeline, deployment, and troubleshooting docs.

Tests executed:

- Python compile for `app.py`, `config.py`, `mddad/__init__.py`, `mddad/logging_config.py`, detector modules, phone analysis, analysis routes, and realtime monitor.

Remaining risks:

- RT-DETR backend is scaffolded but not validated with an actual `models/phone_rtdetr_r18.pt` file yet.
- Fine-tuned YOLO backend is scaffolded but not validated with `models/phone_yolo11n.pt` yet.
- Driver ROI defaults to full frame for compatibility; it should be calibrated per camera/video position.
- Face-phone proximity is documented as planned, not fully implemented, because the current phone pipeline does not pass reliable face bounding boxes into detector filtering.
- Automated end-to-end test coverage is still limited.

Next step:

- Train or obtain a validated phone detector, place it under `models/phone_yolo11n.pt` or `models/phone_rtdetr_r18.pt`, select the backend in `.env`, then run Full AI and realtime phone-detection validation.

## Language Selection Pass

What was implemented:

- Added lightweight built-in language selection for English and Vietnamese.
- Added `mddad/i18n.py` with supported languages, translation dictionary, template helper `{{ _('key', 'Default') }}`, and `/language/<language>` route.
- Added header language switcher visible before and after login.
- Stored selected language in Flask session.
- Preserved selected language across login/logout.
- Expanded Vietnamese coverage across the main application: navigation, login/register, dashboard, drivers, driver form/detail, history, realtime, upload, settings, reports, evaluation, session detail, foundation, placeholder pages, realtime JavaScript states, and common flash messages.
- Added helpers for translated system values such as session type, status, event type, severity, alert level, and settings labels/descriptions.
- Added `rt(...)` helper to translate common stored report phrases at display time, so older English generated reports render more appropriately in Vietnamese.

Files changed:

- `mddad/i18n.py`
- `mddad/__init__.py`
- `mddad/auth.py`
- `mddad/driver_routes.py`
- `mddad/upload_routes.py`
- `mddad/analysis_routes.py`
- `mddad/settings_routes.py`
- `templates/base.html`
- `templates/login.html`
- `templates/register.html`
- `templates/dashboard.html`
- `templates/drivers.html`
- `templates/driver_form.html`
- `templates/driver_detail.html`
- `templates/history.html`
- `templates/realtime.html`
- `templates/upload.html`
- `templates/settings.html`
- `templates/reports.html`
- `templates/evaluation.html`
- `templates/session_detail.html`
- `templates/home.html`
- `templates/placeholder.html`
- `static/css/main.css`
- `static/js/main.js`
- `static/js/realtime.js`
- `docs/CODEX_HANDOFF.md`

How to test:

1. Run `python app.py`.
2. Open `http://127.0.0.1:5000/login`.
3. Click `Tiếng Việt`; confirm the page title, header, navigation, login/register, dashboard, drivers, history, realtime, upload, settings, reports, evaluation, and session detail pages display Vietnamese UI labels.
4. Log in with `admin` / `admin123`; confirm the selected language persists.
5. Click `English`; confirm the UI switches back to English.

Remaining language limitations:

- Driver names, filenames, model names such as MediaPipe/YOLO, and timestamps are not translated.
- Stored report bodies are translated by common phrase replacement at display time, not by a full natural-language translation engine.
- If new UI text is added later, add a key to `mddad/i18n.py` and render it with `{{ _('key', 'Default text') }}` or `{{ tr(value) }}`.

## Work Status Summary

### Done

- Built the Flask application structure with authentication, login/logout, demo account, and protected pages.
- Added multi-driver management and active-driver selection.
- Added SQLite persistence, schema migrations, seeded demo user, and session/report/event storage.
- Implemented dashboard metrics from real SQLite data instead of placeholders.
- Implemented uploaded-video workflow with validation, OpenCV metadata extraction, simulated analysis, MediaPipe analysis, and Full AI Analysis.
- Added uploaded-video YOLO phone detection support when a YOLO weight file exists.
- Added graceful fallback when YOLO/Ultralytics/model weights are missing.
- Implemented realtime webcam monitoring with Flask-SocketIO, OpenCV camera capture, and MediaPipe face/behavior analysis.
- Fixed Socket.IO browser client loading by vendoring `static/vendor/socket.io.min.js` locally.
- Added realtime debug logging for client connect/start/stop/update/snapshot and server receive/camera/frame/update/snapshot events.
- Added `/debug/camera` for authenticated debug camera checks.
- Added low-rate JPEG snapshot preview over SocketIO; full video streaming is intentionally not implemented.
- Added small optional realtime event evidence snapshots under `uploads/evidence/`.
- Added optional realtime YOLO phone detection, disabled by default for performance safety.
- Added detector plug-in scaffold for `mediapipe_only`, `yolo_pretrained`, `yolo_finetuned`, and `rtdetr_finetuned`.
- Added `.env.example`, offline deployment notes, logging guide, production audit, architecture docs, AI pipeline docs, and production test plan.
- Added English/Vietnamese language selector with session persistence and translated common UI pages.
- Added realtime speed tuning: camera buffer reduction, higher default realtime FPS, lower snapshot width, bounded live risk/timeline recalculation.
- Added risk score, safety score, alert level, fatigue trend, event timeline, and rule-based AI driving report.
- Upgraded history, session detail, reports, settings/calibration, evaluation page, and evaluation CSV export.
- Created final defense documents: `README.md`, `docs/demo_script.md`, `docs/testing_checklist.md`, `docs/demo_assets_guide.md`, `docs/evaluation_plan.md`, and `docs/model_guide.md`.
- Verified basic startup with `python app.py` and `/health`.

### Not Done

- No custom model training has been implemented.
- No trained drowsiness model exists; drowsiness/distraction remains MediaPipe landmark plus rule-based threshold logic.
- No fine-tuned phone YOLO model is bundled; place optional weights at `models/yolo11n.pt` or configure another path.
- No fine-tuned RT-DETR phone model is bundled; place optional weights at `models/phone_rtdetr_r18.pt` when available.
- Full realtime video streaming is not implemented; realtime preview remains low-rate JPEG snapshots only.
- Uploaded-video event evidence snapshots are not implemented.
- Ground-truth evaluation is not implemented; evaluation currently exports system metrics, not precision/recall against labeled data.
- Automated test coverage is limited; most end-to-end checks are still manual smoke tests.
- Production deployment packaging such as Docker, HTTPS, production server config, and backup strategy is not implemented.
- Realtime YOLO is not enabled by default and still needs manual performance testing on the defense laptop.
- RT-DETR backend is available as a plug-in scaffold but still needs model-file validation.
- A prepared demo video dataset is not included in the repository.

### In Progress / Next

- Manually test realtime webcam on the actual defense laptop and browser.
- Tune `Realtime target FPS`, `Snapshot FPS`, `Snapshot width`, and behavior thresholds in Settings based on real webcam performance.
- Prepare short demo videos for normal driving, eye closure, yawn, looking away/head down, phone usage, and mixed-risk scenarios.
- Test Full AI Analysis both without YOLO weights and with `models/yolo11n.pt`.
- Rehearse the full `docs/demo_script.md` flow from login through realtime, session detail, upload analysis, evaluation, CSV export, and settings explanation.
- If time allows, improve live risk tuning so risk/safety changes are more obvious during short realtime events.
- If time allows, add uploaded-video evidence snapshots for key timeline events.

## Documentation Created

- `README.md`: project overview, features, stack, run instructions, workflows, limitations, and future work.
- `docs/demo_script.md`: 5-10 minute defense demo flow.
- `docs/testing_checklist.md`: manual QA checklist for setup, upload, realtime, YOLO, reports, settings, evaluation, and failure cases.
- `docs/demo_assets_guide.md`: recommended demo video scenarios, naming, lighting, camera setup, and formats.
- `docs/evaluation_plan.md`: functional, qualitative, and quantitative evaluation approach.
- `docs/model_guide.md`: MediaPipe/YOLO model explanation, optional fine-tuning guidance, and model path instructions.

## Latest Realtime Speed Pass

Root cause addressed: the webcam was functioning, but the preview could feel slow because snapshot preview defaulted to a very low rate and the realtime loop recalculated full timeline/risk data on every processed frame. That work grows during a longer session and can make camera updates less responsive.

Files changed in this pass:

- `config.py`
- `mddad/settings.py`
- `mddad/realtime_monitor.py`
- `docs/CODEX_HANDOFF.md`

What changed:

- Increased new default realtime target FPS from `5` to `8`.
- Increased new default snapshot preview FPS from `1` to `2`.
- Reduced new default snapshot width from `480` to `360` to lower JPEG encode and SocketIO payload cost.
- Allowed Snapshot FPS up to `5` in Settings for local demo tuning.
- Configured webcam capture with low buffer size, 640x480 frame size, and requested target FPS to reduce camera latency.
- Cached live timeline/risk calculation and recalculates it at a bounded rate instead of rebuilding it on every frame.
- Avoided repeated live event duplicates by keying realtime event emission on event type and start time.
- Live risk now uses recent rolling risk logs so short events are more visible in the live panel.
- Updated the local SQLite `app_settings` rows for this workspace to `realtime_target_fps=8.0`, `snapshot_fps=2.0`, and `snapshot_width=360.0` so the speed tuning applies immediately without requiring a manual reset.

How to test speed:

1. Run `python app.py`.
2. Log in and open `Settings`.
3. If old values are stored in SQLite, set `Realtime target FPS` to `8`, `Snapshot FPS` to `2` or `3`, and `Snapshot width` to `360`; or click `Reset To Defaults`.
4. Open `Realtime Monitoring`, click `Start Monitoring`, and confirm the preview updates more smoothly.
5. Keep realtime YOLO disabled unless specifically testing phone detection, because it can reduce webcam responsiveness.

## Latest Realtime Debug Pass

Root cause found: the realtime pipeline existed, but failures were too silent for demo use. If Socket.IO client loading, Socket.IO connection, camera opening, frame reading, or snapshot settings failed, the page could remain on the fallback preview text without enough visible or terminal feedback. No namespace or event-name mismatch was found in the implemented SocketIO flow.

Files changed in this pass:

- `static/js/realtime.js`
- `mddad/sockets.py`
- `mddad/realtime_monitor.py`
- `mddad/realtime_routes.py`
- `docs/CODEX_HANDOFF.md`

What changed:

- Added browser console debug logs for SocketIO connect, Start click, emitted start/stop events, realtime updates, snapshots, warnings, errors, and completion.
- Added visible UI error when the Socket.IO client script fails to load.
- Added server logs for SocketIO connect/disconnect, server receiving `start_realtime_session`, background task start, camera open success/failure, first frame read, first realtime update, and first snapshot.
- Changed active realtime status display to `Running`.
- Added camera open helper with a Windows DirectShow fallback backend.
- Added `/debug/camera` route in debug mode for authenticated users.
- Realtime status now tells the browser when snapshot preview is disabled by settings.

How to test realtime webcam:

1. Run `python app.py`.
2. Log in with `admin` / `admin123`.
3. Create/select a driver.
4. Open `/debug/camera` while in debug mode and confirm camera index `0` or the configured index opens.
5. Open `Realtime Monitoring`.
6. Open browser DevTools Console.
7. Click `Start Monitoring`.
8. Confirm status changes to `Running`, elapsed time increases, risk/safety/alert update, active states update, and snapshot preview shows an image if enabled.
9. Click `Stop Monitoring` and confirm the saved session detail link appears.

Remaining realtime limitations:

- Snapshot preview still depends on SocketIO being available in the browser, now through the vendored local Socket.IO client file.
- If another application owns the camera, `/debug/camera` and realtime monitoring will report camera failure.
- Snapshot preview can be disabled from Settings, in which case the preview area intentionally stays in fallback mode.
- Snapshot preview remains low-rate JPEG snapshots, not full video streaming. It should be tuned to `2-3` FPS for demos, not treated as a 30 FPS video feed.

## Full Implementation Audit Pass

What was audited:

- `app.py`, `config.py`, application factory, SocketIO setup, schema, migrations, routes, templates, static JS/CSS, AI helper modules, README, and docs.
- Realtime start/stop/update/snapshot flow.
- Upload, metadata, simulated, MediaPipe, Full AI, history, reports, settings, evaluation, and CSV workflows.

Critical issues found:

- No critical primary-navigation placeholder route remained.
- `/foundation` still had stale placeholder text saying AI/workflows would be implemented later.
- Migration coverage was incomplete for older DBs: several session columns used by current code were not listed in `required_columns`.
- Realtime failure modes were too quiet when SocketIO client loading, camera opening, frame reading, or snapshot emission failed.

What was fixed:

- Added `docs/implementation_audit.md`.
- Expanded SQLite migration coverage for currently used session columns.
- Updated `templates/home.html` to remove stale placeholder copy.
- Added realtime client/server debug logging and clearer realtime UI states in the previous debug pass.
- Verified route render, upload analysis, realtime fake-camera update/snapshot, settings, evaluation, reports, and app startup.

Files changed in audit pass:

- `mddad/database.py`
- `templates/home.html`
- `docs/implementation_audit.md`
- `docs/CODEX_HANDOFF.md`

Self-tested:

- Python compile.
- `python app.py` startup and `/health`.
- Login with `admin` / `admin123`.
- Dashboard login protection.
- Driver create/select.
- Upload page and generated MP4 upload.
- OpenCV metadata analysis.
- Simulated analysis.
- MediaPipe uploaded-video analysis.
- Full AI fallback when YOLO is missing.
- Realtime page render.
- Realtime fake-camera `realtime_update`.
- Realtime fake-camera `realtime_snapshot`.
- Realtime stop/finalization/session save.
- History, reports, settings, evaluation, and CSV export.
- `/debug/camera` route in debug mode.

Still requires manual testing:

- Real physical webcam behavior in the target browser.
- Browser DevTools logs during realtime Start/Stop.
- Snapshot preview with the actual camera.
- Realtime YOLO with an actual `models/yolo11n.pt`.
- Full AI phone usage detection with prepared phone videos.

Known limitations:

- Socket.IO client is vendored locally at `static/vendor/socket.io.min.js`; the realtime page should work without internet access.
- Realtime camera can fail if another app owns the webcam.
- Realtime YOLO is optional and performance-dependent.
- Uploaded-video evidence snapshots are still not implemented.

Next recommended action:

- Run the manual webcam test on the actual defense laptop, then rehearse the full `docs/demo_script.md` flow with prepared demo assets.

## Socket.IO Client Loading Fix

Root cause:

- `templates/base.html` loaded Socket.IO from the CDN at the bottom of the page.
- `templates/realtime.html` loaded `static/js/realtime.js` inside the content block before the base template reached the Socket.IO script tag.
- When the CDN was unavailable or `realtime.js` ran first, `window.io` was undefined and the browser logged: `Socket.IO client is missing; start_realtime_session cannot be emitted.`

Files changed:

- `templates/base.html`
- `templates/realtime.html`
- `static/js/realtime.js`
- `static/vendor/socket.io.min.js`
- `docs/CODEX_HANDOFF.md`

How Socket.IO client is loaded now:

- The Socket.IO browser client is vendored locally at `static/vendor/socket.io.min.js`.
- `templates/base.html` loads it locally before `static/js/main.js`.
- `templates/base.html` now exposes a `{% block scripts %}` after the Socket.IO client.
- `templates/realtime.html` loads `static/js/realtime.js` in that scripts block, so `window.io` exists before realtime code runs.
- The app no longer depends on internet/CDN access for Socket.IO client loading.

How to test:

1. Run `python app.py`.
2. Log in with `admin` / `admin123`.
3. Select a driver.
4. Open `/realtime`.
5. In browser DevTools Console, run `window.io`; it should be defined.
6. Click `Start Monitoring`.
7. Confirm the server logs `Received start_realtime_session`.
8. Confirm status changes to `Running`, `realtime_update` reaches the browser, and snapshot preview appears if enabled and camera is available.

Verified:

- `/static/vendor/socket.io.min.js` returns HTTP 200 locally.
- Rendered `/realtime` loads scripts in this order: local Socket.IO client, `main.js`, then `realtime.js`.
- No `cdn.socket.io` reference remains in templates/static/docs checked during this pass.
- Fake-camera SocketIO smoke test still receives `realtime_update`, `realtime_snapshot`, and `realtime_completed`.
- `python app.py` starts and `/health` returns OK.

## Current Project Readiness

- Core app workflows are implemented: auth, drivers, dashboard, upload analysis, realtime monitoring, history, reports, settings, evaluation, and CSV export.
- Realtime monitoring works in MediaPipe-only mode by default.
- Realtime YOLO phone detection is optional and disabled by default.
- Uploaded Full AI Analysis can use YOLO when weights are available and degrades with a warning when they are missing.
- Low-rate snapshot preview and realtime evidence snapshots are implemented.
- The project is ready for manual defense rehearsal with prepared videos and webcam samples.

## What Was Improved

- Realtime page now has a clearer session status panel.
- Realtime page shows elapsed time, live risk score, estimated safety score, alert badge, active behavior indicators, live event timeline, and rolling risk values.
- Start button is disabled while monitoring; Stop button is enabled only while monitoring.
- After realtime completion or failure, the page shows `Open Saved Session` when a session id exists.
- Realtime preview uses optional low-rate JPEG snapshots; full video streaming is not implemented.
- Dashboard now shows real SQLite metrics: total sessions, average safety score, latest safety score, drowsiness events, distraction events, phone usage events, recent sessions, active driver, and quick links.
- History page now shows useful demo data for uploaded and realtime sessions.
- Session detail page now has summary score cards, analysis mode, fatigue trend, realtime/video metadata, MediaPipe metrics, phone metrics, YOLO warnings, event timeline, risk breakdown, and report content.
- Reports page now lists generated reports for the logged-in user with optional driver filtering.
- CSS was polished for cards, badges, alert colors, tables, navigation, and empty states.
- SQLite migration flow was tightened so child tables are not left referencing `sessions_old` after a sessions-table rebuild.
- Added SQLite-backed calibration settings for EAR, MAR, event minimum durations, YOLO confidence, uploaded sample FPS, and realtime target FPS.
- Added protected `Settings` page with editable threshold values and reset-to-defaults.
- Added protected `Evaluation` page with recent session metrics, event counts by type, processed frames, face detection rate, processing FPS, and phone counts.
- Added evaluation CSV export at `/evaluation.csv`.
- Added session-level `Evaluation Summary` to session detail.
- Added optional low-rate realtime JPEG snapshot preview over SocketIO.
- Added lightweight snapshot overlays with alert level, risk score, safety score, and active behavior labels.
- Added optional realtime event evidence snapshots saved as small JPEG files under `uploads/evidence/`.
- Added `evidence_image_path` to events and evidence thumbnails on session detail.
- Added optional realtime YOLO phone detection at low sample FPS, disabled by default.
- Realtime phone detections can create `phone_usage` events that affect risk, safety, alert level, timeline, report, history, and detail metrics.
- Missing YOLO model or Ultralytics package degrades gracefully to MediaPipe-only realtime monitoring.
- Added final README and defense preparation documentation.

## Files Changed

- `mddad/dashboard_routes.py`
- `mddad/schema.sql`
- `config.py`
- `mddad/database.py`
- `mddad/analysis_routes.py`
- `mddad/mediapipe_analysis.py`
- `mddad/realtime_monitor.py`
- `mddad/upload_routes.py`
- `mddad/settings.py`
- `mddad/settings_routes.py`
- `mddad/evaluation.py`
- `templates/base.html`
- `templates/dashboard.html`
- `templates/history.html`
- `templates/realtime.html`
- `templates/reports.html`
- `templates/settings.html`
- `templates/evaluation.html`
- `templates/session_detail.html`
- `static/js/realtime.js`
- `static/css/main.css`
- `docs/project_structure.md`
- `docs/setup_guide.md`
- `docs/CODEX_HANDOFF.md`
- `README.md`
- `docs/demo_script.md`
- `docs/testing_checklist.md`
- `docs/demo_assets_guide.md`
- `docs/evaluation_plan.md`
- `docs/model_guide.md`

## How To Test Realtime UX

1. Run `python app.py`.
2. Log in with `admin` / `admin123`.
3. Create and select a driver.
4. Open `Realtime Monitoring`.
5. Click `Start Monitoring`.
6. Confirm the Start button disables and Stop button enables.
7. Confirm elapsed time, risk, safety, alert level, and active behavior indicators update.
8. Click `Stop Monitoring`.
9. Confirm `Open Saved Session` appears.
10. Open the saved session and verify the realtime detail, timeline, risk breakdown, and report.

If the camera is unavailable, the app emits `realtime_error`, marks the saved session failed when a session exists, and does not crash.

## How To Test Dashboard Data

1. Run a realtime session or upload and analyze a video.
2. Open `Dashboard`.
3. Confirm metric cards are populated from SQLite.
4. Confirm `Recent Sessions` contains the saved session.
5. Open the recent session link and verify the session detail page.

## How To Use Settings / Calibration

1. Log in.
2. Open `Settings`.
3. Edit threshold values:
   - EAR threshold
   - MAR threshold
   - eye closure minimum duration
   - yawn minimum duration
   - looking away minimum duration
   - head down minimum duration
   - face missing minimum duration
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
4. Click `Save Settings`.
5. Run a new analysis or realtime session.
6. Use `Reset To Defaults` to return to config defaults.

Settings are stored in SQLite in the `app_settings` table and apply to new analysis runs. Existing sessions do not change until re-analyzed.

## How To Test Snapshot Preview

1. Ensure snapshot preview is enabled in `Settings` or with env config:

```text
REALTIME_ENABLE_SNAPSHOT_PREVIEW=1
REALTIME_SNAPSHOT_FPS=2
REALTIME_SNAPSHOT_WIDTH=360
```

2. Log in, select a driver, and open `Realtime Monitoring`.
3. Click `Start Monitoring`.
4. Confirm `Snapshot Preview` updates at low rate with overlay text.
5. Confirm regular realtime metrics and event timeline still update.
6. Stop monitoring and open the saved session.

SocketIO sends occasional JPEG snapshots only. It does not send every processed frame and does not implement full video streaming.

## How Event Evidence Snapshots Work

- When realtime monitoring confirms a temporal event, the current frame is resized, annotated, JPEG-encoded, and saved under `uploads/evidence/`.
- The event row stores the relative `evidence_image_path`.
- Session detail shows an evidence thumbnail in the event timeline when a path exists.
- If folder creation, JPEG encoding, or file write fails, realtime emits a warning and keeps running.
- Uploaded-video analysis still works without evidence images unless a future phase adds uploaded-video evidence capture.

## Snapshot Configuration

- `REALTIME_ENABLE_SNAPSHOT_PREVIEW`: enables or disables low-rate preview.
- `REALTIME_SNAPSHOT_FPS`: maximum snapshot emission rate. Default is `2`.
- `REALTIME_SNAPSHOT_WIDTH`: resized JPEG width. Default is `360`.
- `REALTIME_ENABLE_EVENT_EVIDENCE`: enables or disables event evidence image saving.

These values also appear on the Settings page as editable numeric values. Use `1` for enabled and `0` for disabled.

## Optional Realtime YOLO

Realtime YOLO phone detection is disabled by default. To test it:

1. Place a lightweight model at `models/yolo11n.pt`, or set `REALTIME_YOLO_MODEL_PATH`.
2. Open `Settings`.
3. Set `Realtime YOLO enabled` to `1`.
4. Keep `Realtime YOLO sample FPS` around `1`.
5. Start realtime monitoring.

When enabled and available, YOLO runs on the current webcam frame at low FPS. Repeated phone detections are converted into temporal `phone_usage` events, then included in the same risk, report, history, and session-detail pipeline as other events.

If the model is missing or Ultralytics is unavailable, the realtime session emits a warning, shows phone detection as unavailable, stores the warning in session phone metrics after stop, and continues MediaPipe-only.

Config keys:

- `REALTIME_ENABLE_YOLO`
- `REALTIME_YOLO_SAMPLE_FPS`
- `REALTIME_YOLO_CONFIDENCE_THRESHOLD`
- `REALTIME_YOLO_MODEL_PATH`

## How To View Evaluation Results

1. Run realtime monitoring or uploaded-video analysis.
2. Open `Evaluation`.
3. Review session type, driver, duration, safety score, risk score, alert level, event counts by type, processed frame count, processing FPS, face detection rate, and phone usage count.
4. Open a session detail page to see the per-session `Evaluation Summary`.

## How To Export CSV

Open `Evaluation` and click `Export CSV`, or directly visit:

```text
/evaluation.csv
```

The CSV includes session id, driver, type, status, duration, scores, alert level, event counts, frame counts, face detection rate, and analysis mode.

## Preview Decision

Low-rate JPEG snapshot preview is implemented. Full video streaming is not implemented. SocketIO sends occasional resized JPEG snapshots only when enabled, not every processed frame.

## What Is Still Not Implemented

- YOLO model training or fine-tuning.
- Full video streaming.
- Uploaded-video event evidence extraction.
- Production-grade 3D head pose.
- Model-trained drowsiness classifier.

## What Should Be Tested Manually Next

- Run the 5-10 minute demo script from start to finish.
- Test all prepared demo videos from `docs/demo_assets_guide.md`.
- Test Full AI with `models/yolo11n.pt` present and missing.
- Test realtime monitoring with YOLO disabled and enabled.
- Confirm snapshot preview works on the actual demo webcam.
- Confirm at least one realtime event evidence image appears in session detail.
- Export `/evaluation.csv` after several sessions and fill the evaluation table.

## Model Files Needed

- Required for MediaPipe: none.
- Recommended for YOLO phone detection: `models/yolo11n.pt`.
- Optional fine-tuned phone model: `models/phone_yolo11n.pt`.

Training is optional, not mandatory. The current defense can use pretrained YOLO or run without YOLO for MediaPipe-only parts.

## Recommended Next Step

Run a full rehearsal using `docs/demo_script.md`, record the results in `docs/testing_checklist.md`, and prepare 3-6 short videos following `docs/demo_assets_guide.md`.

## Phone Detector Fine-Tuning Update

Fine-tuned phone detector outputs were reviewed from:

- `D:\Work\output\results_finetune_rtdetr.zip`
- `D:\Work\output\results_test_yolo_rtdetr.zip`

Evaluation report:

- `docs/phone_detector_training_report.md`

Model decision:

- Main detector: RT-DETR-L serious fine-tune.
- Baseline/fallback detector: YOLO26n fine-tuned baseline.

Validation summary:

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO26n baseline | 0.93237 | 0.86031 | 0.92854 | 0.76828 |
| RT-DETR-L compare | 0.93969 | 0.93286 | 0.95693 | 0.77543 |
| RT-DETR-L serious | 0.93331 | 0.92239 | 0.95497 | 0.80077 |

Integrated model files:

- `models/phone_rtdetr_l.pt` - selected main phone detector.
- `models/phone_yolo26n.pt` - YOLO baseline/fallback.
- `models/phone_yolo11n.pt` - compatibility copy for older config references.
- `models/phone_rtdetr_l_light_compare.pt` - comparison checkpoint.

Config changes:

- `YOLO_FINETUNED_MODEL_PATH` now defaults to `models/phone_yolo26n.pt`.
- `RTDETR_MODEL_PATH` remains `models/phone_rtdetr_l.pt`.
- `requirements.txt` now requires `ultralytics>=8.4.68`.

How to test:

1. Run `python app.py`.
2. Open `/phone-test`.
3. Select `RT-DETR fine-tuned` and test images/videos with phone usage.
4. Select `YOLO fine-tuned` and compare speed/boxes on the same media.

Remaining manual work:

- Inspect sample prediction images from the exported Kaggle runs.
- Test real demo media through `/phone-test`.
- Decide whether realtime should use RT-DETR at low sample FPS or YOLO26n for speed.

## Server Stability Fix

- Root cause: Flask debug mode was using the Werkzeug auto-reloader. On Windows, the reloader detected apparent changes inside `D:\Work\.venv\Lib\site-packages\paddle*` and restarted the app repeatedly.
- Fix applied: `config.py` now has `USE_RELOADER`, controlled by `FLASK_USE_RELOADER`, defaulting to `0`.
- `app.py` passes `use_reloader=app.config.get("USE_RELOADER", False)` to `socketio.run`.
- `.env.example` documents `FLASK_USE_RELOADER=0`.
- Recommended demo setting: keep `FLASK_DEBUG=1` only if you want local error pages, but keep `FLASK_USE_RELOADER=0` for stable realtime/camera behavior.
