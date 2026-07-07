# Defense Demo Script

Target duration: 5-10 minutes.

## 1. Login

- Open `http://127.0.0.1:5000`.
- Log in with `admin` / `admin123`.
- Explain that the app is local, SQLite-backed, and prepared for controlled demo workflows.

## 2. Create Or Select Driver

- Open `Drivers`.
- Create a driver if needed.
- Select the driver.
- Explain that every session is linked to a driver for history, reports, and evaluation.

## 3. Dashboard Overview

- Open `Dashboard`.
- Point out total sessions, average safety score, latest safety score, event counts, active driver, quick links, and recent sessions.
- Emphasize that the numbers come from SQLite, not placeholders.

## 4. Realtime Monitoring

- Open `Realtime Monitoring`.
- Click `Start Monitoring`.
- Show elapsed time, live risk score, safety score, alert level, behavior indicators, snapshot preview, and timeline events.
- Explain that MediaPipe runs continuously and optional YOLO runs only at low FPS if enabled.
- If realtime YOLO is disabled, explain that this is the default performance-safe mode.

## 5. Stop And Open Saved Session

- Click `Stop Monitoring`.
- Click `Open Saved Session`.
- Explain that the app persists session metadata, events, risk logs, final scores, and report.

## 6. Timeline, Evidence, And Report

- On session detail, show safety/risk/alert cards.
- Show `Evaluation Summary`.
- Show `Event Timeline`.
- If evidence snapshots exist, open one thumbnail.
- Show MediaPipe metrics, phone metrics, risk breakdown, and report.

## 7. Upload Video

- Open `Video Upload`.
- Upload a prepared demo video.
- Click `Analyze Video`.
- Explain OpenCV metadata extraction: FPS, duration, frame count, resolution, sampled frames.

## 8. Run Full AI Analysis

- Click `Run Full AI Analysis`.
- Explain that this combines MediaPipe behavior analysis with YOLO phone detection for uploaded videos.
- If YOLO weights are missing, show the warning and explain graceful fallback.

## 9. History, Evaluation, CSV Export

- Open `History`.
- Show uploaded and realtime sessions together.
- Open `Evaluation`.
- Show event counts, face detection rate, processing FPS, processed frames, and phone counts.
- Click `Export CSV`.
- Explain that this supports defense evaluation and comparison across test videos.

## 10. Settings / Calibration

- Open `Settings`.
- Show EAR/MAR thresholds, duration thresholds, sample FPS, snapshot controls, and realtime YOLO controls.
- Explain that settings affect new analyses and can be reset to defaults.

## Closing Explanation

- The system is explainable: raw signals become events, events produce risk logs, risk logs produce scores and reports.
- Training is not required for MediaPipe.
- YOLO training is optional future work, not required for the current demo.
