# Evaluation Plan

This plan describes how to evaluate the project for a graduation defense without requiring model training.

## 1. Functional Evaluation

Verify that each expected workflow works end to end:

- Login/logout.
- Driver creation and active driver selection.
- Uploaded video creation.
- OpenCV metadata extraction.
- Simulated analysis.
- MediaPipe analysis.
- Full AI Analysis with YOLO model.
- Full AI Analysis without YOLO model.
- Realtime MediaPipe monitoring.
- Optional realtime YOLO monitoring.
- Snapshot preview.
- Event evidence snapshots.
- Dashboard, history, reports, settings, evaluation, and CSV export.

Result format:

| Feature | Expected Result | Actual Result | Pass/Fail | Notes |
|---|---|---|---|---|
| Realtime MediaPipe | Session saved after stop |  |  |  |
| Full AI missing YOLO | Warning, no crash |  |  |  |

## 2. Qualitative Evaluation

Use prepared videos and live webcam samples to judge explainability:

- Are event labels understandable?
- Does the timeline match visible behavior?
- Do evidence snapshots help explain the event?
- Are recommendations reasonable?
- Are warnings clear when YOLO or face detection fails?
- Can a reviewer follow how risk becomes safety score and report?

Suggested rating:

| Session | Timeline Clarity | Report Clarity | Evidence Usefulness | Notes |
|---|---:|---:|---:|---|
| Normal driver | 1-5 | 1-5 | 1-5 |  |
| Phone usage | 1-5 | 1-5 | 1-5 |  |

## 3. Simple Quantitative Metrics

Use the Evaluation page and CSV export.

Track:

- Total sessions.
- Safety score.
- Risk score.
- Alert level.
- Event count.
- Event count by type.
- Processed frame count.
- Face detected frame count.
- Face detection rate.
- Approximate processing FPS.
- Phone usage count.

## 4. FPS / Latency Observations

For realtime monitoring:

- Record `REALTIME_TARGET_FPS`.
- Record whether snapshot preview is enabled.
- Record whether realtime YOLO is enabled.
- Observe whether UI updates remain smooth.
- Note CPU load qualitatively if no profiler is available.

Example:

| Mode | Target FPS | Snapshot FPS | Realtime YOLO FPS | Observed Behavior |
|---|---:|---:|---:|---|
| MediaPipe only | 5 | 1 | 0 | Smooth |
| MediaPipe + YOLO | 5 | 1 | 1 | Acceptable / slow |

## 5. Face Detection Rate

Face detection rate is:

```text
face_detected_frame_count / processed_frame_count
```

Interpretation:

- High rate: camera position and lighting are suitable.
- Low rate: face may be hidden, too dark, too far, or poorly framed.

Use this metric when explaining false negatives or weak drowsiness results.

## 6. Event Counts

Compare expected behavior to generated event counts:

| Video | Expected Main Event | Eye | Yawn | Away | Down | Phone | Missing |
|---|---|---:|---:|---:|---:|---:|---:|
| Normal | None | 0 | 0 | 0 | 0 | 0 | 0 |
| Yawning | Yawn | 0 | >0 | 0 | 0 | 0 | 0 |
| Phone | Phone usage | 0 | 0 | 0 | 0 | >0 | 0 |

## 7. Safety Score Comparison

Run the same set of videos and compare final scores:

| Session | Scenario | Safety Score | Risk Score | Alert | Notes |
|---|---|---:|---:|---|---|
| 1 | Normal |  |  |  |  |
| 2 | Drowsiness |  |  |  |  |
| 3 | Phone |  |  |  |  |
| 4 | Mixed risk |  |  |  |  |

Expected pattern:

- Normal should score highest.
- Mixed-risk should score lower.
- Phone/drowsiness videos should show relevant event categories.

## 8. Failure Case Analysis

Document controlled failure cases:

- Camera unavailable.
- Missing YOLO model.
- Missing uploaded video file.
- No face detected.
- Poor lighting.
- Invalid upload file.

For each case:

| Failure Case | Expected Behavior | Actual Behavior | Pass/Fail |
|---|---|---|---|
| Missing YOLO | Warning and fallback |  |  |
| Camera unavailable | Realtime error, no crash |  |  |

## 9. Example Final Evaluation Table

| Test ID | Scenario | Mode | Duration | FPS | Face Rate | Events | Safety | Risk | Result |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| T01 | Normal | MediaPipe | 45s | 5 | 95% | 0 | 98 | 0 | Pass |
| T02 | Yawn | MediaPipe | 50s | 5 | 92% | 2 | 86 | 25 | Pass |
| T03 | Phone | Full AI | 40s | 5 | 90% | 1 | 78 | 40 | Pass |
| T04 | Mixed | Full AI | 70s | 5 | 88% | 5 | 62 | 70 | Pass |

Use actual exported CSV values when filling the final table.
