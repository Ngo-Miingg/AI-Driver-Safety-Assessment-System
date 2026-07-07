# AI Driver Safety Assessment System - Final System Design

## 1. Purpose

This document locks the final internal system design before implementation. It defines the feature map, user workflows, session lifecycle, event lifecycle, and report lifecycle.

This document does not define source code, Flask routes, HTML pages, SQL statements, or project folders.

## 2. Complete Feature Map

### 2.1 User and Driver Features

- User login.
- User logout.
- Protected access to dashboard, monitoring, upload, history, drivers, and reports.
- Multi-driver profile support.
- Driver creation.
- Driver selection before analysis.
- Driver history review.

### 2.2 Analysis Features

- Webcam realtime monitoring.
- Uploaded video analysis.
- OpenCV frame capture and preprocessing.
- MediaPipe face landmark detection.
- EAR-based eye closure analysis.
- MAR-based yawn analysis.
- Head pose analysis for distraction.
- YOLO11n/s phone detection.
- Temporal event tracking.
- Temporal risk scoring.
- Safety score calculation.
- Alert level calculation.
- Explainable risk breakdown.
- Fatigue trend forecast.

### 2.3 Dashboard and Output Features

- Dashboard summary.
- Realtime risk display.
- Event timeline.
- Session history.
- Driver history.
- Session detail.
- Rule-based AI driving report.
- Risk category breakdown.
- Safety score trend.
- Processing status and error messages.

### 2.4 Locked Scope

- The system is local-first.
- SQLite is the persistent database.
- No paid API dependency is allowed.
- Scoring and reporting are rule-based and explainable.
- Heavy research models and cloud inference are excluded.
- The implementation must remain realistic for a solo developer, RTX 3050 Laptop GPU 6GB, and 15-day timeline.

## 3. Final User Workflow

### 3.1 Workflow Summary

```text
Open application
  -> Login
  -> Dashboard
  -> Select or create driver
  -> Choose realtime monitoring or video upload
  -> Run analysis
  -> View timeline, scores, forecast, and report
  -> Review driver history
```

### 3.2 Input

- User login credentials.
- Selected driver profile or new driver information.
- Webcam stream or uploaded video.
- User action to start, stop, upload, or view results.

### 3.3 Processing

- Authenticate user.
- Load accessible drivers.
- Associate selected driver with analysis session.
- Run the selected media through the AI pipeline.
- Track temporal events.
- Calculate risk score, safety score, alert level, and fatigue trend.
- Generate a rule-based report.
- Store session, event, score, and report data.

### 3.4 Output

- Dashboard access.
- Active driver context.
- Realtime monitoring updates or uploaded video result.
- Event timeline.
- Safety score.
- Risk explanation.
- Fatigue forecast.
- AI driving report.
- Stored driver history.

## 4. Realtime Workflow

### 4.1 Workflow Summary

```text
Dashboard or realtime page
  -> Select driver
  -> Start webcam session
  -> OpenCV reads camera frames
  -> AI pipeline analyzes frames
  -> SocketIO emits live risk updates
  -> User stops session
  -> System finalizes scores and report
  -> History is saved
```

### 4.2 Input

- Authenticated user context.
- Selected driver identifier.
- Start monitoring command.
- Webcam frames.
- Stop monitoring command.

### 4.3 Processing

- Validate authenticated user and selected driver.
- Create an active realtime session.
- Initialize media capture, temporal trackers, risk engine, and timeline state.
- Read camera frames using OpenCV.
- Preprocess frames for MediaPipe and YOLO.
- Detect landmarks using MediaPipe.
- Detect phone presence using YOLO11n/s.
- Calculate EAR, MAR, and head pose.
- Convert frame-level signals into temporal events.
- Update rolling risk score and alert level.
- Emit compact realtime updates with Flask-SocketIO.
- On stop, close active events and finalize scores.
- Generate fatigue forecast and report.
- Save session history.

### 4.4 Output

- Active session status.
- Current risk score.
- Current safety estimate.
- Alert level.
- Active behavior states.
- New timeline events.
- Warnings such as no face detected or camera unavailable.
- Final session summary.
- Final safety score.
- Stored report.

## 5. Video Upload Workflow

### 5.1 Workflow Summary

```text
Upload page
  -> Select driver
  -> Upload video
  -> Validate video
  -> OpenCV reads frames
  -> AI pipeline analyzes sampled frames
  -> Risk engine finalizes scores
  -> Report is generated
  -> Result is shown and saved
```

### 5.2 Input

- Authenticated user context.
- Selected driver identifier.
- Uploaded video file.
- Upload metadata such as original filename and upload time.

### 5.3 Processing

- Validate user and driver.
- Validate that the file is an acceptable video.
- Create a video analysis session.
- Read video frames through OpenCV.
- Sample frames at a practical processing rate.
- Run the same AI pipeline used by realtime monitoring.
- Track events by video timestamp.
- Calculate risk score series, final safety score, alert summary, and fatigue forecast.
- Generate rule-based AI driving report.
- Store session, timeline, risk logs, forecast, and report data.

### 5.4 Output

- Upload status.
- Processing status.
- Analysis complete or failed status.
- Event timeline.
- Risk and safety scores.
- Fatigue forecast.
- Report page data.
- Stored history entry.

## 6. Driver Workflow

### 6.1 Workflow Summary

```text
Dashboard or driver page
  -> View driver list
  -> Create or select driver
  -> Use driver in realtime or upload session
  -> Review driver-specific history and reports
```

### 6.2 Input

- Authenticated user context.
- Driver name or display label.
- Optional driver metadata.
- Selected driver identifier.

### 6.3 Processing

- Retrieve drivers owned by or accessible to the authenticated user.
- Create new driver profile when requested.
- Update basic driver information when requested.
- Store selected driver context for the next analysis workflow.
- Filter history and reports by driver.

### 6.4 Output

- Driver list.
- Driver detail.
- Selected driver state.
- Driver-specific history.
- Driver-specific reports.

## 7. History Workflow

### 7.1 Workflow Summary

```text
History page
  -> Select driver or all accessible drivers
  -> Load completed sessions
  -> Show score and risk summaries
  -> Open session detail or report
```

### 7.2 Input

- Authenticated user context.
- Optional driver filter.
- Optional date filter.
- Optional session type filter.

### 7.3 Processing

- Retrieve matching sessions.
- Retrieve summary scores and dominant event categories.
- Prepare history rows.
- Prepare trend chart data.
- Link sessions to timelines and reports.

### 7.4 Output

- Session list.
- Safety score history.
- Risk category summaries.
- Links to event timeline and report.
- Driver history view.

## 8. Report Workflow

### 8.1 Workflow Summary

```text
Completed session
  -> Load session summary
  -> Load events and risk logs
  -> Load fatigue forecast
  -> Generate rule-based report
  -> Store report
  -> Display report
```

### 8.2 Input

- Completed session.
- Driver information.
- Event timeline.
- Risk score summary.
- Safety score.
- Alert level summary.
- Explainable risk breakdown.
- Fatigue trend forecast.

### 8.3 Processing

- Summarize session metadata.
- Explain final safety score.
- Identify dominant risk categories.
- Highlight important events.
- Summarize fatigue trend.
- Generate rule-based recommendations.
- Include limitations.
- Store report content or report source data.

### 8.4 Output

- AI driving report.
- Safety assessment.
- Risk explanation.
- Timeline summary.
- Fatigue forecast section.
- Recommendations.
- Limitations.

## 9. Session Lifecycle

### 9.1 Session States

- **Created**: session context exists but processing has not started.
- **Active**: realtime monitoring is running.
- **Processing**: uploaded video analysis is running.
- **Stopping**: stop requested and finalization is in progress.
- **Completed**: analysis, scoring, forecast, report, and storage are complete.
- **Failed**: analysis could not complete.
- **Canceled**: user canceled before completion.

### 9.2 Lifecycle Steps

1. User logs in.
2. User selects driver.
3. User starts realtime session or uploads video.
4. System creates session context.
5. System initializes AI pipeline and risk engine.
6. System processes frames.
7. System creates temporal events.
8. System updates risk logs.
9. User stops realtime session or video reaches end.
10. System closes active events.
11. System finalizes safety score and alert summary.
12. System calculates fatigue trend forecast.
13. System generates report.
14. System saves history.
15. System displays final result.

### 9.3 Session Finalization Rules

- A session must not be marked completed until events, scores, forecast, and report data are available.
- A failed session should keep enough status information to explain the failure.
- A realtime session must release the camera when stopped, failed, or canceled.
- Uploaded video processing must not create a successful report when the video cannot be analyzed.

## 10. Event Lifecycle

### 10.1 Event States

- **Candidate**: frame-level signal detected but not long enough to confirm.
- **Active**: event has passed minimum duration and is ongoing.
- **Closed**: signal ended and event duration is finalized.
- **Merged**: event was combined with a nearby event of the same type.
- **Discarded**: signal was too short or too unreliable.

### 10.2 Lifecycle Steps

1. Frame-level signal appears.
2. Temporal tracker opens candidate event.
3. Signal continues long enough to pass confirmation threshold.
4. Event becomes active.
5. Risk engine receives active event state.
6. SocketIO or UI receives event update where applicable.
7. Signal ends or timeout occurs.
8. Event is closed.
9. Nearby matching events may be merged.
10. Event severity is assigned.
11. Event is stored in the timeline.
12. Event contributes to final scoring and report.

### 10.3 Event Design Rules

- A brief blink must not become an eye closure event.
- A short glance must not become a distraction event.
- A single uncertain phone detection should not become a high-severity phone usage event.
- Face missing should be tracked as warning or risk depending on duration.
- Events must include start time, end time, duration, type, severity, and explanation.

## 11. Report Lifecycle

### 11.1 Report States

- **Pending**: session exists but report is not ready.
- **Generating**: report content is being created from session data.
- **Available**: report can be viewed.
- **Failed**: report generation failed.

### 11.2 Lifecycle Steps

1. Session completes.
2. System retrieves session summary.
3. System retrieves event timeline.
4. System retrieves score and alert data.
5. System retrieves fatigue forecast.
6. System creates explainable risk breakdown.
7. System generates rule-based report sections.
8. System stores report data.
9. System marks report available.
10. User views report from result, history, or reports page.

### 11.3 Report Design Rules

- Report generation must not use paid APIs.
- Report text must be derived from stored session data.
- Report must clearly state major risk contributors.
- Report must include analysis limitations.
- Report must avoid medical, legal, or insurance-grade claims.

