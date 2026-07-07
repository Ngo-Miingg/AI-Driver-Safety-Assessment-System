# AI Driver Safety Assessment System - Requirement Specification

## 1. Project Overview

The AI Driver Safety Assessment System is a graduation project focused on assessing driver safety from webcam streams and uploaded driving videos. The system shall detect signs of drowsiness, distraction, and phone usage, then convert detected events into temporal risk scores, safety scores, explainable analysis, fatigue trend forecasts, event timelines, driver history, and AI-generated driving reports.

The system is intended for demonstration and academic evaluation, not for deployment in real vehicles or for making legally binding safety decisions. It shall operate without paid API dependencies and shall be feasible for a solo developer within a 15-day development timeline using the available hardware and software constraints.

The expected system users include an authenticated driver or evaluator who can run realtime monitoring, upload video files for analysis, view dashboard summaries, inspect event timelines, compare driver history, and export or review driving safety reports.

## 2. Objectives

### Must Have

- Provide realtime webcam-based driver monitoring.
- Provide uploaded video analysis for prerecorded driving sessions.
- Detect drowsiness-related indicators such as eye closure, yawning, and head pose risk patterns.
- Detect distraction-related indicators such as looking away, unsafe head orientation, and inattentive posture.
- Detect phone usage as a distinct risk event.
- Calculate temporal risk scores over time instead of relying only on single-frame decisions.
- Calculate a safety score for each analyzed session.
- Provide explainable risk analysis describing why a session or time interval is considered risky.
- Show an event timeline of detected risky behaviors.
- Store driver history for later review.
- Support user login and multi-driver usage.
- Provide a dashboard for summaries, charts, recent sessions, and risk insights.
- Generate an AI driving report from session analysis results.
- Remain feasible for a solo developer to complete in 15 days.

### Should Have

- Forecast fatigue trends using recent temporal indicators from the session.
- Support side-by-side comparison of recent driver sessions.
- Provide confidence or severity levels for detected events where technically reasonable.
- Allow filtering timeline events by event type and severity.
- Provide clear demo flows for realtime webcam and video upload modes.

### Nice to Have

- Provide basic recommendations for safer driving based on detected risk patterns.
- Support report export in a simple downloadable format.
- Provide basic visual overlays for face landmarks, bounding boxes, and risk states during analysis.
- Support configurable threshold values through an internal settings view.

### Out of Scope

- Real vehicle integration.
- Mobile application development.
- Cloud deployment as a required deliverable.
- Paid AI, cloud vision, or commercial API dependencies.
- Legal certification, insurance scoring, or production safety approval.
- Identity verification, face recognition, or biometric authentication.
- Advanced fleet management features.

## 3. User Roles

### Must Have

- **Unauthenticated User**: Can access only login-related screens and cannot view driver data or run analysis.
- **Authenticated User**: Can log in, manage accessible driver profiles, run realtime monitoring, upload videos, view dashboards, inspect history, and generate reports.
- **Driver Profile**: Represents a driver being assessed. A single authenticated user may manage multiple driver profiles.

### Should Have

- **Evaluator/Demo Operator**: A practical role fulfilled by an authenticated user during project demonstration, able to select drivers and run demo sessions.

### Nice to Have

- **Admin-like User**: Can manage users and driver profiles separately from normal driver analysis workflows.

### Out of Scope

- Complex role-based access control with many permission levels.
- Organization, company, or fleet hierarchy management.

## 4. Functional Requirements

### Must Have

- FR-001: The system shall allow users to log in before accessing dashboard and analysis features.
- FR-002: The system shall allow an authenticated user to create, select, view, and update driver profiles.
- FR-003: The system shall support multiple driver profiles under the same application.
- FR-004: The system shall provide realtime monitoring from a webcam.
- FR-005: The system shall provide video upload analysis for common video formats suitable for demo use.
- FR-006: The system shall process video frames to detect driver face and relevant visual indicators.
- FR-007: The system shall detect drowsiness events based on observable indicators such as prolonged eye closure, yawning, and fatigue-related head movement.
- FR-008: The system shall detect distraction events based on attention direction, head pose, and gaze-related indicators where available.
- FR-009: The system shall detect phone usage using an object detection model.
- FR-010: The system shall assign timestamps to detected events.
- FR-011: The system shall calculate per-session temporal risk scores.
- FR-012: The system shall calculate an overall safety score for each completed session.
- FR-013: The system shall display an event timeline for each session.
- FR-014: The system shall store analyzed session summaries and event history.
- FR-015: The system shall display driver history across multiple sessions.
- FR-016: The system shall provide explainable risk analysis for important detected risks.
- FR-017: The system shall generate an AI driving report after a session is analyzed.
- FR-018: The system shall provide a dashboard that summarizes driver safety status and historical trends.
- FR-019: The system shall support both webcam sessions and uploaded video sessions in the same application.
- FR-020: The system shall clearly indicate whether a session is currently running, completed, failed, or stopped.
- FR-021: The system shall allow users to start and stop realtime monitoring.
- FR-022: The system shall show analysis results without requiring paid external services.

### Should Have

- FR-023: The system should display live risk state updates during webcam monitoring.
- FR-024: The system should display processed frame previews or overlays during realtime monitoring.
- FR-025: The system should support pause, resume, or cancel behavior for uploaded video analysis if feasible.
- FR-026: The system should allow users to filter history by driver and date range.
- FR-027: The system should show detected event severity as low, medium, or high.
- FR-028: The system should group repeated frame-level detections into meaningful temporal events.
- FR-029: The system should prevent duplicate or noisy events from overwhelming the timeline.
- FR-030: The system should provide user-readable explanations for safety score changes.
- FR-031: The system should provide a fatigue trend forecast for the current or most recent session.
- FR-032: The system should allow users to view report details from previous sessions.

### Nice to Have

- FR-033: The system may provide downloadable reports.
- FR-034: The system may allow manual notes to be attached to a driver session.
- FR-035: The system may include demo sample videos for controlled evaluation.
- FR-036: The system may include a simple calibration step for camera position and face visibility.
- FR-037: The system may allow users to configure risk thresholds for demonstration purposes.

### Out of Scope

- FR-038: The system shall not provide live intervention or vehicle control.
- FR-039: The system shall not make medical diagnoses.
- FR-040: The system shall not guarantee detection accuracy under all lighting, camera, driver, or vehicle conditions.
- FR-041: The system shall not require cloud-hosted AI inference.

## 5. Non-Functional Requirements

### Must Have

- NFR-001: The system shall run locally using Flask, Flask-SocketIO, HTML, CSS, JavaScript, SQLite, OpenCV, YOLO11n or YOLO11s, and MediaPipe.
- NFR-002: The system shall avoid paid API dependencies.
- NFR-003: The system shall be feasible on an RTX 3050 Laptop GPU with 6GB VRAM for local development and demo inference.
- NFR-004: The system shall be compatible with Kaggle T4 for model training or fine-tuning activities.
- NFR-005: The system shall provide acceptable responsiveness for realtime demo usage.
- NFR-006: The system shall store data locally in SQLite.
- NFR-007: The system shall handle common user errors gracefully, including missing webcam, invalid upload, failed analysis, and no face detected.
- NFR-008: The system shall provide clear UI feedback for long-running analysis operations.
- NFR-009: The system shall protect user sessions from casual unauthorized access through login.
- NFR-010: The system shall be maintainable by a solo developer during the project timeline.

### Should Have

- NFR-011: The system should maintain a stable realtime monitoring experience under normal webcam resolution settings.
- NFR-012: The system should degrade gracefully when processing is slower than the video frame rate.
- NFR-013: The system should keep the UI responsive while analysis is running.
- NFR-014: The system should use consistent visual styling suitable for an academic demo.
- NFR-015: The system should keep stored session data compact enough for repeated testing.

### Nice to Have

- NFR-016: The system may support GPU acceleration when available and CPU fallback when required.
- NFR-017: The system may provide basic application logs for debugging and demo verification.
- NFR-018: The system may support configurable frame sampling for performance tuning.

### Out of Scope

- NFR-019: High-availability production deployment.
- NFR-020: Enterprise security compliance.
- NFR-021: Large-scale concurrent user support.
- NFR-022: Real-time guarantees required for safety-critical automotive systems.

## 6. System Inputs

### Must Have

- Webcam video stream from a local camera.
- Uploaded driving video files.
- User login credentials.
- Driver profile information.
- Session metadata such as driver, session type, start time, and end time.
- Visual frame data used for drowsiness, distraction, and phone usage detection.

### Should Have

- Optional session notes or labels for evaluation.
- Optional analysis configuration such as selected driver and analysis mode.
- Optional demo video samples for repeatable testing.

### Nice to Have

- Optional camera calibration information.
- Optional risk threshold settings for controlled demonstration.

### Out of Scope

- Vehicle CAN bus data.
- GPS, speed, steering, braking, or lane detection sensor data.
- Audio analysis.
- Wearable sensor data.
- External identity provider data.

## 7. System Outputs

### Must Have

- Realtime risk state during webcam monitoring.
- Detected drowsiness events.
- Detected distraction events.
- Detected phone usage events.
- Temporal risk score.
- Overall safety score.
- Event timeline with timestamps.
- Driver session history.
- Explainable risk analysis.
- Fatigue trend forecast result.
- AI driving report.
- Dashboard metrics and visual summaries.

### Should Have

- Event severity level.
- Event duration.
- Detection confidence where available.
- Session status and processing progress.
- Risk score trend chart.
- Summary of most frequent risk behaviors.

### Nice to Have

- Downloadable report file.
- Annotated preview image or frame snapshot for important events.
- Demo summary page for presentation use.

### Out of Scope

- Legally certified driver score.
- Insurance-grade risk profile.
- Medical fatigue diagnosis.
- Production-grade audit log.

## 8. Dashboard Requirements

### Must Have

- DR-001: The dashboard shall show the selected driver or allow selecting a driver profile.
- DR-002: The dashboard shall show recent safety score information.
- DR-003: The dashboard shall show recent risk events grouped by type.
- DR-004: The dashboard shall show session history.
- DR-005: The dashboard shall provide access to realtime webcam monitoring.
- DR-006: The dashboard shall provide access to video upload analysis.
- DR-007: The dashboard shall provide access to generated reports.
- DR-008: The dashboard shall show at least one trend visualization related to risk or safety over time.

### Should Have

- DR-009: The dashboard should show summary cards for total sessions, average safety score, drowsiness count, distraction count, and phone usage count.
- DR-010: The dashboard should show a chart of safety score changes across recent sessions.
- DR-011: The dashboard should show a breakdown of risk categories.
- DR-012: The dashboard should support filtering by driver and date range.
- DR-013: The dashboard should highlight the most recent high-risk session.

### Nice to Have

- DR-014: The dashboard may include a compact recommendation panel.
- DR-015: The dashboard may include visual comparison between two sessions.
- DR-016: The dashboard may include a demo mode summary for graduation presentation.

### Out of Scope

- Fleet-wide executive analytics.
- Real-time map or route visualization.
- Business intelligence integrations.

## 9. AI Analysis Requirements

### Must Have

- AIR-001: The system shall use computer vision methods to analyze driver behavior from video frames.
- AIR-002: The system shall use MediaPipe or equivalent local landmark processing for face-related indicators.
- AIR-003: The system shall use YOLO11n or YOLO11s for object detection where suitable, especially phone usage detection.
- AIR-004: The system shall identify drowsiness indicators from observable facial and head-related cues.
- AIR-005: The system shall identify distraction indicators from attention and head orientation cues.
- AIR-006: The system shall identify phone usage as a separate risk category.
- AIR-007: The system shall aggregate frame-level detections into temporal events.
- AIR-008: The system shall calculate risk over time rather than treating each frame independently.
- AIR-009: The system shall produce an explainable risk summary for each analyzed session.
- AIR-010: The system shall produce a fatigue trend forecast from recent session behavior.
- AIR-011: The system shall expose analysis limitations in the report or UI.

### Should Have

- AIR-012: The system should reduce false positives by requiring temporal consistency for risky states.
- AIR-013: The system should distinguish short glances from sustained distraction.
- AIR-014: The system should distinguish brief blinks from prolonged eye closure.
- AIR-015: The system should report which risk categories contributed most to the safety score.
- AIR-016: The system should retain enough event evidence to explain major risk periods.
- AIR-017: The system should allow model performance to be evaluated using prepared test videos.

### Nice to Have

- AIR-018: The system may use simple smoothing or trend analysis for fatigue forecasting.
- AIR-019: The system may classify events into severity levels using duration and frequency.
- AIR-020: The system may include annotated frames for high-risk events.

### Out of Scope

- End-to-end autonomous driving perception.
- Emotion recognition as a required feature.
- Biometric identity recognition.
- Cloud-based large language model analysis.
- Guaranteed medical fatigue assessment.

## 10. Database Requirements

### Must Have

- DBR-001: The system shall use SQLite for local persistence.
- DBR-002: The system shall store user account information required for login.
- DBR-003: The system shall store driver profile information.
- DBR-004: The system shall store session metadata.
- DBR-005: The system shall store session-level safety scores and risk summaries.
- DBR-006: The system shall store detected event records with timestamps and event types.
- DBR-007: The system shall store enough historical data to show driver history and dashboard trends.
- DBR-008: The system shall associate sessions and events with the correct driver profile.
- DBR-009: The system shall support retrieval of previous reports or report data.

### Should Have

- DBR-010: The system should store event severity and duration.
- DBR-011: The system should store analysis status for uploaded videos.
- DBR-012: The system should store created and updated timestamps for important records.
- DBR-013: The system should avoid storing unnecessary raw frame data by default.

### Nice to Have

- DBR-014: The system may store representative event snapshots if storage remains manageable.
- DBR-015: The system may support simple database reset or seed data behavior for demo preparation.

### Out of Scope

- Detailed database schema in this requirements document.
- Distributed database systems.
- Cloud-hosted database requirement.
- Long-term archival storage.

## 11. Realtime Requirements

### Must Have

- RR-001: The system shall support realtime webcam monitoring through the web application.
- RR-002: The system shall use Flask-SocketIO or equivalent realtime communication within the stated technology constraints.
- RR-003: The system shall stream or update analysis results to the browser during active monitoring.
- RR-004: The system shall show current risk state during realtime monitoring.
- RR-005: The system shall allow the user to start and stop monitoring.
- RR-006: The system shall record a session summary after monitoring ends.
- RR-007: The system shall tolerate temporary no-face or low-confidence periods without crashing.

### Should Have

- RR-008: The system should maintain a smooth enough preview for demonstration on the target laptop.
- RR-009: The system should avoid blocking the UI while analysis is running.
- RR-010: The system should throttle or sample frames when needed to keep latency acceptable.
- RR-011: The system should update timeline events during or immediately after realtime monitoring.
- RR-012: The system should display processing status and warnings such as no camera, poor lighting, or face not visible.

### Nice to Have

- RR-013: The system may display live annotated overlays.
- RR-014: The system may show live rolling charts for risk score or fatigue indicators.

### Out of Scope

- Safety-critical hard realtime performance.
- Multi-camera realtime monitoring.
- Remote streaming across public networks.

## 12. Evaluation Requirements

### Must Have

- ER-001: The project shall include evaluation of drowsiness detection behavior on selected test cases.
- ER-002: The project shall include evaluation of distraction detection behavior on selected test cases.
- ER-003: The project shall include evaluation of phone usage detection behavior on selected test cases.
- ER-004: The project shall include evaluation using uploaded videos and realtime webcam scenarios.
- ER-005: The project shall report measurable results suitable for a graduation project defense.
- ER-006: The evaluation shall include examples of correct detections and known failure cases.
- ER-007: The evaluation shall consider feasibility on the RTX 3050 Laptop GPU and Kaggle T4 training environment.

### Should Have

- ER-008: The evaluation should include accuracy-oriented metrics where labeled test data is available.
- ER-009: The evaluation should include precision, recall, or F1-score for phone detection if annotated data is available.
- ER-010: The evaluation should include latency or throughput observations for realtime mode.
- ER-011: The evaluation should include qualitative analysis of explainable reports.
- ER-012: The evaluation should compare risk scores across safe and risky sample sessions.

### Nice to Have

- ER-013: The evaluation may include a small confusion matrix for classified event types.
- ER-014: The evaluation may include before-and-after results for model fine-tuning.
- ER-015: The evaluation may include user feedback from a small demo group.

### Out of Scope

- Large-scale clinical validation.
- Road deployment evaluation.
- Government or industry certification testing.
- Exhaustive benchmark against commercial driver monitoring systems.

## 13. Demo Requirements

### Must Have

- DER-001: The demo shall show user login.
- DER-002: The demo shall show selecting or creating a driver profile.
- DER-003: The demo shall show realtime webcam monitoring.
- DER-004: The demo shall show video upload analysis.
- DER-005: The demo shall show drowsiness detection output.
- DER-006: The demo shall show distraction detection output.
- DER-007: The demo shall show phone usage detection output.
- DER-008: The demo shall show temporal risk scoring.
- DER-009: The demo shall show a safety score.
- DER-010: The demo shall show an event timeline.
- DER-011: The demo shall show driver history.
- DER-012: The demo shall show dashboard summaries.
- DER-013: The demo shall show an AI driving report.
- DER-014: The demo shall work without paid external services.

### Should Have

- DER-015: The demo should include prepared sample videos for predictable results.
- DER-016: The demo should include at least one normal driving sample and one risky driving sample.
- DER-017: The demo should show explanations for why risk score increased.
- DER-018: The demo should show fatigue trend forecast behavior.
- DER-019: The demo should include fallback messaging if the webcam is unavailable.

### Nice to Have

- DER-020: The demo may include an annotated visual overlay.
- DER-021: The demo may include downloadable report output.
- DER-022: The demo may include a compact presentation mode.

### Out of Scope

- Live vehicle demonstration.
- Multi-user live concurrent demonstration.
- Public internet deployment demonstration.

## 14. Project Constraints

### Must Have

- PC-001: The project shall be developed by a solo developer.
- PC-002: The project shall be planned for a 15-day development timeline.
- PC-003: The local development and demo hardware shall account for an RTX 3050 Laptop GPU with 6GB VRAM.
- PC-004: Kaggle T4 may be used for model training or fine-tuning.
- PC-005: The backend shall use Flask and Flask-SocketIO.
- PC-006: The frontend shall use HTML, CSS, and JavaScript.
- PC-007: The database shall use SQLite.
- PC-008: Computer vision processing shall use OpenCV.
- PC-009: Object detection shall use YOLO11n or YOLO11s.
- PC-010: Landmark-based face analysis shall use MediaPipe where suitable.
- PC-011: The system shall not depend on paid APIs.
- PC-012: The project scope shall prioritize demonstrable academic value over production completeness.

### Should Have

- PC-013: The implementation plan should favor features that can be completed and demonstrated reliably within 15 days.
- PC-014: Model choices should favor lightweight inference over maximum accuracy.
- PC-015: Data storage should remain simple and local.
- PC-016: The user interface should prioritize clear demonstration workflows.

### Nice to Have

- PC-017: The project may include optional fine-tuning if time permits.
- PC-018: The project may include additional visual polish after core requirements are complete.

### Out of Scope

- Paid hosting or managed database services.
- Heavy models that cannot run acceptably on the target laptop.
- Large team workflows or enterprise project management assumptions.
- Production deployment as a mandatory requirement.

## 15. Success Criteria

### Must Have

- SC-001: A user can log in and access the application dashboard.
- SC-002: A user can manage or select at least one driver profile.
- SC-003: A user can run realtime webcam monitoring.
- SC-004: A user can upload a video and receive analysis results.
- SC-005: The system detects and records drowsiness-related events in test scenarios.
- SC-006: The system detects and records distraction-related events in test scenarios.
- SC-007: The system detects and records phone usage events in test scenarios.
- SC-008: The system produces temporal risk scores for analyzed sessions.
- SC-009: The system produces an overall safety score for analyzed sessions.
- SC-010: The system shows an event timeline with timestamps.
- SC-011: The system stores and displays driver history.
- SC-012: The system provides an explainable analysis of major risk factors.
- SC-013: The system provides a fatigue trend forecast.
- SC-014: The system generates an AI driving report.
- SC-015: The system can be demonstrated locally without paid external services.
- SC-016: The final system scope remains achievable within the solo 15-day timeline.

### Should Have

- SC-017: Realtime monitoring remains responsive enough for a classroom or defense demo.
- SC-018: Uploaded video analysis completes within a reasonable time for short demo videos.
- SC-019: Dashboard charts and summaries clearly communicate driver safety status.
- SC-020: Evaluation results include both quantitative and qualitative observations.
- SC-021: The demo includes prepared examples for normal, drowsy, distracted, and phone usage cases.

### Nice to Have

- SC-022: The system exports or downloads a report.
- SC-023: The system displays annotated overlays for selected analysis modes.
- SC-024: The system includes configurable demo thresholds or settings.

### Out of Scope

- The project is not considered unsuccessful solely because it does not support production deployment.
- The project is not required to achieve commercial driver monitoring accuracy.
- The project is not required to support all possible camera angles, lighting conditions, occlusions, or driver appearances.
- The project is not required to provide legal, medical, or insurance-grade conclusions.

