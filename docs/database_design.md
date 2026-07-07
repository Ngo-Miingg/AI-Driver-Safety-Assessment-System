# AI Driver Safety Assessment System - Database Design

## 1. Purpose

This document defines the complete database design before coding. It describes conceptual entities, fields, relationships, ownership, retention, and history requirements.

This document does not include SQL statements or exact database schema syntax.

## 2. Database Design Principles

- SQLite is the only required database.
- Data model must support user login, multi-driver support, sessions, events, risk logs, reports, and history.
- Every analysis session must belong to one driver.
- Every driver must be owned by or associated with one user.
- Stored data should support dashboard and report views without rerunning analysis.
- Raw frames and full videos should not be stored in the database.
- Keep the model simple enough for a solo 15-day implementation.

## 3. Critical, Optional, and Later Tables

### Critical Tables

- Users
- Drivers
- Sessions
- Events
- Reports

These are required for core login, multi-driver support, history, timeline, and reporting.

### Strongly Recommended Table

- RiskLogs

RiskLogs are strongly recommended because they make realtime charts, score timelines, explainable reports, and evaluation easier.

### Can Be Added Later

- Settings
- EventSnapshots
- EvaluationLabels
- UploadedFiles

These can improve polish, evaluation, or configurability but are not required for the first implementation.

## 4. Users

### Purpose

Store local user account information for login and ownership of driver profiles.

### Required Fields

- user identifier
- username or login name
- password verification data
- created timestamp
- account status

### Optional Fields

- display name
- email
- last login timestamp
- role label

### Relationships

- One user can own many drivers.
- One user can create many sessions through owned or accessible drivers.
- One user can generate many reports indirectly through sessions.

### Retention Rules

- Keep user records while the project database is active.
- Do not delete users automatically.
- If a user is removed manually, related driver and session ownership must be handled intentionally.

### Ownership Rules

- A user owns the driver profiles they create.
- A user can only access drivers and sessions associated with their account unless a later sharing feature is added.

### History Requirements

- User data must support filtering dashboard and history to the authenticated user's accessible drivers and sessions.
- User history does not need advanced audit trails.

## 5. Drivers

### Purpose

Represent individual driver profiles being assessed.

### Required Fields

- driver identifier
- owner user identifier
- driver display name
- created timestamp
- active status

### Optional Fields

- description
- age group or demo label
- notes
- updated timestamp

### Relationships

- Many drivers belong to one user.
- One driver can have many sessions.
- One driver can have many events through sessions.
- One driver can have many reports through sessions.

### Retention Rules

- Keep driver records while their history is useful for dashboard and evaluation.
- Deactivation is preferred over deletion when sessions already exist.
- Deleting a driver should not silently orphan historical sessions.

### Ownership Rules

- A driver belongs to the user who created the profile.
- Only the owner should create sessions for that driver in the base implementation.

### History Requirements

- Driver history must show completed sessions, safety scores, risk summaries, timelines, and reports associated with the driver.
- Driver history must support comparing recent sessions conceptually.

## 6. Sessions

### Purpose

Represent one realtime monitoring run or one uploaded video analysis run.

### Required Fields

- session identifier
- driver identifier
- user identifier
- session type: realtime or upload
- status
- start timestamp
- end timestamp
- duration
- final safety score
- final risk score
- final alert level or alert summary
- created timestamp

### Optional Fields

- original video filename
- video duration
- processed frame count
- average processing FPS
- failure reason
- notes
- report status
- fatigue trend label
- updated timestamp

### Relationships

- Many sessions belong to one driver.
- Many sessions belong to one user through driver ownership and session creator.
- One session has many events.
- One session has many risk logs.
- One session has one report in the base implementation.

### Retention Rules

- Keep completed sessions for driver history.
- Keep failed sessions only if they are useful for debugging or demo explanation.
- Do not store raw video content in the database.

### Ownership Rules

- A session belongs to the user who started or uploaded it.
- A session must belong to exactly one driver.
- Users should only view sessions they own or are allowed to access.

### History Requirements

- Sessions are the main history unit.
- History views must show session type, time, duration, safety score, final risk, alert summary, and dominant event types.
- Dashboard summaries must be derivable from sessions and related events.

## 7. Events

### Purpose

Store confirmed temporal events detected during a session.

### Required Fields

- event identifier
- session identifier
- event type
- start time within session
- end time within session
- duration
- severity
- risk contribution
- explanation text or evidence summary
- created timestamp

### Optional Fields

- confidence score
- frame index start
- frame index end
- peak risk during event
- related metric value such as EAR, MAR, or head pose category
- bounding box summary for phone detection

### Relationships

- Many events belong to one session.
- Events contribute to one session's risk and safety scores.
- Events are used by reports and dashboard summaries.

### Retention Rules

- Keep events for all completed sessions.
- Events should be compact and should not include raw frames by default.
- Optional snapshots can be added later outside the critical database design.

### Ownership Rules

- Event ownership is inherited from the session.
- Users can access events only through sessions they can access.

### History Requirements

- Events must support event timeline display.
- Events must support risk category breakdown.
- Events must support report explanation.
- Events must support counts by event type and severity.

## 8. RiskLogs

### Purpose

Store time-based risk score information for charts, realtime playback summaries, reports, and evaluation.

### Required Fields

- risk log identifier
- session identifier
- timestamp within session
- risk score
- estimated safety score
- alert level

### Optional Fields

- drowsiness contribution
- distraction contribution
- phone contribution
- face missing contribution
- active event types
- explanation summary

### Relationships

- Many risk logs belong to one session.
- Risk logs support dashboard charts and report explanations.
- Risk logs are derived from events and active behavior state.

### Retention Rules

- Store risk logs at a reasonable interval, not necessarily every processed frame.
- For realtime and upload, one log per second is sufficient for the base implementation.
- Risk logs may be downsampled later if storage grows.

### Ownership Rules

- Risk log ownership is inherited from the session.
- Users can access risk logs only through sessions they can access.

### History Requirements

- RiskLogs should support safety and risk trend charts.
- RiskLogs should support identifying high-risk periods in reports.
- RiskLogs should make evaluation screenshots and defense explanation easier.

## 9. Reports

### Purpose

Store generated rule-based AI driving report content or report source data for completed sessions.

### Required Fields

- report identifier
- session identifier
- generated timestamp
- report status
- report summary
- safety score explanation
- key risk explanation
- fatigue forecast explanation
- recommendations
- limitations

### Optional Fields

- report title
- report version
- dominant risk breakdown text
- timeline summary text
- export file path if export is added later

### Relationships

- One report belongs to one session.
- One session has one primary report in the base implementation.
- Reports are associated with drivers and users through sessions.

### Retention Rules

- Keep reports for completed sessions.
- If report rules change, either regenerate report or store report version.
- Reports should be small text records, not binary files by default.

### Ownership Rules

- Report ownership is inherited from the session.
- Users can view reports only for sessions they can access.

### History Requirements

- Reports must be accessible from session history.
- Reports must summarize event timeline, safety score, risk contributors, forecast, and limitations.

## 10. Optional Later Entities

### 10.1 Settings

Purpose:

- Store configurable thresholds or demo settings.

Can be added later because:

- Initial implementation can use constants or local configuration.

### 10.2 EventSnapshots

Purpose:

- Store optional image references for important events.

Can be added later because:

- Core timeline and report can work without snapshots.
- Storing images increases storage and implementation complexity.

### 10.3 EvaluationLabels

Purpose:

- Store manual labels for test videos and compare detections.

Can be added later because:

- Basic project evaluation can be documented manually from sessions and events.

### 10.4 UploadedFiles

Purpose:

- Store upload metadata separately from sessions.

Can be added later because:

- Base implementation can store minimal upload metadata in Sessions.

## 11. Cross-Entity Rules

- Users own Drivers.
- Drivers own Sessions.
- Sessions own Events, RiskLogs, and Reports.
- Events and RiskLogs must never exist without a parent Session.
- Reports must never exist without a completed or reportable Session.
- Session history must be queryable by driver and user.
- Dashboard summaries must be derivable from Sessions, Events, RiskLogs, and Reports.

## 12. Implementation Readiness

The database design is ready for implementation after converting these conceptual entities into SQLite tables. SQL generation is intentionally deferred to the coding phase.

