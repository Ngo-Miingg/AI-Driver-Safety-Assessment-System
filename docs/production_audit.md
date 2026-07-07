# Production Audit

## Scope

Audited areas: architecture, SQLite database, routes, templates, static assets, realtime SocketIO flow, upload flow, analysis flow, reports, settings, evaluation, and documentation.

## Strengths

- Flask app keeps routes, templates, static files, analysis helpers, and AI utilities separated.
- SQLite schema supports users, drivers, sessions, events, risk logs, reports, and app settings.
- Realtime monitoring has explicit start/stop SocketIO events, camera error handling, low-rate snapshots, and saved session finalization.
- Upload analysis supports metadata extraction, simulated analysis, MediaPipe analysis, and Full AI analysis with phone detector fallback.
- Reports, history, dashboard, settings, evaluation, and CSV export already use stored data.
- Socket.IO browser client is vendored locally for offline demos.
- Missing YOLO model is handled as a warning path instead of crashing the whole app.

## Weaknesses

- SQLite is suitable for local demo and small deployments, but not high-concurrency production use.
- Realtime processing runs in a background thread and is performance-dependent on CPU and camera backend.
- Phone usage currently uses ROI filtering and temporal confirmation, but face-phone proximity is not fully implemented because phone detector results do not include face geometry.
- Uploaded-video evidence snapshots are not implemented.
- End-to-end test coverage is mostly manual.
- Authentication is local/demo-oriented and lacks password reset, email verification, audit trails, and production session hardening.

## Production Risks

- Camera access can fail if another app owns the device or the OS blocks permissions.
- MediaPipe performance can vary strongly by CPU and lighting.
- Realtime YOLO/RT-DETR can reduce FPS and should remain optional.
- Large uploads can consume disk space quickly.
- The demo account should not be used in a public deployment.

## Technical Debt

- Full AI phone detection should use a single detector factory configuration everywhere.
- Driver ROI defaults to full-frame for compatibility; it should be calibrated per camera position.
- Risk scoring is rule-based and should be validated with labeled sessions.
- Database migration logic is pragmatic but not versioned with a formal migration tool.

## Recommended Production Path

1. Keep current Flask/SocketIO architecture for defense and local pilot.
2. Add automated route and analysis smoke tests.
3. Calibrate thresholds and driver ROI on the target camera.
4. Train or plug in a phone detector only after preparing a validated phone dataset.
5. Move to PostgreSQL and a production WSGI/SocketIO deployment only if multi-user scale is required.
