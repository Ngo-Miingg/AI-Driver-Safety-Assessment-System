# Database Schema

The app uses SQLite at `instance/mddad.sqlite3` by default.

## Tables

- `users`: login accounts, seeded demo user, role/status metadata.
- `drivers`: driver profiles owned by a user.
- `sessions`: uploaded-video and realtime-webcam sessions, analysis status, video metadata, final scores, MediaPipe metrics, and phone metrics.
- `events`: temporal behavior events such as eye closure, yawn, looking away, head down, face missing, and phone usage.
- `risk_logs`: rolling risk/safety/alert values over time.
- `reports`: generated explainable driving reports for sessions.
- `app_settings`: SQLite-backed threshold and calibration overrides.

## Migration Safety

Startup runs `mddad.database.migrate_schema()` after applying `schema.sql`.

Current protections:

- Creates `app_settings` if missing.
- Adds missing session columns used by the current app.
- Adds `events.evidence_image_path` if missing.
- Rebuilds child result tables if older migrations left foreign keys pointing at `sessions_old`.

## Startup Validation Checklist

1. Run `python app.py`.
2. Confirm `/health` returns database `connected`.
3. Confirm demo login works.
4. Create or select a driver.
5. Create a realtime or upload session and confirm it appears in History.

## Production Notes

SQLite is acceptable for local defense/demo use. For production with multiple concurrent users or long retention, migrate to PostgreSQL and use explicit migration versions.
