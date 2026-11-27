# Product Database Architecture

The project uses one local SQLite database as the source of truth for the
offline laptop/edge product. The database is intentionally split into stable
domain tables and reusable product-inference tables so new capabilities can be
added without creating a separate schema for each model.

## Core Domain Tables

| Table | Purpose |
| --- | --- |
| `users` | Local accounts and demo login |
| `drivers` | Driver profiles, status, and legacy cached face enrollment fields |
| `sessions` | Realtime or uploaded-video monitoring sessions |
| `events` | Confirmed safety events such as drowsiness |
| `risk_logs` | Time-series risk and safety scores |
| `reports` | Generated session reports |
| `app_settings` | Runtime thresholds and local configuration |

## Shared Product Tables

| Table | Purpose |
| --- | --- |
| `model_artifacts` | Versioned local model registry with backend key, role, path, SHA-256 and metadata |
| `media_assets` | Uploaded videos, enrollment images, evidence images, and future sensor assets |
| `driver_biometrics` | Active biometric enrollments such as face embeddings |
| `inference_runs` | One execution record per realtime/upload/model test run |
| `model_predictions` | Frame/clip-level predictions, probabilities, features and metadata |
| `session_models` | Models and thresholds used by each saved session |
| `app_audit_logs` | Important user/system actions for handoff and troubleshooting |

## Current Integration

```text
Driver enrollment
  -> media_assets(face_enrollment_image) for each valid enrollment sample
  -> driver_biometrics(face_embedding centroid, driver_safety_v1.face_verification)
  -> app_audit_logs(driver.face_enrolled)

Realtime monitoring
  -> sessions(realtime_webcam)
  -> session_models(drowsiness, face_verification)
  -> inference_runs(realtime_webcam)
  -> model_predictions(driver_safety_v1.drowsiness)
  -> events / risk_logs / reports

Uploaded video
  -> sessions(uploaded_video)
  -> media_assets(uploaded_video)
  -> app_audit_logs(video.uploaded)
  -> inference_runs(uploaded_video_analysis) when analysis is started
```

## Health Check

Use this endpoint after startup:

```text
/health/db
```

It returns table counts and the registered model artifacts without exposing
absolute local paths.

## Extension Rule

New capabilities should reuse the shared tables:

1. Add the model to `model_artifacts`.
2. Store uploaded/captured inputs in `media_assets`.
3. Create one `inference_runs` row for each execution.
4. Store model-level output in `model_predictions`.
5. Store confirmed user-facing incidents in `events`.
6. Link model usage to a session through `session_models`.

This keeps dashboard, history, reports, evaluation and future edge deployment
on one consistent data model.
