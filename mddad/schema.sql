CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    email TEXT,
    role TEXT NOT NULL DEFAULT 'user',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    last_login_at TEXT
);

CREATE TABLE IF NOT EXISTS drivers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT,
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    face_embedding_json TEXT,
    face_enrolled_at TEXT,
    face_enrollment_image_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    session_type TEXT NOT NULL CHECK (session_type IN ('realtime_webcam', 'uploaded_video')),
    status TEXT NOT NULL DEFAULT 'created',
    started_at TEXT,
    ended_at TEXT,
    duration_seconds REAL DEFAULT 0,
    original_video_filename TEXT,
    saved_video_path TEXT,
    file_size_bytes INTEGER,
    video_fps REAL,
    video_frame_count INTEGER,
    video_duration_seconds REAL,
    video_width INTEGER,
    video_height INTEGER,
    processed_frame_count INTEGER DEFAULT 0,
    average_processing_fps REAL,
    analysis_mode TEXT,
    multimodal_model_backend TEXT,
    multimodal_model_path TEXT,
    multimodal_clip_count INTEGER DEFAULT 0,
    audio_track_detected INTEGER DEFAULT 0,
    skeleton_frame_count INTEGER DEFAULT 0,
    face_detected_frame_count INTEGER DEFAULT 0,
    face_missing_frame_count INTEGER DEFAULT 0,
    average_ear REAL,
    average_mar REAL,
    min_ear REAL,
    max_mar REAL,
    phone_detected_frame_count INTEGER DEFAULT 0,
    max_phone_confidence REAL,
    phone_usage_event_count INTEGER DEFAULT 0,
    phone_detection_error TEXT,
    final_risk_score REAL,
    final_safety_score REAL,
    final_alert_level TEXT,
    fatigue_trend_label TEXT,
    analysis_error TEXT,
    failure_reason TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (driver_id) REFERENCES drivers (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    start_time_seconds REAL NOT NULL,
    end_time_seconds REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    confidence REAL,
    risk_contribution REAL NOT NULL DEFAULT 0,
    explanation TEXT,
    frame_start INTEGER,
    frame_end INTEGER,
    peak_risk_score REAL,
    metric_summary TEXT,
    evidence_image_path TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS risk_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    risk_score REAL NOT NULL,
    estimated_safety_score REAL,
    alert_level TEXT NOT NULL,
    drowsiness_contribution REAL DEFAULT 0,
    distraction_contribution REAL DEFAULT 0,
    phone_contribution REAL DEFAULT 0,
    face_missing_contribution REAL DEFAULT 0,
    active_event_types TEXT,
    explanation_summary TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending',
    title TEXT,
    summary TEXT,
    safety_score_explanation TEXT,
    key_risk_explanation TEXT,
    fatigue_forecast_explanation TEXT,
    recommendations TEXT,
    limitations TEXT,
    dominant_risk_breakdown TEXT,
    timeline_summary TEXT,
    report_version TEXT NOT NULL DEFAULT '1.0',
    generated_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backend_key TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    artifact_path TEXT,
    sha256 TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS media_assets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    driver_id INTEGER,
    session_id INTEGER,
    asset_type TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    original_filename TEXT,
    mime_type TEXT,
    file_size_bytes INTEGER,
    width INTEGER,
    height INTEGER,
    duration_seconds REAL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (driver_id) REFERENCES drivers (id) ON DELETE SET NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS driver_biometrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    driver_id INTEGER NOT NULL,
    biometric_type TEXT NOT NULL,
    model_backend TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    enrollment_asset_id INTEGER,
    threshold REAL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (driver_id) REFERENCES drivers (id) ON DELETE CASCADE,
    FOREIGN KEY (enrollment_asset_id) REFERENCES media_assets (id) ON DELETE SET NULL,
    UNIQUE (driver_id, biometric_type, model_backend)
);

CREATE TABLE IF NOT EXISTS inference_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    driver_id INTEGER,
    session_id INTEGER,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    model_backend TEXT,
    model_artifact_id INTEGER,
    input_asset_id INTEGER,
    started_at TEXT,
    ended_at TEXT,
    latency_ms REAL,
    summary_json TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    FOREIGN KEY (driver_id) REFERENCES drivers (id) ON DELETE SET NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE,
    FOREIGN KEY (model_artifact_id) REFERENCES model_artifacts (id) ON DELETE SET NULL,
    FOREIGN KEY (input_asset_id) REFERENCES media_assets (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    session_id INTEGER,
    timestamp_seconds REAL,
    frame_index INTEGER,
    model_backend TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    probabilities_json TEXT,
    features_json TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES inference_runs (id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS session_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    model_artifact_id INTEGER,
    model_backend TEXT NOT NULL,
    model_role TEXT NOT NULL,
    threshold REAL,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE,
    FOREIGN KEY (model_artifact_id) REFERENCES model_artifacts (id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS app_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    driver_id INTEGER,
    session_id INTEGER,
    action TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL,
    FOREIGN KEY (driver_id) REFERENCES drivers (id) ON DELETE SET NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_drivers_user_id ON drivers (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_driver_id ON sessions (driver_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions (status);
CREATE INDEX IF NOT EXISTS idx_events_session_id ON events (session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type);
CREATE INDEX IF NOT EXISTS idx_risk_logs_session_id ON risk_logs (session_id);
CREATE INDEX IF NOT EXISTS idx_reports_session_id ON reports (session_id);
CREATE INDEX IF NOT EXISTS idx_model_artifacts_type ON model_artifacts (model_type, status);
CREATE INDEX IF NOT EXISTS idx_media_assets_user ON media_assets (user_id, asset_type);
CREATE INDEX IF NOT EXISTS idx_media_assets_session ON media_assets (session_id);
CREATE INDEX IF NOT EXISTS idx_driver_biometrics_driver ON driver_biometrics (driver_id, biometric_type, status);
CREATE INDEX IF NOT EXISTS idx_inference_runs_session ON inference_runs (session_id, run_type);
CREATE INDEX IF NOT EXISTS idx_model_predictions_session ON model_predictions (session_id, model_backend);
CREATE INDEX IF NOT EXISTS idx_session_models_session ON session_models (session_id, model_role);
CREATE INDEX IF NOT EXISTS idx_app_audit_logs_user ON app_audit_logs (user_id, created_at);
