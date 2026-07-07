import sqlite3
from pathlib import Path

from flask import current_app, g
from werkzeug.security import generate_password_hash


def get_db():
    if "db" not in g:
        database_path = Path(current_app.config["DATABASE_PATH"])
        database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        g.db = connection
    return g.db


def close_db(_error=None):
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db():
    schema_path = Path(__file__).with_name("schema.sql")
    db = get_db()
    with schema_path.open("r", encoding="utf-8") as schema_file:
        db.executescript(schema_file.read())
    migrate_schema(db, schema_path)
    seed_demo_user(db)
    db.commit()


def migrate_schema(db, schema_path):
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    sessions_sql = db.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'sessions'"
    ).fetchone()
    if not sessions_sql:
        return

    sql_text = sessions_sql["sql"] or ""
    has_uploaded_video = "uploaded_video" in sql_text
    has_realtime_webcam = "realtime_webcam" in sql_text
    has_saved_path = db.execute(
        "SELECT 1 FROM pragma_table_info('sessions') WHERE name = 'saved_video_path'"
    ).fetchone()
    has_file_size = db.execute(
        "SELECT 1 FROM pragma_table_info('sessions') WHERE name = 'file_size_bytes'"
    ).fetchone()
    required_columns = {
        "duration_seconds": "REAL DEFAULT 0",
        "original_video_filename": "TEXT",
        "saved_video_path": "TEXT",
        "file_size_bytes": "INTEGER",
        "video_fps": "REAL",
        "video_frame_count": "INTEGER",
        "video_duration_seconds": "REAL",
        "video_width": "INTEGER",
        "video_height": "INTEGER",
        "processed_frame_count": "INTEGER DEFAULT 0",
        "average_processing_fps": "REAL",
        "analysis_error": "TEXT",
        "analysis_mode": "TEXT",
        "multimodal_model_backend": "TEXT",
        "multimodal_model_path": "TEXT",
        "multimodal_clip_count": "INTEGER DEFAULT 0",
        "audio_track_detected": "INTEGER DEFAULT 0",
        "skeleton_frame_count": "INTEGER DEFAULT 0",
        "face_detected_frame_count": "INTEGER DEFAULT 0",
        "face_missing_frame_count": "INTEGER DEFAULT 0",
        "average_ear": "REAL",
        "average_mar": "REAL",
        "min_ear": "REAL",
        "max_mar": "REAL",
        "phone_detected_frame_count": "INTEGER DEFAULT 0",
        "max_phone_confidence": "REAL",
        "phone_usage_event_count": "INTEGER DEFAULT 0",
        "phone_detection_error": "TEXT",
        "final_risk_score": "REAL",
        "final_safety_score": "REAL",
        "final_alert_level": "TEXT",
        "fatigue_trend_label": "TEXT",
        "failure_reason": "TEXT",
        "notes": "TEXT",
        "updated_at": "TEXT",
    }
    required_event_columns = {
        "evidence_image_path": "TEXT",
    }

    if not has_realtime_webcam and has_uploaded_video and has_saved_path and has_file_size:
        rebuild_sessions_table(db, schema_path)
    elif not (has_uploaded_video and has_saved_path and has_file_size):
        session_count = db.execute("SELECT COUNT(*) AS count FROM sessions").fetchone()["count"]
        if session_count:
            raise RuntimeError(
                "Existing sessions table needs migration before uploaded video support. "
                "Export or clear session data, then restart the app."
            )

        db.execute("PRAGMA foreign_keys = OFF")
        db.execute("DROP TABLE IF EXISTS reports")
        db.execute("DROP TABLE IF EXISTS risk_logs")
        db.execute("DROP TABLE IF EXISTS events")
        db.execute("DROP TABLE IF EXISTS sessions")
        with schema_path.open("r", encoding="utf-8") as schema_file:
            db.executescript(schema_file.read())
        db.execute("PRAGMA foreign_keys = ON")

    existing_columns = {
        row["name"]
        for row in db.execute("SELECT name FROM pragma_table_info('sessions')").fetchall()
    }
    for column_name, column_type in required_columns.items():
        if column_name not in existing_columns:
            db.execute(f"ALTER TABLE sessions ADD COLUMN {column_name} {column_type}")

    event_columns = {
        row["name"]
        for row in db.execute("SELECT name FROM pragma_table_info('events')").fetchall()
    }
    for column_name, column_type in required_event_columns.items():
        if column_name not in event_columns:
            db.execute(f"ALTER TABLE events ADD COLUMN {column_name} {column_type}")

    if child_tables_reference_old_sessions(db):
        rebuild_child_result_tables(db, schema_path)


def rebuild_sessions_table(db, schema_path):
    columns = [
        row["name"]
        for row in db.execute("SELECT name FROM pragma_table_info('sessions')").fetchall()
    ]
    db.execute("PRAGMA foreign_keys = OFF")
    for index_name in ("idx_sessions_user_id", "idx_sessions_driver_id", "idx_sessions_status"):
        db.execute(f"DROP INDEX IF EXISTS {index_name}")
    db.execute("ALTER TABLE sessions RENAME TO sessions_old")
    with schema_path.open("r", encoding="utf-8") as schema_file:
        db.executescript(schema_file.read())
    new_columns = [
        row["name"]
        for row in db.execute("SELECT name FROM pragma_table_info('sessions')").fetchall()
    ]
    common_columns = [column for column in columns if column in new_columns]
    select_columns = [
        "CASE WHEN session_type = 'realtime' THEN 'realtime_webcam' ELSE session_type END"
        if column == "session_type"
        else column
        for column in common_columns
    ]
    db.execute(
        f"""
        INSERT INTO sessions ({", ".join(common_columns)})
        SELECT {", ".join(select_columns)}
        FROM sessions_old
        """
    )
    db.execute("DROP TABLE sessions_old")
    db.execute("PRAGMA foreign_keys = ON")


def child_tables_reference_old_sessions(db):
    for table_name in ("events", "risk_logs", "reports"):
        row = db.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if row and "sessions_old" in (row["sql"] or ""):
            return True
    return False


def rebuild_child_result_tables(db, schema_path):
    table_names = ("events", "risk_logs", "reports")
    old_columns = {}
    db.execute("PRAGMA foreign_keys = OFF")
    for index_name in (
        "idx_events_session_id",
        "idx_events_type",
        "idx_risk_logs_session_id",
        "idx_reports_session_id",
    ):
        db.execute(f"DROP INDEX IF EXISTS {index_name}")

    for table_name in table_names:
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        if not row:
            continue
        old_columns[table_name] = [
            col["name"]
            for col in db.execute(f"SELECT name FROM pragma_table_info('{table_name}')").fetchall()
        ]
        db.execute(f"ALTER TABLE {table_name} RENAME TO {table_name}_old")

    with schema_path.open("r", encoding="utf-8") as schema_file:
        db.executescript(schema_file.read())

    for table_name, columns in old_columns.items():
        new_columns = [
            col["name"]
            for col in db.execute(f"SELECT name FROM pragma_table_info('{table_name}')").fetchall()
        ]
        common_columns = [column for column in columns if column in new_columns]
        if common_columns:
            db.execute(
                f"""
                INSERT INTO {table_name} ({", ".join(common_columns)})
                SELECT {", ".join(common_columns)}
                FROM {table_name}_old
                """
            )
        db.execute(f"DROP TABLE {table_name}_old")

    db.execute("PRAGMA foreign_keys = ON")


def seed_demo_user(db):
    if not current_app.config.get("SEED_DEMO_USER"):
        return

    username = current_app.config["DEMO_USERNAME"]
    existing = db.execute(
        "SELECT id FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if existing:
        return

    db.execute(
        """
        INSERT INTO users (username, password_hash, display_name, role, status)
        VALUES (?, ?, ?, 'user', 'active')
        """,
        (
            username,
            generate_password_hash(current_app.config["DEMO_PASSWORD"]),
            "Demo Administrator",
        ),
    )


def init_app(app):
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["DATABASE_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    app.teardown_appcontext(close_db)
    with app.app_context():
        init_db()
