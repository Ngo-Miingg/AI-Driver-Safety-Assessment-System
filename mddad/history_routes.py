from flask import g, render_template

from .auth import login_required
from .database import get_db


def register_history_routes(app):
    @app.get("/history")
    @login_required
    def history():
        sessions = get_db().execute(
            """
            SELECT
                s.id,
                s.session_type,
                s.status,
                s.original_video_filename,
                s.video_duration_seconds,
                s.duration_seconds,
                s.created_at,
                s.started_at,
                s.final_safety_score,
                s.final_risk_score,
                s.final_alert_level,
                s.fatigue_trend_label,
                s.analysis_mode,
                COUNT(e.id) AS event_count,
                SUM(CASE WHEN e.event_type = 'phone_usage' THEN 1 ELSE 0 END) AS phone_usage_event_count,
                r.status AS report_status,
                d.display_name AS driver_name
            FROM sessions s
            JOIN drivers d ON d.id = s.driver_id
            LEFT JOIN events e ON e.session_id = s.id
            LEFT JOIN reports r ON r.session_id = s.id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY s.created_at DESC
            """,
            (g.user["id"],),
        ).fetchall()
        return render_template("history.html", sessions=sessions)
