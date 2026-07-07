from flask import g, render_template, request, session

from .auth import login_required
from .database import get_db


def register_dashboard_routes(app):
    @app.get("/dashboard")
    @login_required
    def dashboard():
        db = get_db()
        user_id = g.user["id"]

        drivers = db.execute(
            """
            SELECT id, display_name, description, status, created_at
            FROM drivers
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()

        total_sessions = db.execute(
            "SELECT COUNT(*) AS count FROM sessions WHERE user_id = ?",
            (user_id,),
        ).fetchone()["count"]

        average_safety = db.execute(
            """
            SELECT AVG(final_safety_score) AS value
            FROM sessions
            WHERE user_id = ? AND final_safety_score IS NOT NULL
            """,
            (user_id,),
        ).fetchone()["value"]

        event_counts = {
            "drowsiness": 0,
            "distraction": 0,
            "phone": 0,
        }
        row = db.execute(
            """
            SELECT
                SUM(CASE WHEN e.event_type IN ('eye_closure', 'yawn') THEN 1 ELSE 0 END) AS drowsiness,
                SUM(CASE WHEN e.event_type IN ('looking_away', 'head_down', 'face_missing') THEN 1 ELSE 0 END) AS distraction,
                SUM(CASE WHEN e.event_type = 'phone_usage' THEN 1 ELSE 0 END) AS phone
            FROM events e
            JOIN sessions s ON s.id = e.session_id
            WHERE s.user_id = ?
            """,
            (user_id,),
        ).fetchone()
        if row:
            event_counts = {
                "drowsiness": row["drowsiness"] or 0,
                "distraction": row["distraction"] or 0,
                "phone": row["phone"] or 0,
            }

        latest_session = db.execute(
            """
            SELECT final_safety_score, final_risk_score, final_alert_level, status, created_at
            FROM sessions
            WHERE user_id = ?
            ORDER BY COALESCE(started_at, created_at) DESC, id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

        recent_sessions = db.execute(
            """
            SELECT
                s.id,
                s.session_type,
                s.status,
                s.final_safety_score,
                s.final_risk_score,
                s.final_alert_level,
                s.created_at,
                s.started_at,
                s.duration_seconds,
                s.video_duration_seconds,
                d.display_name AS driver_name,
                COUNT(e.id) AS event_count
            FROM sessions s
            JOIN drivers d ON d.id = s.driver_id
            LEFT JOIN events e ON e.session_id = s.id
            WHERE s.user_id = ?
            GROUP BY s.id
            ORDER BY COALESCE(s.started_at, s.created_at) DESC, s.id DESC
            LIMIT 6
            """,
            (user_id,),
        ).fetchall()

        active_driver = None
        active_driver_id = session.get("active_driver_id")
        if active_driver_id:
            active_driver = db.execute(
                """
                SELECT id, display_name, description, status, created_at
                FROM drivers
                WHERE id = ? AND user_id = ?
                """,
                (active_driver_id, user_id),
            ).fetchone()

        return render_template(
            "dashboard.html",
            drivers=drivers,
            total_sessions=total_sessions,
            average_safety=average_safety,
            event_counts=event_counts,
            latest_session=latest_session,
            recent_sessions=recent_sessions,
            active_driver=active_driver,
        )

    @app.get("/reports")
    @login_required
    def reports():
        db = get_db()
        driver_id = request.args.get("driver_id", type=int)
        params = [g.user["id"]]
        driver_filter = ""
        if driver_id:
            driver_filter = "AND d.id = ?"
            params.append(driver_id)

        reports_list = db.execute(
            f"""
            SELECT
                r.id,
                r.status,
                r.title,
                r.summary,
                r.generated_at,
                r.created_at,
                s.id AS session_id,
                s.session_type,
                s.final_safety_score,
                s.final_risk_score,
                s.final_alert_level,
                d.id AS driver_id,
                d.display_name AS driver_name
            FROM reports r
            JOIN sessions s ON s.id = r.session_id
            JOIN drivers d ON d.id = s.driver_id
            WHERE s.user_id = ?
            {driver_filter}
            ORDER BY COALESCE(r.generated_at, r.created_at) DESC, r.id DESC
            """,
            params,
        ).fetchall()

        drivers = db.execute(
            """
            SELECT id, display_name
            FROM drivers
            WHERE user_id = ?
            ORDER BY display_name ASC
            """,
            (g.user["id"],),
        ).fetchall()

        return render_template(
            "reports.html",
            reports=reports_list,
            drivers=drivers,
            selected_driver_id=driver_id,
        )
