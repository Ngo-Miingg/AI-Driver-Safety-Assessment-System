from flask import redirect, render_template, url_for

from .database import get_db


def register_routes(app):
    @app.get("/")
    def home():
        return redirect(url_for("dashboard"))

    @app.get("/foundation")
    def foundation():
        db = get_db()
        db.execute("SELECT 1").fetchone()
        return render_template(
            "home.html",
            app_title="AI Driver Safety Assessment System",
            database_path=app.config["DATABASE_PATH"],
        )

    @app.get("/health")
    def health():
        db = get_db()
        db.execute("SELECT 1").fetchone()
        return {
            "status": "ok",
            "database": "connected",
            "socketio": "initialized",
        }
