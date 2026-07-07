from functools import wraps

from flask import flash, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .database import get_db
from .i18n import translate


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for("login", next=request.path))
        return view(**kwargs)

    return wrapped_view


def get_current_user():
    return g.get("user")


def register_auth(app):
    @app.before_request
    def load_logged_in_user():
        user_id = session.get("user_id")
        if user_id is None:
            g.user = None
            return

        g.user = get_db().execute(
            """
            SELECT id, username, display_name, email, role, status
            FROM users
            WHERE id = ? AND status = 'active'
            """,
            (user_id,),
        ).fetchone()

    @app.context_processor
    def inject_user_context():
        return {
            "current_user": g.get("user"),
            "active_driver_id": session.get("active_driver_id"),
            "active_driver_name": session.get("active_driver_name"),
        }

    @app.route("/login", methods=("GET", "POST"))
    def login():
        if g.get("user") is not None:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            error = None

            user = get_db().execute(
                "SELECT * FROM users WHERE username = ? AND status = 'active'",
                (username,),
            ).fetchone()

            if not username or not password:
                error = translate("flash.enter_credentials", "Enter both username and password.")
            elif user is None or not check_password_hash(user["password_hash"], password):
                error = translate("flash.invalid_credentials", "Invalid username or password.")

            if error is None:
                selected_language = session.get("language", "en")
                session.clear()
                session["language"] = selected_language
                session["user_id"] = user["id"]
                get_db().execute(
                    "UPDATE users SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (user["id"],),
                )
                get_db().commit()
                next_url = request.args.get("next")
                return redirect(next_url or url_for("dashboard"))

            flash(error, "error")

        return render_template("login.html")

    @app.route("/register", methods=("GET", "POST"))
    def register():
        if g.get("user") is not None:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            display_name = request.form.get("display_name", "").strip()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")
            error = None

            if not username:
                error = translate("flash.username_required", "Username is required.")
            elif not password:
                error = translate("flash.password_required", "Password is required.")
            elif password != confirm_password:
                error = translate("flash.passwords_mismatch", "Passwords do not match.")
            elif get_db().execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
                error = translate("flash.username_exists", "Username is already registered.")

            if error is None:
                get_db().execute(
                    """
                    INSERT INTO users (username, password_hash, display_name, role, status)
                    VALUES (?, ?, ?, 'user', 'active')
                    """,
                    (username, generate_password_hash(password), display_name or username),
                )
                get_db().commit()
                flash(translate("flash.account_created", "Account created. You can now log in."), "success")
                return redirect(url_for("login"))

            flash(error, "error")

        return render_template("register.html")

    @app.get("/logout")
    def logout():
        selected_language = session.get("language", "en")
        session.clear()
        session["language"] = selected_language
        flash(translate("flash.logout", "You have been logged out."), "success")
        return redirect(url_for("login"))
