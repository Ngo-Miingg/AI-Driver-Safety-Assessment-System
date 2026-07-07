from flask import abort, flash, redirect, render_template, request, session, url_for

from .auth import login_required
from .database import get_db
from .i18n import translate


def _current_user_id():
    from flask import g

    return g.user["id"]


def _get_owned_driver(driver_id):
    driver = get_db().execute(
        """
        SELECT id, user_id, display_name, description, notes, status, created_at, updated_at
        FROM drivers
        WHERE id = ? AND user_id = ?
        """,
        (driver_id, _current_user_id()),
    ).fetchone()
    if driver is None:
        abort(404)
    return driver


def register_driver_routes(app):
    @app.get("/drivers")
    @login_required
    def drivers():
        rows = get_db().execute(
            """
            SELECT id, display_name, description, status, created_at
            FROM drivers
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (_current_user_id(),),
        ).fetchall()
        return render_template("drivers.html", drivers=rows)

    @app.route("/drivers/new", methods=("GET", "POST"))
    @login_required
    def new_driver():
        if request.method == "POST":
            display_name = request.form.get("display_name", "").strip()
            description = request.form.get("description", "").strip()
            notes = request.form.get("notes", "").strip()

            if not display_name:
                flash(translate("flash.driver_name_required", "Driver name is required."), "error")
                return render_template(
                    "driver_form.html",
                    form_title="Create Driver",
                    driver=None,
                )

            cursor = get_db().execute(
                """
                INSERT INTO drivers (user_id, display_name, description, notes)
                VALUES (?, ?, ?, ?)
                """,
                (_current_user_id(), display_name, description or None, notes or None),
            )
            get_db().commit()
            flash(translate("flash.driver_created", "Driver profile created."), "success")
            return redirect(url_for("driver_detail", driver_id=cursor.lastrowid))

        return render_template("driver_form.html", form_title="Create Driver", driver=None)

    @app.get("/drivers/<int:driver_id>")
    @login_required
    def driver_detail(driver_id):
        driver = _get_owned_driver(driver_id)
        summary = get_db().execute(
            """
            SELECT
                COUNT(*) AS total_sessions,
                AVG(final_safety_score) AS average_safety
            FROM sessions
            WHERE driver_id = ? AND user_id = ?
            """,
            (driver_id, _current_user_id()),
        ).fetchone()
        return render_template("driver_detail.html", driver=driver, summary=summary)

    @app.route("/drivers/<int:driver_id>/edit", methods=("GET", "POST"))
    @login_required
    def edit_driver(driver_id):
        driver = _get_owned_driver(driver_id)

        if request.method == "POST":
            display_name = request.form.get("display_name", "").strip()
            description = request.form.get("description", "").strip()
            notes = request.form.get("notes", "").strip()
            status = request.form.get("status", "active")

            if not display_name:
                flash(translate("flash.driver_name_required", "Driver name is required."), "error")
                return render_template(
                    "driver_form.html",
                    form_title="Edit Driver",
                    driver=driver,
                )

            if status not in {"active", "inactive"}:
                status = "active"

            get_db().execute(
                """
                UPDATE drivers
                SET display_name = ?, description = ?, notes = ?, status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (
                    display_name,
                    description or None,
                    notes or None,
                    status,
                    driver_id,
                    _current_user_id(),
                ),
            )
            get_db().commit()
            if session.get("active_driver_id") == driver_id:
                session["active_driver_name"] = display_name
            flash(translate("flash.driver_updated", "Driver profile updated."), "success")
            return redirect(url_for("driver_detail", driver_id=driver_id))

        return render_template("driver_form.html", form_title="Edit Driver", driver=driver)

    @app.post("/drivers/<int:driver_id>/select")
    @login_required
    def select_driver(driver_id):
        driver = _get_owned_driver(driver_id)
        session["active_driver_id"] = driver["id"]
        session["active_driver_name"] = driver["display_name"]
        flash(f"{translate('flash.selected_driver', 'Selected driver')}: {driver['display_name']}", "success")
        return redirect(request.referrer or url_for("dashboard"))
