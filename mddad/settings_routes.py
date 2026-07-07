import csv
from io import StringIO

from flask import Response, flash, g, redirect, render_template, request, url_for

from .auth import login_required
from .database import get_db
from .evaluation import csv_header, csv_row, rows_for_user, row_to_summary
from .i18n import translate
from .settings import SETTING_DEFINITIONS, get_analysis_settings, reset_settings, save_settings, validate_settings


def register_evaluation_routes(app):
    @app.route("/settings", methods=("GET", "POST"))
    @login_required
    def settings_page():
        db = get_db()

        if request.method == "POST":
            action = request.form.get("action", "save")
            if action == "reset":
                reset_settings(db)
                db.commit()
                flash(translate("flash.settings_reset", "Settings were reset to defaults."), "success")
                return redirect(url_for("settings_page"))

            values, errors = validate_settings(request.form)
            if errors:
                for error in errors:
                    flash(error, "error")
            else:
                save_settings(db, values)
                db.commit()
                flash(translate("flash.settings_saved", "Settings were saved."), "success")
                return redirect(url_for("settings_page"))

        settings_values = get_analysis_settings(db, app)
        return render_template(
            "settings.html",
            setting_definitions=SETTING_DEFINITIONS,
            settings_values=settings_values,
        )

    @app.get("/evaluation")
    @login_required
    def evaluation_page():
        rows = rows_for_user(get_db(), g.user["id"])
        summaries = {row["session_id"]: row_to_summary(row) for row in rows}
        return render_template("evaluation.html", rows=rows, summaries=summaries)

    @app.get("/evaluation.csv")
    @login_required
    def evaluation_csv():
        rows = rows_for_user(get_db(), g.user["id"])
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(csv_header())
        for row in rows:
            writer.writerow(csv_row(row))
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=mddad_evaluation.csv"},
        )
