from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(app):
    log_dir = Path(app.config["LOG_DIR"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = getattr(logging, str(app.config.get("LOG_LEVEL", "INFO")).upper(), logging.INFO)

    app.logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    file_path = log_dir / "app.log"

    if not _has_file_handler(app.logger, file_path):
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)

    app.logger.info(
        "Application startup env=%s debug=%s database=%s upload_folder=%s detector_backend=%s",
        app.config.get("ENV", "development"),
        app.config.get("DEBUG"),
        app.config.get("DATABASE_PATH"),
        app.config.get("UPLOAD_FOLDER"),
        app.config.get("PHONE_DETECTOR_BACKEND"),
    )


def _has_file_handler(logger, file_path):
    target = str(file_path)
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and getattr(handler, "baseFilename", None) == target:
            return True
    return False
