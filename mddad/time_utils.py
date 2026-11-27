from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


try:
    HANOI_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
except Exception:
    # Windows installations may not ship the IANA tzdata package. Vietnam
    # uses UTC+7 year-round, so this fallback is deterministic and sufficient.
    HANOI_TIMEZONE = timezone(timedelta(hours=7), name="ICT")


def format_hanoi_time(value) -> str:
    """Format SQLite UTC timestamps as local Hanoi time for the UI."""
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return str(value)

    # SQLite CURRENT_TIMESTAMP is UTC but has no timezone suffix.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(HANOI_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")
