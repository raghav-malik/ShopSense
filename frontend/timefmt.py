"""Times as the UI shows them, in the backend's time zone (GET /health "timezone").

The API sends UTC ISO 8601 timestamps; every label here converts them to the
same zone the server uses to date chats and day files, so the sidebar, the
"Updated" labels and the day files always agree.
"""

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "Asia/Kolkata"


def _local(timestamp: str, tz: str) -> datetime:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)).astimezone(ZoneInfo(tz))


def _clock(moment: datetime) -> str:
    return moment.strftime("%I:%M %p").lstrip("0")


def updated_label(timestamp: str | None, tz: str = DEFAULT_TIMEZONE) -> str:
    """ "Updated OCT 6, 2026 | 10:15 AM", or "Not saved yet"."""
    if not timestamp:
        return "Not saved yet"
    local = _local(timestamp, tz)
    month = f"{local:%b}".upper()
    return f"Updated {month} {local.day}, {local.year} | {_clock(local)}"


def relative_time(timestamp: str, tz: str = DEFAULT_TIMEZONE, now: datetime | None = None) -> str:
    """When a chat was last active: "Just now", "5 min ago", "3 hr ago", "Yesterday",
    "4 days ago", then the date ("6 Oct", with the year if it isn't this year).
    Days are calendar days in the user's time zone, not 24-hour spans."""
    local = _local(timestamp, tz)
    current = (now or datetime.now(UTC)).astimezone(ZoneInfo(tz))
    seconds = (current - local).total_seconds()
    days = (current.date() - local.date()).days
    if seconds < 60:
        return "Just now"
    if days == 0 and seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if days == 0:
        return f"{int(seconds // 3600)} hr ago"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return f"{days} days ago"
    return f"{local.day} {local:%b}" + ("" if local.year == current.year else f" {local.year}")


def day_label(day: str, tz: str = DEFAULT_TIMEZONE, now: datetime | None = None) -> str:
    """A day file's day: "Today", "Yesterday", or "Tue, 6 Oct 2026"."""
    the_day = date.fromisoformat(day)
    today = (now or datetime.now(UTC)).astimezone(ZoneInfo(tz)).date()
    if the_day == today:
        return f"Today, {the_day.day} {the_day:%b}"
    if the_day == today - timedelta(days=1):
        return f"Yesterday, {the_day.day} {the_day:%b}"
    return f"{the_day:%a}, {the_day.day} {the_day:%b %Y}"
