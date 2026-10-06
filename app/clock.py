"""Dates and times as the user sees them.

Everything is stored in UTC (ISO 8601 with an offset, from `now_iso()`), and
converted here to the configured TIMEZONE whenever a date is shown or used:
the agent's "today", the day a chat belongs to, and the times in the UI. One
zone for all of them, so a chat at 00:30 in India is on the same day
everywhere instead of the previous UTC day.
"""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from app.config import settings


def zone() -> ZoneInfo:
    """The user's time zone (TIMEZONE, an IANA name such as Asia/Kolkata)."""
    return ZoneInfo(settings.timezone)


def parse_utc(timestamp: str) -> datetime:
    """A stored timestamp as an aware datetime. A naive one is taken as UTC,
    which is how every timestamp in the database is written."""
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def to_local(timestamp: str) -> datetime:
    """A stored timestamp in the user's time zone."""
    return parse_utc(timestamp).astimezone(zone())


def local_date(timestamp: str) -> date:
    """The user's calendar day for a stored timestamp."""
    return to_local(timestamp).date()


def local_now(now: datetime | None = None) -> datetime:
    """Now in the user's time zone (`now`, if given, for tests)."""
    return (now or datetime.now(UTC)).astimezone(zone())


def today_line(now: datetime | None = None) -> str:
    """Today's date for the agent, e.g. "Tuesday, 6 October 2026 (Asia/Kolkata)"."""
    local = local_now(now)
    return f"{local:%A}, {local.day} {local:%B %Y} ({settings.timezone})"
