"""The UI's time labels: always in the user's time zone, on calendar days."""

from datetime import UTC, datetime

import pytest

from frontend.timefmt import day_label, relative_time, updated_label

# 10:30 AM on Tuesday 6 October 2026 in India.
NOW = datetime(2026, 10, 6, 5, 0, tzinfo=UTC)


def test_updated_label_is_in_the_users_time_zone() -> None:
    assert updated_label("2026-10-05T18:45:00+00:00") == "Updated OCT 6, 2026 | 12:15 AM"
    assert updated_label("2026-10-06T04:45:00Z") == "Updated OCT 6, 2026 | 10:15 AM"
    assert updated_label(None) == "Not saved yet"


@pytest.mark.parametrize(
    ("stamp", "label"),
    [
        ("2026-10-06T04:59:30+00:00", "Just now"),
        ("2026-10-06T04:45:00+00:00", "15 min ago"),
        ("2026-10-05T19:00:00+00:00", "10 hr ago"),  # 00:30 today in India
        ("2026-10-05T18:00:00+00:00", "Yesterday"),  # 23:30 yesterday in India, though only 11 hours ago
        ("2026-10-03T05:00:00+00:00", "3 days ago"),
        ("2026-09-20T05:00:00+00:00", "20 Sep"),
        ("2025-12-31T05:00:00+00:00", "31 Dec 2025"),
    ],
)
def test_relative_time_counts_calendar_days(stamp: str, label: str) -> None:
    assert relative_time(stamp, now=NOW) == label


def test_another_time_zone_changes_the_day() -> None:
    # 23:30 on the 5th in India is 13:00 on the 5th in New York; at 01:00 New York time on the 6th,
    # that's yesterday there too, while 00:30 India time on the 6th (15:00 on the 5th) is also yesterday.
    assert relative_time("2026-10-05T19:00:00+00:00", "America/New_York", datetime(2026, 10, 6, 5, 0, tzinfo=UTC)) == (
        "Yesterday"
    )


def test_day_labels() -> None:
    assert day_label("2026-10-06", now=NOW) == "Today, 6 Oct"
    assert day_label("2026-10-05", now=NOW) == "Yesterday, 5 Oct"
    assert day_label("2026-09-28", now=NOW) == "Mon, 28 Sep 2026"
