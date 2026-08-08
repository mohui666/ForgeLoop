import unittest
from datetime import datetime, timedelta, timezone

from deadlines import is_overdue


class DeadlineTests(unittest.TestCase):
    def test_compares_same_instant_across_timezones(self):
        deadline = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=8)))
        now = datetime(2026, 1, 1, 4, tzinfo=timezone.utc)
        self.assertFalse(is_overdue(deadline, now))

    def test_detects_overdue_across_timezones(self):
        deadline = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=8)))
        now = datetime(2026, 1, 1, 4, 1, tzinfo=timezone.utc)
        self.assertTrue(is_overdue(deadline, now))

    def test_rejects_naive_datetimes(self):
        aware = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with self.assertRaises(ValueError):
            is_overdue(datetime(2026, 1, 1), aware)  # noqa: DTZ001


if __name__ == "__main__":
    unittest.main()
