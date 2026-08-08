import unittest
from datetime import datetime

from reports import build_timestamp


class TimestampTests(unittest.TestCase):
    def test_timestamp_is_utc_and_parseable(self):
        value = build_timestamp()
        parsed = datetime.fromisoformat(value)
        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)
        self.assertTrue(value.endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
