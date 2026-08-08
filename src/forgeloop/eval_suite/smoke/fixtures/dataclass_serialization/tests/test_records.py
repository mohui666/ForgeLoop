import json
import unittest
from datetime import datetime, timezone

from records import Event, event_to_dict


class EventSerializationTests(unittest.TestCase):
    def test_serializes_datetime_and_omits_private_fields(self):
        event = Event("deploy", datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc), "api")
        self.assertEqual(
            event_to_dict(event),
            {"name": "deploy", "occurred_at": "2026-01-02T03:04:00+00:00"},
        )

    def test_result_is_json_serializable(self):
        payload = event_to_dict(
            Event("build", datetime(2026, 1, 1, tzinfo=timezone.utc))
        )
        self.assertIn('"name": "build"', json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
