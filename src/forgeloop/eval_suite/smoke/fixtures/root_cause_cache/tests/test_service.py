import unittest

from cache import MemoryCache
from service import AnalyticsService


class AnalyticsServiceTests(unittest.TestCase):
    def test_zero_is_a_valid_cached_value(self):
        calls = []

        def compute(name):
            calls.append(name)
            return 0

        service = AnalyticsService(MemoryCache(), compute)
        self.assertEqual(service.metric("errors"), 0)
        self.assertEqual(service.metric("errors"), 0)
        self.assertEqual(calls, ["errors"])


if __name__ == "__main__":
    unittest.main()
