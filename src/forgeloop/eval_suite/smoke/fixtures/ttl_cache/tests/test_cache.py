import unittest

from cache import TTLCache


class TTLCacheTests(unittest.TestCase):
    def test_returns_unexpired_value(self):
        cache = TTLCache()
        cache.set("a", 0, expires_at=10)
        self.assertEqual(cache.get("a", now=9.9), 0)

    def test_exact_boundary_is_expired_and_removed(self):
        cache = TTLCache()
        cache.set("a", "value", expires_at=10)
        self.assertIsNone(cache.get("a", now=10))
        self.assertNotIn("a", cache._values)

    def test_missing_value(self):
        self.assertIsNone(TTLCache().get("missing", now=0))


if __name__ == "__main__":
    unittest.main()
