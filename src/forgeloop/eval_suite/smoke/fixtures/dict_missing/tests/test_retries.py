import unittest

from retries import retry_count


class RetryCountTests(unittest.TestCase):
    def test_configured_count(self):
        self.assertEqual(retry_count({"retry": {"count": 5}}), 5)

    def test_missing_retry_section_uses_default(self):
        self.assertEqual(retry_count({}), 3)

    def test_missing_count_uses_default(self):
        self.assertEqual(retry_count({"retry": {}}), 3)

    def test_explicit_zero_is_preserved(self):
        self.assertEqual(retry_count({"retry": {"count": 0}}), 0)


if __name__ == "__main__":
    unittest.main()
