import unittest

from retry import run_with_retries


class RunWithRetriesTests(unittest.TestCase):
    def test_stops_after_exact_attempt_limit(self):
        calls = []

        def fail():
            calls.append(1)
            raise ValueError("nope")

        with self.assertRaisesRegex(ValueError, "nope"):
            run_with_retries(fail, 3)
        self.assertEqual(len(calls), 3)

    def test_returns_first_success(self):
        values = iter([ValueError("first"), "ok"])

        def eventually():
            value = next(values)
            if isinstance(value, Exception):
                raise value
            return value

        self.assertEqual(run_with_retries(eventually, 2), "ok")

    def test_rejects_non_positive_attempts(self):
        with self.assertRaises(ValueError):
            run_with_retries(lambda: None, 0)


if __name__ == "__main__":
    unittest.main()
