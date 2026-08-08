import unittest

from metrics import safe_ratio


class SafeRatioTests(unittest.TestCase):
    def test_regular_ratio(self):
        self.assertEqual(safe_ratio(3, 2), 1.5)

    def test_zero_denominator_returns_zero(self):
        self.assertEqual(safe_ratio(10, 0), 0.0)

    def test_zero_numerator(self):
        self.assertEqual(safe_ratio(0, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
