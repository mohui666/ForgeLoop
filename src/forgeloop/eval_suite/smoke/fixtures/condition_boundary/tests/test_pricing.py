import unittest

from pricing import discount_rate


class DiscountRateTests(unittest.TestCase):
    def test_eligible_member(self):
        self.assertEqual(discount_rate(True, 100), 0.10)

    def test_non_member_never_gets_loyalty_discount(self):
        self.assertEqual(discount_rate(False, 250), 0.0)

    def test_member_below_threshold(self):
        self.assertEqual(discount_rate(True, 99.99), 0.0)


if __name__ == "__main__":
    unittest.main()
