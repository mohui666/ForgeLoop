import unittest

from orders import summarize_orders


class OrderSummaryTests(unittest.TestCase):
    def test_summary(self):
        orders = [
            {"status": "paid", "total": "10.126"},
            {"status": "paid", "total": None},
            {"status": "pending", "total": 5},
            {"status": "cancelled", "total": 99},
        ]
        self.assertEqual(summarize_orders(orders), {"paid": 10.13, "pending": 5.0})


if __name__ == "__main__":
    unittest.main()
