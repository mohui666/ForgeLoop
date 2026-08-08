import unittest

from inventory import Inventory


class InventoryTests(unittest.TestCase):
    def test_successful_reservation_updates_all_items(self):
        inventory = Inventory({"a": 5, "b": 3})
        inventory.reserve({"a": 2, "b": 1})
        self.assertEqual(inventory.stock, {"a": 3, "b": 2})

    def test_failed_reservation_is_atomic(self):
        inventory = Inventory({"a": 5, "b": 1})
        with self.assertRaisesRegex(ValueError, "b"):
            inventory.reserve({"a": 2, "b": 2})
        self.assertEqual(inventory.stock, {"a": 5, "b": 1})

    def test_rejects_non_positive_quantities_without_mutation(self):
        inventory = Inventory({"a": 5})
        with self.assertRaises(ValueError):
            inventory.reserve({"a": 0})
        self.assertEqual(inventory.stock, {"a": 5})


if __name__ == "__main__":
    unittest.main()
