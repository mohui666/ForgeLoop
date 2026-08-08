import unittest

from ranges import inclusive_values


class InclusiveValuesTests(unittest.TestCase):
    def test_includes_both_boundaries(self):
        self.assertEqual(inclusive_values(2, 5), [2, 3, 4, 5])

    def test_single_value_range(self):
        self.assertEqual(inclusive_values(4, 4), [4])

    def test_reversed_range_is_empty(self):
        self.assertEqual(inclusive_values(5, 2), [])


if __name__ == "__main__":
    unittest.main()
