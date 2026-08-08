import unittest

from flatten import flatten


class FlattenTests(unittest.TestCase):
    def test_flattens_lists_and_tuples_recursively(self):
        self.assertEqual(flatten([1, [2, (3, [4])], 5]), [1, 2, 3, 4, 5])

    def test_strings_and_dicts_are_atomic(self):
        marker = {"a": 1}
        self.assertEqual(flatten(["ab", marker]), ["ab", marker])

    def test_empty_nested_values(self):
        self.assertEqual(flatten([[], ((),)]), [])


if __name__ == "__main__":
    unittest.main()
