import unittest

from intervals import merge_intervals


class MergeIntervalsTests(unittest.TestCase):
    def test_sorts_and_merges_overlapping_intervals(self):
        self.assertEqual(merge_intervals([(8, 10), (1, 4), (3, 6)]), [(1, 6), (8, 10)])

    def test_merges_touching_integer_intervals(self):
        self.assertEqual(merge_intervals([(1, 2), (3, 5)]), [(1, 5)])

    def test_rejects_reversed_intervals(self):
        with self.assertRaises(ValueError):
            merge_intervals([(4, 2)])

    def test_does_not_mutate_input(self):
        intervals = [(5, 7), (1, 2)]
        merge_intervals(intervals)
        self.assertEqual(intervals, [(5, 7), (1, 2)])


if __name__ == "__main__":
    unittest.main()
