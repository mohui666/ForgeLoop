import unittest

from labels import append_label


class AppendLabelTests(unittest.TestCase):
    def test_default_lists_are_independent(self):
        self.assertEqual(append_label("first"), ["first"])
        self.assertEqual(append_label("second"), ["second"])

    def test_explicit_list_is_updated(self):
        values = ["existing"]
        self.assertIs(append_label("new", values), values)
        self.assertEqual(values, ["existing", "new"])


if __name__ == "__main__":
    unittest.main()
