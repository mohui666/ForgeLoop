import unittest

from profiles import display_name


class DisplayNameTests(unittest.TestCase):
    def test_nickname_wins(self):
        self.assertEqual(display_name({"nickname": "  kim ", "name": "Kim Lee"}), "kim")

    def test_none_nickname_falls_back(self):
        self.assertEqual(
            display_name({"nickname": None, "name": " Kim Lee "}), "Kim Lee"
        )

    def test_missing_values_are_anonymous(self):
        self.assertEqual(display_name({}), "Anonymous")

    def test_whitespace_values_are_anonymous(self):
        self.assertEqual(display_name({"nickname": " ", "name": None}), "Anonymous")


if __name__ == "__main__":
    unittest.main()
