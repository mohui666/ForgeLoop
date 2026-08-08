import unittest

from emails import normalize_email


class NormalizeEmailTests(unittest.TestCase):
    def test_lowercases_email(self):
        self.assertEqual(normalize_email("User@Example.COM"), "user@example.com")


if __name__ == "__main__":
    unittest.main()
