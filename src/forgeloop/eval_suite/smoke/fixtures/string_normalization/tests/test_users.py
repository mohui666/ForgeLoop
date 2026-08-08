import unittest

from users import normalize_username


class NormalizeUsernameTests(unittest.TestCase):
    def test_trims_and_lowercases(self):
        self.assertEqual(normalize_username("  Alice  "), "alice")

    def test_collapses_all_whitespace(self):
        self.assertEqual(normalize_username("Alice\t  Smith"), "alice_smith")

    def test_preserves_existing_underscores(self):
        self.assertEqual(normalize_username("ALICE_SMITH"), "alice_smith")


if __name__ == "__main__":
    unittest.main()
