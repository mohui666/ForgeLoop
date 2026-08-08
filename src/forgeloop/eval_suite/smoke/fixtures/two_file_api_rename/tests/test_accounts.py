import unittest

import accounts


class AccountFormattingTests(unittest.TestCase):
    def test_new_public_api(self):
        self.assertEqual(accounts.format_account({"id": 7, "name": " Ada "}), "7: Ada")

    def test_old_api_is_removed(self):
        self.assertFalse(hasattr(accounts, "format_user"))
        self.assertEqual(accounts.__all__, ["format_account"])


if __name__ == "__main__":
    unittest.main()
