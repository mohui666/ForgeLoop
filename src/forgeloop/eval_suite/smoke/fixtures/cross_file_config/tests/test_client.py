import unittest

import settings
from client import request_options


class RequestOptionsTests(unittest.TestCase):
    def test_uses_renamed_request_timeout(self):
        self.assertEqual(settings.REQUEST_TIMEOUT, 5)
        self.assertEqual(request_options(), {"timeout": 5})

    def test_old_constant_is_removed(self):
        self.assertFalse(hasattr(settings, "TIMEOUT"))


if __name__ == "__main__":
    unittest.main()
