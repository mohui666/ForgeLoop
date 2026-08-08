import unittest

import adapter


class AdapterFixtureTests(unittest.TestCase):
    def test_module_exposes_expected_entrypoint(self):
        self.assertTrue(callable(adapter.encode_payload))


if __name__ == "__main__":
    unittest.main()
