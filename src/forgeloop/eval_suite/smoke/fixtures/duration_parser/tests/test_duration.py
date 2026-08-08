import unittest

from duration import parse_duration


class ParseDurationTests(unittest.TestCase):
    def test_single_component(self):
        self.assertEqual(parse_duration("45s"), 45)

    def test_multiple_components(self):
        self.assertEqual(parse_duration("1h30m5s"), 5405)

    def test_partial_ordered_components(self):
        self.assertEqual(parse_duration("2h15s"), 7215)

    def test_invalid_values(self):
        for value in ("", "1m2h", "1h2h", "1.5h", "1h!", "hours"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_duration(value)


if __name__ == "__main__":
    unittest.main()
