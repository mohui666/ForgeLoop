import unittest

from pagination import page_window


class PageWindowTests(unittest.TestCase):
    def test_valid_window(self):
        self.assertEqual(page_window(3, 25), (50, 75))

    def test_page_must_start_at_one(self):
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                page_window(value, 20)

    def test_page_size_range(self):
        for value in (0, -5, 101):
            with self.subTest(value=value), self.assertRaises(ValueError):
                page_window(1, value)


if __name__ == "__main__":
    unittest.main()
