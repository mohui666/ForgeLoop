import unittest

from slug import slugify


class SlugifyTests(unittest.TestCase):
    def test_basic_words(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_punctuation(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")


if __name__ == "__main__":
    unittest.main()
