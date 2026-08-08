import unittest

from urls import build_query


class BuildQueryTests(unittest.TestCase):
    def test_encodes_reserved_characters(self):
        self.assertEqual(
            build_query({"q": "hello world", "tag": "a&b"}), "q=hello+world&tag=a%26b"
        )

    def test_skips_none_and_expands_sequences(self):
        self.assertEqual(
            build_query({"tag": ["a", "b"], "cursor": None}), "tag=a&tag=b"
        )

    def test_empty_params(self):
        self.assertEqual(build_query({}), "")


if __name__ == "__main__":
    unittest.main()
