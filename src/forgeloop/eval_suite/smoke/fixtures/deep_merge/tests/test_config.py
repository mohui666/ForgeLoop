import unittest

from config import merge_config


class MergeConfigTests(unittest.TestCase):
    def test_recursively_merges_nested_dicts(self):
        defaults = {"db": {"host": "localhost", "port": 5432}, "debug": False}
        overrides = {"db": {"host": "db.internal"}}
        self.assertEqual(
            merge_config(defaults, overrides),
            {"db": {"host": "db.internal", "port": 5432}, "debug": False},
        )

    def test_lists_and_scalars_are_replaced(self):
        self.assertEqual(
            merge_config(
                {"tags": ["a"], "value": {"x": 1}}, {"tags": ["b"], "value": 4}
            ),
            {"tags": ["b"], "value": 4},
        )

    def test_inputs_are_not_mutated_or_aliased(self):
        defaults = {"nested": {"values": []}}
        result = merge_config(defaults, {})
        result["nested"]["values"].append("new")
        self.assertEqual(defaults, {"nested": {"values": []}})


if __name__ == "__main__":
    unittest.main()
