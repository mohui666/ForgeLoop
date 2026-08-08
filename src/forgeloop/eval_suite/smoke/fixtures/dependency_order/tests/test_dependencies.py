import unittest

from dependencies import dependency_order


class DependencyOrderTests(unittest.TestCase):
    def test_dependencies_appear_before_consumers(self):
        graph = {"app": ["db", "api"], "api": ["core"], "db": ["core"], "core": []}
        order = dependency_order(graph)
        self.assertEqual(set(order), set(graph))
        self.assertLess(order.index("core"), order.index("api"))
        self.assertLess(order.index("core"), order.index("db"))
        self.assertLess(order.index("api"), order.index("app"))

    def test_includes_dependency_only_nodes(self):
        self.assertEqual(dependency_order({"app": ["external"]}), ["external", "app"])

    def test_cycle_raises_value_error_with_path(self):
        with self.assertRaisesRegex(ValueError, "a.*b.*a"):
            dependency_order({"a": ["b"], "b": ["a"]})


if __name__ == "__main__":
    unittest.main()
