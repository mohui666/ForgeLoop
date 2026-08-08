import ast
import pathlib
import sys
import unittest

suite = unittest.defaultTestLoader.discover("tests")
result = unittest.TextTestRunner(verbosity=2).run(suite)
tree = ast.parse(pathlib.Path("tests/test_slug.py").read_text(encoding="utf-8"))
names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
if "test_collapses_repeated_separators" not in names:
    print("required regression test is missing", file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0 if result.wasSuccessful() else 1)
