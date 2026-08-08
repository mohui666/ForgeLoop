import pathlib
import sys
import unittest

import orders

suite = unittest.defaultTestLoader.discover("tests")
result = unittest.TextTestRunner(verbosity=2).run(suite)
source = pathlib.Path("orders.py").read_text(encoding="utf-8")
if not hasattr(orders, "_normalized_total") or source.count('order.get("total")') > 1:
    print("_normalized_total refactor was not completed", file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(0 if result.wasSuccessful() else 1)
