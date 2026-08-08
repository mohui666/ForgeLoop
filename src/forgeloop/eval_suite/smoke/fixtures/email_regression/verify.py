import ast
import pathlib
import subprocess
import sys

test_path = pathlib.Path("tests/test_emails.py")
tree = ast.parse(test_path.read_text(encoding="utf-8"))
names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
if "test_trims_surrounding_whitespace" not in names:
    print("missing test method: test_trims_surrounding_whitespace", file=sys.stderr)
    raise SystemExit(1)
raise SystemExit(
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        check=False,
    ).returncode
)
