import unittest

from files import is_python_source


class PythonSourceTests(unittest.TestCase):
    def test_python_extensions_are_case_insensitive(self):
        self.assertTrue(is_python_source("app.py"))
        self.assertTrue(is_python_source("BUILD.PY"))

    def test_non_source_suffixes_are_rejected(self):
        self.assertFalse(is_python_source("copy"))
        self.assertFalse(is_python_source("module.pyc"))
        self.assertFalse(is_python_source("archive.py.zip"))


if __name__ == "__main__":
    unittest.main()
