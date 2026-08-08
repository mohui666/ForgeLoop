import unittest
from http import status_category


class StatusCategoryTests(unittest.TestCase):
    def test_boundary_statuses(self):
        self.assertEqual(status_category(200), "success")
        self.assertEqual(status_category(299), "success")
        self.assertEqual(status_category(400), "client_error")
        self.assertEqual(status_category(499), "client_error")
        self.assertEqual(status_category(500), "server_error")
        self.assertEqual(status_category(599), "server_error")

    def test_other_status(self):
        self.assertEqual(status_category(302), "other")


if __name__ == "__main__":
    unittest.main()
