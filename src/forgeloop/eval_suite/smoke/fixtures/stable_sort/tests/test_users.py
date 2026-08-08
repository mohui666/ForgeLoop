import unittest

from users import sort_users


class SortUsersTests(unittest.TestCase):
    def test_sorts_by_age_then_case_insensitive_name(self):
        users = [
            {"name": "zoe", "age": 30},
            {"name": "Bob", "age": 20},
            {"name": "alice", "age": 20},
        ]
        self.assertEqual(
            [user["name"] for user in sort_users(users)], ["alice", "Bob", "zoe"]
        )

    def test_does_not_mutate_input(self):
        users = [{"name": "B", "age": 2}, {"name": "A", "age": 1}]
        original = list(users)
        sort_users(users)
        self.assertEqual(users, original)


if __name__ == "__main__":
    unittest.main()
