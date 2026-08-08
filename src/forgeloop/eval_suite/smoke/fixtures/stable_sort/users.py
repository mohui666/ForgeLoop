def sort_users(users: list[dict]) -> list[dict]:
    return sorted(users, key=lambda user: user["age"])
