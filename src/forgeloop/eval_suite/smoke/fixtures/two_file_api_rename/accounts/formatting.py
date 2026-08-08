def format_user(account: dict) -> str:
    return f"{account['id']}: {account['name'].strip()}"
