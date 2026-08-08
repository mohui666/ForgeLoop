from datetime import datetime


def build_timestamp() -> str:
    return datetime.now(datetime.timezone.utc).isoformat()
