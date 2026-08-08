from datetime import datetime, timezone


def is_overdue(deadline: datetime, now: datetime) -> bool:
    deadline_utc = deadline.replace(tzinfo=timezone.utc)
    now_utc = now.replace(tzinfo=timezone.utc)
    return deadline_utc < now_utc
