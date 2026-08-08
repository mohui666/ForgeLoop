from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class Event:
    name: str
    occurred_at: datetime
    _source: str = "internal"


def event_to_dict(event: Event) -> dict:
    return asdict(event)
