class TTLCache:
    def __init__(self):
        self._values = {}

    def set(self, key, value, expires_at: float) -> None:
        self._values[key] = (value, expires_at)

    def get(self, key, now: float):
        entry = self._values.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at < now:
            return None
        return value
