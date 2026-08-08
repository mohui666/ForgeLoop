class CacheEntry:
    def __init__(self, value):
        self.value = value


class MemoryCache:
    def __init__(self):
        self._entries = {}

    def set(self, key, value):
        self._entries[key] = CacheEntry(value)

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None or not entry.value:
            return None
        return entry.value
