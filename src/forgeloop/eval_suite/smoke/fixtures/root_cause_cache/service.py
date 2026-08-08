class AnalyticsService:
    def __init__(self, cache, compute):
        self.cache = cache
        self.compute = compute

    def metric(self, name):
        cached = self.cache.get(name)
        if cached is not None:
            return cached
        value = self.compute(name)
        self.cache.set(name, value)
        return value
