def parse_duration(value: str) -> int:
    """Parse a compact duration and return seconds."""
    if not value:
        raise ValueError("duration is empty")
    amount = int(value[:-1])
    unit = value[-1]
    factors = {"h": 3600, "m": 60, "s": 1}
    if unit not in factors:
        raise ValueError("unknown unit")
    return amount * factors[unit]
