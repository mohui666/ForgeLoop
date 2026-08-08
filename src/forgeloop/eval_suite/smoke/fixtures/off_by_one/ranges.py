def inclusive_values(start: int, end: int) -> list[int]:
    if start > end:
        return []
    return list(range(start, end))
