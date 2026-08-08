def page_window(page: int, page_size: int) -> tuple[int, int]:
    if page < 0:
        raise ValueError("page must not be negative")
    if page_size > 100:
        raise ValueError("page_size is too large")
    start = (page - 1) * page_size
    return start, start + page_size
