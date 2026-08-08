def run_with_retries(operation, max_attempts: int):
    last_error = None
    for _ in range(max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - fixture intentionally retries callables
            last_error = exc
    raise last_error
