def status_category(status: int) -> str:
    if 200 < status < 300:
        return "success"
    if 400 < status < 500:
        return "client_error"
    if 500 < status < 600:
        return "server_error"
    return "other"
