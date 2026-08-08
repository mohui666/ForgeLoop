def build_query(params: dict) -> str:
    return "&".join(f"{key}={value}" for key, value in params.items())
