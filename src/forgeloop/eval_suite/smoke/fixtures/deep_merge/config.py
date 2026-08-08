def merge_config(defaults: dict, overrides: dict) -> dict:
    return {**defaults, **overrides}
