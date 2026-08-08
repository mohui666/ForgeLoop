import re


def slugify(value: str) -> str:
    cleaned = value.strip().lower().replace(" ", "-").replace("_", "-")
    return re.sub(r"[^a-z0-9-]", "", cleaned)
