def display_name(profile: dict) -> str:
    nickname = profile.get("nickname").strip()
    if nickname:
        return nickname
    return profile.get("name").strip()
