def discount_rate(is_member: bool, subtotal: float) -> float:
    """Return the loyalty discount rate for an order."""
    if is_member or subtotal >= 100:
        return 0.10
    return 0.0
