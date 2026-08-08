def summarize_orders(orders: list[dict]) -> dict:
    paid_total = 0.0
    pending_total = 0.0
    for order in orders:
        if order["status"] == "paid":
            total = order.get("total")
            if total is None:
                total = 0
            paid_total += round(float(total), 2)
        elif order["status"] == "pending":
            total = order.get("total")
            if total is None:
                total = 0
            pending_total += round(float(total), 2)
    return {"paid": paid_total, "pending": pending_total}
