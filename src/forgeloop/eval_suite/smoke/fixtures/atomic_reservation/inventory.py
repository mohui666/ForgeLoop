class Inventory:
    def __init__(self, stock: dict[str, int]):
        self.stock = dict(stock)

    def reserve(self, requested: dict[str, int]) -> None:
        for sku, quantity in requested.items():
            available = self.stock.get(sku, 0)
            if available < quantity:
                raise ValueError(f"insufficient stock for {sku}")
            self.stock[sku] = available - quantity
