def dict_repr(self):
    return self.__dict__.__repr__()

def dict_str(self):
    return self.__dict__.__str__()

def round_price(price: float, exchange_precision: int) -> float:
    """Round prices to the nearest cent.

    Args:
        price (float)
        exchange_precision (int): number of decimal digits for exchange price

    Returns:
        float: The rounded price.
    """
    return round(price, exchange_precision)

