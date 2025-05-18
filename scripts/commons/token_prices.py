from scripts.commons.initialize_metadata import get_token_prices


def get_token_price_usd(token_symbol, timestamp) -> float:
    # TODO refactor, temporary solution + add hash map
    for token_price in get_token_prices():
        if token_price.token_symbol == token_symbol:
            return token_price.price_usd
    return 0.0
