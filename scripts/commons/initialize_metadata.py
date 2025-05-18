from datetime import datetime

from scripts.commons.logging_config import get_logger
from scripts.commons.model import TokenPriceUsd
from scripts.commons.known_token_list import ETH_TOKENS_WHITELIST
from scripts.commons.tokens_metadata_scraper import fetch_current_token_price

SUPPORTED_TOKEN_PRICES: set[TokenPriceUsd] = set()
log = get_logger()


# TODO refactor, temporary solution - i don't like this approach

def init():
    # initialize_token_prices()
    initialize_token_prices_dummy()
    log.info("Initialization finished")


def initialize_token_prices():
    global SUPPORTED_TOKEN_PRICES
    log.info("Initializing token prices")
    for token_symbol in ETH_TOKENS_WHITELIST:
        (timestamp, price_usd) = fetch_current_token_price(token_symbol)
        SUPPORTED_TOKEN_PRICES.add(TokenPriceUsd(token_symbol, timestamp, price_usd))
        log.info(f"Current price for token {token_symbol}={price_usd} USD")
    log.info(f"Initialized prices for all {len(SUPPORTED_TOKEN_PRICES)} whitelisted tokens")


def initialize_token_prices_dummy():
    test_prices = set()
    timestamp = int(datetime.now().timestamp())

    # Sample prices for common tokens (roughly based on May 2025 values)
    price_map = {
        "ETH": 2506.53,
        "WETH": 2505.12,
        "USDT": 1.00,
        "USDC": 1.00,
        "DAI": 1.00,
        "BNB": 545.87,
        "LINK": 17.92,
        "UNI": 7.85,
        "AAVE": 95.23,
        "MKR": 2890.45,
        "CRV": 0.52,
        "COMP": 62.15,
        "SNX": 3.72,
        "SHIB": 0.000028,
        "GRT": 0.24,
        "LDO": 3.15,
        "RPL": 29.84,
        "RETH": 2650.33,
        "STETH": 2515.67,
        "WSTETH": 2520.43,
        "ARB": 1.45,
        "OP": 2.78,
        "DYDX": 2.93,
        "ENS": 21.37,
        "RENDER": 8.42,
        "PEPE": 0.000012,
        "IMX": 1.87,
        "FRAX": 1.00,
        "MANA": 0.48,
        "SUSHI": 0.89,
        "stETH": 2515.67,  # Same as STETH above
        "osETH": 2530.12,
        "ankrETH": 2515.90,
        "USDS":1.00,
        "sUSDS": 1.05,
    }

    # Create TokenPriceUsd objects for each token
    for token_symbol in ETH_TOKENS_WHITELIST:
        # Use the price from the map, or a default price if not found
        price = price_map.get(token_symbol, 1.0)
        test_prices.add(TokenPriceUsd(token_symbol, timestamp, price))
        log.warning(f"Added test price for {token_symbol}: {price} USD")

    log.warning(f"Initialized test prices for {len(test_prices)} tokens")
    global SUPPORTED_TOKEN_PRICES
    SUPPORTED_TOKEN_PRICES = test_prices


def get_token_prices():
    return SUPPORTED_TOKEN_PRICES
