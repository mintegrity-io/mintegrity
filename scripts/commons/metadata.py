import json
from datetime import datetime

from scripts.commons.logging_config import get_logger
from scripts.commons.model import TokenPriceUsd
from scripts.commons.known_token_list import ETH_TOKENS_WHITELIST
from scripts.commons.tokens_metadata_scraper import fetch_current_token_prices

CURRENT_TOKEN_PRICES: dict[str, float] = {}
log = get_logger()


def init():
    # print_current_token_prices()
    initialize_prefetched_token_prices()
    log.info("Initialization finished")


def print_current_token_prices():
    log.info("Fetching current token prices")
    prices: dict[str, float] = {}
    for token, (timestamp, price_usd) in fetch_current_token_prices(ETH_TOKENS_WHITELIST).items():
        prices[token] = price_usd
        log.info(f"Current price for token {token}={price_usd} USD")
    print(json.dumps(prices))


def initialize_prefetched_token_prices():
    token_prices = {}
    timestamp = int(datetime.now().timestamp())

    # Sample prices for common tokens (roughly based on May 2025 values)
    prices: dict[str, float] = {
        "ETH": 2460.1764150835,
        "WETH": 2459.7609416328,
        "USDT": 1.0002672686,
        "USDC": 0.9998815963,
        "USDS": 0.9997995701,
        "sUSDS": 1.0519653774,
        "DAI": 1.0000100212,
        "BNB": 644.3198420914,
        "LINK": 15.3094630669,
        "UNI": 5.7371037899,
        "AAVE": 257.3119367431,
        "MKR": 1691.2142434135,
        "CRV": 0.697105203,
        "COMP": 42.1337032432,
        "SNX": 0.7938024009,
        "SHIB": 0.0000142115,
        "GRT": 0.1068151425,
        "LDO": 0.8553097577,
        "RPL": 4.921903396,
        "RETH": 2796.608752254,
        "STETH": 2461.7115394813,
        "WSTETH": 2954.6327652068,
        "ARB": 0.3830007132,
        "OP": 0.6955829503,
        "DYDX": 0.6287703673,
        "ENS": 21.568259934,
        "RENDER": 4.541450556,
        "PEPE": 0.0000127884,
        "IMX": 0.6299072055,
        "FRAX": 3.8132843029,
        "MANA": 0.3138147264,
        "SUSHI": 0.6926182693,
        "stETH": 2461.7115394813,
        "osETH": 2563.0097893333,
        "ankrETH": 2942.5089684168,
        "ETHX": 2595.6833027338,
        "CAKE": 2.2491629446,
        "WBTC": 105463.232589684,
        "OETH": 2461.5921092597,
        "SUSD": 0.9593973087,
        "sUSDe": 1.1737593145,
        "EZETH": 2584.341881116,
        "RSETH": 2563.1138572713,
        "AGETH": 2528.9488837171,
        "MATIC": 0.2285508354,
        "POL": 0.2280123381,
        "GHO": 0.9990184513,
        "METIS": 19.43,
    }

    # Store prices directly in the dictionary
    for token_symbol in ETH_TOKENS_WHITELIST:
        # Use the price from the map
        price = prices.get(token_symbol)
        if price is None:
            raise Exception(f"Token {token_symbol} should be supported but price is not present in test prices")
        token_prices[token_symbol] = price
        log.warning(f"Added test price for {token_symbol}: {price} USD")

    log.warning(f"Initialized test prices for {len(token_prices)} tokens")
    global CURRENT_TOKEN_PRICES
    CURRENT_TOKEN_PRICES = token_prices


def get_token_price_usd(token_symbol: str, timestamp: str) -> float:
    # Currently this function returns the current price from the pre-fetched token prices, timestamp is not used
    if token_symbol in CURRENT_TOKEN_PRICES:
        return CURRENT_TOKEN_PRICES[token_symbol]
    else:
        log.warning(f"Token {token_symbol} not found in supported token prices. Returning 0.0")
        return 0.0
