import json
from datetime import datetime

from scripts.commons.known_token_list import ETH_TOKENS_WHITELIST
from scripts.commons.logging_config import get_logger
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
    prices: dict[str, float] = {"ETH": 2550.6769170918, "WETH": 2551.0539540149, "USDT": 1.0002490121, "USDC": 0.9997909069, "USDS": 0.9998031023, "sUSDS": 1.0524623402,
                                "DAI": 1.0000516287, "BNB": 673.3579776053, "LINK": 15.5877569648, "UNI": 6.2003065356, "AAVE": 269.0561608443, "MKR": 1670.1847888145,
                                "CRV": 0.7861946698, "COMP": 42.6079949999, "SNX": 0.7841156312, "SHIB": 1.43893e-05, "GRT": 0.1095344284, "LDO": 0.8654202453, "RPL": 4.9570709833,
                                "RETH": 2898.0353211156, "STETH": 2548.1659172221, "WSTETH": 3069.5535298956, "ARB": 0.3956256815, "OP": 0.7606708901, "DYDX": 0.636126718,
                                "ENS": 22.2712115612, "RENDER": 4.7529524416, "PEPE": 1.37889e-05, "IMX": 0.6425103534, "FRAX": 3.1014281752, "MANA": 0.3141433423,
                                "SUSHI": 0.74043714, "stETH": 2548.1659172221, "osETH": 2668.7389880051, "ankrETH": 3062.7074020394, "ETHX": 2707.4987467148, "CAKE": 2.3728985568,
                                "WBTC": 108616.7048910467, "OETH": 2549.0830592186, "SUSD": 0.9644345684, "sUSDe": 1.1741688543, "EZETH": 2676.2049186263, "RSETH": 2663.5898337084,
                                "AGETH": 2622.2354185635, "MATIC": 0.2359589754, "POL": 0.2363016479, "GHO": 0.9992824965, "METIS": 19.7344257474, "PENDLE": 4.3889351083,
                                "USDe": 1.0008131638, "CRVUSD": 0.9998659737, "COW": 0.4140353539, "USD0": 0.997991152, "USUAL": 0.1327251497, "CBBTC": 108791.7260984395,
                                "LBTC": 108478.889811304, "EURC": 1.1354157453, "SAFE": 0.5233332305, "PYUSD": 0.9991614125, "DEUSD": 1.0000158211, "CVX": 3.3880037586}

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
