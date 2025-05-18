from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Any

from scripts.commons.logging_config import get_logger
from scripts.commons.tokens_metadata_scraper import fetch_current_token_price

log = get_logger()

ETH_TOKENS_WHITELIST = [
    "ETH",    # Ethereum (native token)
    "WETH",   # Wrapped Ethereum
    "USDT",   # Tether
    "USDC",   # USD Coin
    "USDS"    # USDS
    "sUSDS"   # Staked USDS
    "DAI",    # Dai Stablecoin
    "BNB",    # Binance Coin
    "LINK",   # Chainlink
    "UNI",    # Uniswap
    "AAVE",   # Aave
    "MKR",    # Maker
    "CRV",    # Curve DAO Token
    "COMP",   # Compound
    "SNX",    # Synthetix
    "SHIB",   # Shiba Inu
    "GRT",    # The Graph
    "LDO",    # Lido DAO
    "RPL",    # Rocket Pool
    "RETH",   # Rocket Pool ETH
    "STETH",  # Lido Staked ETH
    "WSTETH", # Wrapped stETH
    "ARB",    # Arbitrum
    "OP",     # Optimism
    "DYDX",   # dYdX
    "ENS",    # Ethereum Name Service
    "RENDER", # Render Token
    "PEPE",   # Pepe
    "IMX",    # Immutable X
    "FRAX",   # Frax Share
    "MANA",   # Decentraland
    "SUSHI",  # SushiSwap
    "stETH",  # Lido Staked ETH
    "osETH",  # StakeWise Staked ETH
    "ankrETH" # Ankr Staked ETH
]

ETH_TOKENS_BLACKLIST = [
    "cWETHv3",
    "aEthrETH",
    "iETHv2",
    "brktETH",
    "qUSDT",
    "AIC",
    "inwstETHs"
    ]
