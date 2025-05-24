from scripts.commons.logging_config import get_logger

log = get_logger()

ETH_TOKENS_WHITELIST = [
    "ETH",    # Ethereum (native token)
    "WETH",   # Wrapped Ethereum
    "USDT",   # Tether
    "USDC",   # USD Coin
    "USDS",    # USDS
    "sUSDS",   # Staked USDS
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
    "ankrETH", # Ankr Staked ETH
    "ETHX",        # Lido or Stader derivative, depending on context
    "CAKE",        # PancakeSwap token, legit
    "WBTC",        # Wrapped Bitcoin
    "OETH",        # Origin ETH, a legit LST
    "SUSD",        # Synthetix USD
    "sUSDe",       # Likely Ethena’s synthetic USD
    "EZETH",       # Renzo’s LST — relatively new, but not a scam
    "RSETH",       # KelpDAO or similar re-staking ETH
    "AGETH",       # Asymmetry Finance ETH
    "MATIC",       # Polygon — blue chip
    "POL",         # Polygon’s governance token (newer than MATIC)
    "GHO",         # Aave's own stablecoin
    "METIS"

]

# ETHX, , aEthUSDT,aEthUSDC,aEthDAI,CAKE,WBTC,mooConvexCVX-ETH,TOKEN,VR,PUFFER,IMT,MEME,ETHG,EIGEN,USDf,USDC/USDf,USDC/USDf-gauge,OETH,SUSD,sUSDe,sUSD+sUSDe-gauge,EZETH,RSETH,hgETH,AGETH,STRDY,MATIC,POL,Aave GHO/USDT/USDC,S*USDT,GHO

ETH_TOKENS_BLACKLIST = [
    "aEthUSDT",
    "aEthUSDC",
    "aEthDAI",
    "mooConvexCVX-ETH",  # Beefy vault on Convex/CVX-ETH LP — complex, but real
    "sUSD+sUSDe-gauge",  # Gauge token for LP, valid if platform is valid
    "Aave GHO/USDT/USDC",  # Aave GHO stablecoin pool
    "hgETH",       # Swell Network token (Hyperscale gauge ETH)
    "STRDY",       # Possibly Sturdy Finance token
    "USDC/USDf",   # Liquidity pool token
    "USDC/USDf-gauge",  # LP gauge, valid on Curve-like platformа
    "cWETHv3",
    "aEthrETH",
    "iETHv2",
    "brktETH",
    "qUSDT",
    "AIC",
    "inwstETHs",
    "claim rewards on t.ly/ethblaze",
    "TOKEN",  # Generic placeholder, likely junk or scam
    "VR",     # Too generic, ambiguous (could be fake or very low cap)
    "PUFFER", # Airdropped token — possibly legit, but still controversial or unproven
    "IMT",    # Could be one of many unrelated projects; often associated with spam tokens
    "MEME",   # Meme tokens can be legitimate or scammy — context matters; default to blacklist
    "ETHG",   # Very ambiguous, low recognition — could be fake
    "EIGEN",  # EigenLayer token – not launched officially as of 2024, impersonated a lot
    "USDf",   # Could be fake/forked — several impersonators
    "S*USDT", # Ambiguous — unclear what protocol it's from (synthetic/star-token? spoofed?)

    ]
