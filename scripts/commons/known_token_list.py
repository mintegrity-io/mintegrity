from scripts.commons.logging_config import get_logger

log = get_logger()

TOKENS_WHITELIST = [
    "BTC",  # Bitcoin (native token)
    "ETH",  # Ethereum (native token)
    "WETH",  # Wrapped Ethereum
    "USDT",  # Tether
    "USDC",  # USD Coin
    "USDS",  # USDS
    "sUSDS",  # Staked USDS
    "DAI",  # Dai Stablecoin
    "BNB",  # Binance Coin
    "LINK",  # Chainlink
    "UNI",  # Uniswap
    "AAVE",  # Aave
    "MKR",  # Maker
    "CRV",  # Curve DAO Token
    "COMP",  # Compound
    "SNX",  # Synthetix
    "SHIB",  # Shiba Inu
    "GRT",  # The Graph
    "LDO",  # Lido DAO
    "RPL",  # Rocket Pool
    "RETH",  # Rocket Pool ETH
    "STETH",  # Lido Staked ETH
    "WSTETH",  # Wrapped stETH
    "ARB",  # Arbitrum
    "OP",  # Optimism
    "DYDX",  # dYdX
    "ENS",  # Ethereum Name Service
    "RENDER",  # Render Token
    "PEPE",  # Pepe
    "IMX",  # Immutable X
    "FRAX",  # Frax Share
    "MANA",  # Decentraland
    "SUSHI",  # SushiSwap
    "stETH",  # Lido Staked ETH
    "osETH",  # StakeWise Staked ETH
    "ankrETH",  # Ankr Staked ETH
    "ETHX",  # Lido or Stader derivative, depending on context
    "CAKE",  # PancakeSwap token, legit
    "WBTC",  # Wrapped Bitcoin
    "OETH",  # Origin ETH, a legit LST
    "SUSD",  # Synthetix USD
    "sUSDe",  # Likely Ethena’s synthetic USD
    "EZETH",  # Renzo’s LST — relatively new, but not a scam
    "RSETH",  # KelpDAO or similar re-staking ETH
    "AGETH",  # Asymmetry Finance ETH
    "MATIC",  # Polygon — blue chip
    "POL",  # Polygon’s governance token (newer than MATIC)
    "GHO",  # Aave's own stablecoin
    "METIS",
    "PENDLE",
    "USDe",
    "CRVUSD",
    "COW",
    "USD0",
    "USUAL",
    "CBBTC",
    "LBTC",
    "EURC",
    "SAFE",
    "PYUSD",
    "DEUSD",
    "CVX",
]

# ETHX, , aEthUSDT,aEthUSDC,aEthDAI,CAKE,WBTC,mooConvexCVX-ETH,TOKEN,VR,PUFFER,IMT,MEME,ETHG,EIGEN,USDf,USDC/USDf,USDC/USDf-gauge,OETH,SUSD,sUSDe,sUSD+sUSDe-gauge,EZETH,RSETH,hgETH,AGETH,STRDY,MATIC,POL,Aave GHO/USDT/USDC,S*USDT,GHO

ETH_TOKENS_BLACKLIST = ['DOLA', 'yCRV', 'DBR', 'SPX', 'BITCOIN', 'FLUID', 'RSUP', 'TITANX', 'ENA', 'USUALX', 'AMPL', 'CHEX', 'USR', 'dEURO', 'BAL', 'PIN', 'ONDO', 'ULTI', 'MOG',
                        'STG', 'GNO', 'ASF', 'WTAO', 'SHIRO', 'CVXCRV', 'weETH', 'XAUt', 'OHM', 'RYU', 'EURA', 'EUL', 'RSR', 'tETH', 'TBTC', 'HYPER', 'SCRVUSD', 'CELL', 'ADS',
                        'SDAI', 'weETHs', 'DINERO', 'PAXG', 'HEX', 'VANA', 'zunUSD', 'fxUSD', 'CPOOL', 'MAXIMUSA', 'DRAGONX', 'FXN', 'SOL', 'ETHDYDX', 'frxUSD', 'beraSTONE',
                        'ZCHF', 'MAGA', 'APU', 'RIO', 'P', 'SPOT', 'eEIGEN', 'ZIG', 'ACX', 'OMIKAMI', 'MAZA', 'EETH', 'ANDY', 'SYRUP', 'TONCOIN', 'TRAC', 'NEAR', 'LCX', 'DEVVE',
                        'reUSD', 'MORPHO', 'MPL', 'sENA', 'EBTC', 'LAKE', 'EURe', 'FRXETH', 'GROK', 'INV', 'RLB', 'EURt', 'RAIL', 'SKY', 'MOVE', 'NEIRO', 'PEPECOIN', 'USDa',
                        'sfrxUSD', 'FUEL', 'pxETH', 'WAMPL', 'PUFETH', 'GMAC', 'RSWETH', 'COCORO', 'WOLF', 'ORX', 'BOBO', 'SPELL', 'Kekiusa', 'XEN', 'PSY', 'YUSD', 'DSYNC', 'FXS',
                        'USDx', 'BIO', 'VITA', 'IXS', 'PAAL', 'WOO', 'BNT', 'ERN', 'CULT', 'ALCX', '1INCH', 'liquidBeraETH', 'SFRXETH', 'X28', 'WQUIL', 'PRISMA', 'PRIME', 'RSC',
                        'ZYN', 'INJ', 'MUBI', 'OCEAN', 'LORDS', 'WFTM', 'XSGD', 'PSP', 'MOODENG', 'ALVA', 'ELX', 'EAI', 'TORN', 'cvxPrisma', 'USX', 'SHAO', 'aEthWETH', 'ALPH',
                        'AST', 'XMW', 'SWETH', 'ASTO', 'USDP', 'MNT', 'sdFXN', 'QF', 'OM', 'TKX', 'SYNT', 'BUSINESS', 'BEEBEE', 'KEKEC', 'cUSDO', 'AXGT', 'NEXO', 'cvxFXN', 'AURA',
                        'REZ', 'WFIO', 'ETHFI', 'STRK', 'FLUX', 'lvlUSD', 'T', 'SERV', 'EURS', 'ZRO', 'GATSBY', 'BEAM', 'stdeUSD', 'ELON', 'DF', 'MKUSD', 'clevCVX', 'BANANA',
                        'MIPRAMI', 'LUSD', 'XAI', 'LRT2', 'WEPE', 'MORPH', 'XYO', 'IOTX', 'TGC', 'cbETH', 'TINC', 'L3', 'MATRIX', 'REQ', 'AMP', 'LQTY', 'KUKURU', 'PAL', 'AIUS',
                        'INF', 'HILO', 'RLP', 'PNK', 'CNDL', 'SDCRV', 'BUBBLE', 'MON', 'sdPENDLE', 'TEMPLE', 'GTC', 'EDGE', 'KEK', 'SDT', 'SWELL', 'EMP', 'MIM', 'st-yCRV', 'VUSD',
                        'HIGH', 'POWER', 'BLZ', 'DOGE', 'NPC', 'WPLS', 'GME', 'stkAAVE', 'LOJBAN', 'ZETA', 'TARA', 'wstUSR', 'PRQ', 'KP3R', 'EGOLD', 'LPT', 'SHFL', 'IPOR', 'AUTOS',
                        'SD', 'LD', 'PNDC', 'MIR', 'BADGER', 'BLAZE', 'FTT', 'XCN', 'PT', 'BASEDAI', 'RUG', 'GUSD', 'HONK', 'INDEX', 'xWBTC', 'KEX', 'TARIFF', 'FLX', 'BUNNI',
                        'GALA', 'ANKR', 'JYAI', 'BONE', 'BONK', 'ID', 'dYFI', 'EUROE', 'AIOZ', 'OSAK', 'BAT', 'PORK', 'SQT', 'CXT', 'FORT', 'PUMPBTC', 'VDO', 'GROW', 'TROLL',
                        'SDAO', 'WNXM', 'WXRP', 'ITGR', 'OPN', 'BFX', 'GORA', 'ynETHx', 'wethrsup', 'W', 'POLS', 'WNCN', 'OGN', 'GONDOLA', '$MICRO', 'TALK', 'WBETH', 'MAX', 'AGRS',
                        'AVA', 'stBTC', 'DOG', 'MSTR', 'TOPIA', 'IQ', 'SWISE', 'CHARLIE', 'USDL', 'ZGEN', 'FDUSD', 'MOCHA', 'KEKIUS', 'NATI', 'sDOLA', 'BYB', 'METH', 'yUSD', 'UST',
                        'RCH', 'CHART', 'KENDU', 'GEOFF', 'RENBTC', 'GURU', 'sUSDa', 'INFRA', 'PYR', 'MOCA', 'CARROT', 'SPECTRE', 'DOPE', 'FOLD', 'PEAS', 'SMOL', 'AUCTION',
                        'VYPER', 'CUDOS', 'SAGE', 'YGG', 'BIOHACK', 'HOP', 'BLSEYE?', 'RADAR', 'CUBE', 'RARE', 'SDL', 'NAOS', 'VAI', 'BMP', 'NAI', 'agUSD', '0x0', 'COTI', 'CMETH',
                        'MARV', 'TUSD', 'CRAMER', 'MYSTERY', 'SEI', 'ATH', 'ALD', 'ARKM', 'VOLT', 'FET', 'AIKEK', 'JOE', 'MUZZ', 'ABT', 'DUEL', 'LADYS', 'TURBO', 'ISLAND', 'DATA',
                        'SWAG', 'DRIP', 'SKL', 'RLC', 'UMA', 'stkGHO', 'BCAT', 'BBTC', 'BLUR', 'MARSH', 'NEURON', 'VITA-FAST', 'ZKJ', 'EVEAI', 'PAR', 'RED', 'USUALUSDC+', 'ECL',
                        'SUDO', 'HDRN', 'RAI', 'SDEX', 'DEXTF', 'PAPPLE', 'syrupUSDC', 'PORTAL', 'FMC', 'Bold', 'MAPO', 'GREETER', 'TRU', 'ARC', 'ALT', 'DPI', 'FOX', 'POOL', 'VOW',
                        'MAV', 'deUSD/DOLA', 'DOLA/USR', 'DOLA-sUSDe', 'reusdscrv', 'dola-save', 'DOLA-sUSDS', 'reusdsfrx', 'ynETH', 'yn-ETH/LSD', 'alfrxETH-f', 'ORN', 'KEROSENE',
                        'PRO', 'SMT', 'FAME', 'WOJAK', 'BYTES', 'eUSD', 'ORDS', 'apxETH', 'LUNA', 'VIX', 'FLRBRG', 'BeeARD', 'TOWELI', 'GEL', 'BERRY', 'VANRY', 'LEDGER', 'UNIBOT',
                        'LAMBO', 'DOGEFATHER', 'VERI', 'DFDX', 'MUSE', 'SNSY', 'PEPO', 'BRETT', 'PSPS', 'WEL', 'OLAS', 'yvCurve-wethrsup-f', 'SAND', 'csUSDL', 'wUSDL', 'USDtb',
                        'HOBA', 'KAP', 'BICO', 'MUMU', 'GEAR', 'SIR', 'WRBNT', 'ILV', 'alETH', 'ALI', 'ETH+', 'JESUS', 'TREE', 'LINA', 'TERM', 'TRUF', 'sdBal', 'B-80BAL-20WETH',
                        'AURABAL', 'msETH', 'SANJI', 'yETH', 'TOKE', 'ORAI', 'FINALE', 'LYXe', 'ZUN', 'Ghibli', 'ONDOAI', 'DRGN', 'PKEX', 'BYTE', 'WLUNC', 'FROG', 'DAO', 'SUPER',
                        'ICOM', 'MINIDOGE', 'BOOMKIN', 'STEVE', 'VitalikGhibli', 'pwease', 'HASHAI', 'baoETH', 'PEPU', 'KEL', 'POND', 'aEthwstETH', 'VXR', 'slvlUSD', 'RSP',
                        'ICELAND', 'WHITE', 'DIP', 'ANYONE', 'QNT', 'GOHM', 'STFX', 'SILV2', 'CLEV', 'USD0++']
