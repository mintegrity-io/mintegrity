#!/usr/bin/env python3
"""
Enhanced script for analysis of the Rocket Pool addresses using pre-built graph

How it worksФункциональность:
1. loads pre-built graph from files/rocket_pool_full_graph_90_days.json
2. extracts addresses which directly interacted with Rocket Pool contracts (by default)
or all addresses from the graph (--all-addresses option)
3. retrieves enhances statistics of each address of the last 365 days throght Etherscan and Rocket Pool Subgraph
    - total TO, max transaction value;
    - total number of interactions with wallets and SC;
    - wallet creation date;
    - activity patterns, gas fees;
    - daily and monthly activity.
4. using existing pricing modules to obtain historical tocken prices
5. saves outputs in JSON and CSV format

IMPORTANT: the graph contains interaction over 90 days but the script gathers statistics over 365 days
"""

import json
import csv
import requests
import time
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from tqdm import tqdm

# Adding project's root to the sys.path for the module import
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent  # mintegrity/scripts/stats_vis/ -> mintegrity/
sys.path.insert(0, str(project_root))

# Importing existing project (absolute paths)
from scripts.commons import metadata
from scripts.commons.logging_config import get_logger
from scripts.commons.model import Address, AddressType
from scripts.commons.tokens_metadata_scraper import fetch_current_token_prices
from scripts.commons.known_token_list import ETH_TOKENS_WHITELIST
from scripts.graph.model.transactions_graph import TransactionsGraph, NodeType
from scripts.graph.util.transactions_graph_json import load_graph_from_json

log = get_logger()


@dataclass
class WalletStatistics:
    """Расширенная статистика адреса за 365 дней"""
    address: str
    address_type: str = None  # "wallet" or "contract"

    # Basic info
    creation_date: Optional[str] = None
    first_transaction_date: Optional[str] = None
    last_transaction_date: Optional[str] = None
    wallet_age_days: Optional[int] = None

    # 365 day values
    total_volume_usd_365d: Optional[float] = None
    average_volume_usd_365d: Optional[float] = None
    max_volume_usd_365d: Optional[float] = None
    median_volume_usd_365d: Optional[float] = None

    # 365 days activity
    total_transactions_365d: Optional[int] = None
    outgoing_transactions_365d: Optional[int] = None
    incoming_transactions_365d: Optional[int] = None

    # 365 days interactions
    unique_addresses_interacted_365d: Optional[int] = None
    wallet_interactions_365d: Optional[int] = None
    contract_interactions_365d: Optional[int] = None

    # 365 days activity patterns
    active_days_365d: Optional[int] = None
    avg_daily_volume_usd_365d: Optional[float] = None
    max_daily_volume_usd_365d: Optional[float] = None
    most_active_month_365d: Optional[str] = None

    # Gas fees over 365 days
    total_gas_used_365d: Optional[int] = None
    total_gas_fees_usd_365d: Optional[float] = None
    avg_gas_price_gwei_365d: Optional[float] = None

    # auxillary information
    token_prices_used: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class RocketPoolAnalyzer:
    """Simpified Rocket Pool addresses analyzer"""

    def __init__(self,
                 graph_file_path: str = None,
                 output_dir: str = None,
                 max_workers: int = 3,
                 analyze_all_addresses: bool = False):

        # Paths corresponding to the project's root directory
        if graph_file_path is None:
            graph_file_path = str(project_root / "files" / "rocket_pool_full_graph_90_days.json")
        if output_dir is None:
            output_dir = str(project_root / "files" / "rocket_pool_analysis")

        self.graph_file_path = Path(graph_file_path)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.analyze_all_addresses = analyze_all_addresses
        self.price_cache = {}  # historic prices cache {token_symbol-timestamp: price_usd}

        # output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # metadata initialization
        metadata.init()

        # obtain current token prices using existing module (fallback)
        self.current_token_prices = self._fetch_current_prices()

        log.info(f"Initialized Rocket Pool Analyzer")
        log.info(f"Graph file: {self.graph_file_path}")
        log.info(f"Output directory: {self.output_dir}")
        log.info(f"Analysis mode: {'All addresses' if analyze_all_addresses else 'Only Rocket Pool interactions'}")
        log.info(f"Loaded fallback prices for {len(self.current_token_prices)} tokens")

    def _fetch_current_prices(self) -> Dict[str, float]:
        try:
            token_prices_with_timestamps = fetch_current_token_prices(ETH_TOKENS_WHITELIST)

            current_prices = {}
            for token, (timestamp, price) in token_prices_with_timestamps.items():
                current_prices[token] = price
                log.debug(f"Loaded fallback price for {token}: ${price:.4f}")

            return current_prices

        except Exception as e:
            log.warning(f"Failed to fetch current prices via API: {e}")
            log.info("Falling back to metadata prices")

            # Fallback: using prices from the metadata
            fallback_prices = {}
            for token in ETH_TOKENS_WHITELIST:
                price = metadata.get_token_price_usd(token, str(int(time.time())))
                if price > 0:
                    fallback_prices[token] = price

            return fallback_prices

    def get_historical_token_price(self, token_symbol: str, timestamp: int) -> float:
        """
        Obtains historical token's prices at the moment
        uses cache in combination with existing modules and external API
        
        Args:
            token_symbol: Token symbol (ETH, BTC, USDC, etc.)
            timestamp: Unix timestamp
            
        Returns:
            current token's price in USD
        """

        # create key's cache
        cache_key = f"{token_symbol.upper()}-{timestamp}"

        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        # first trying using metadata
        try:
            price = metadata.get_token_price_usd(token_symbol, str(timestamp))
            if price > 0:
                self.price_cache[cache_key] = price
                return price
        except Exception as e:
            log.debug(f"Metadata price lookup failed for {token_symbol}: {e}")

        # if no historical prices in metadata, use external API
        try:
            # token mapping via Coinbase
            token_to_pair = {
                'ETH': 'ETH-USD',
                'BTC': 'BTC-USD',
                'WETH': 'ETH-USD',  # WETH = ETH
                'USDT': 'USDT-USD',
                'USDC': 'USDC-USD',
                'DAI': 'DAI-USD',
                'LINK': 'LINK-USD',
                'UNI': 'UNI-USD',
                'AAVE': 'AAVE-USD',
                'MKR': 'MKR-USD',
                'CRV': 'CRV-USD',
                'COMP': 'COMP-USD',
                'SNX': 'SNX-USD',
                'GRT': 'GRT-USD',
                'LDO': 'LDO-USD',
                'MATIC': 'MATIC-USD',
                'SHIB': 'SHIB-USD'
            }

            pair = token_to_pair.get(token_symbol.upper())
            if not pair:
                # if a token is not supported, use ETH as fallback
                log.warning(f"Token {token_symbol} not supported, using ETH price as fallback")
                if token_symbol.upper() != 'ETH':
                    return self.get_historical_token_price('ETH', timestamp)
                pair = 'ETH-USD'

            # Coinbase Advanced Trade API endpoint for historical prices
            # obtaining price data for requested timestamp (±1 hout)
            start_time = timestamp - 3600
            end_time = timestamp + 3600

            url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
            params = {
                'start': datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
                'end': datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
                'granularity': 3600  # 1 hr candles
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            candles = response.json()

            if not candles:
                # trying to get wider time range in case of no data
                log.warning(f"No candle data for {pair} at {timestamp}, trying wider range")
                start_time = timestamp - 86400  # 1 day before
                end_time = timestamp + 86400  # 1 after

                params = {
                    'start': datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
                    'end': datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
                    'granularity': 86400  # daily candles
                }

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                candles = response.json()

            if not candles:
                log.warning(f"No historical data available for {pair} at timestamp {timestamp}")
                # Fallback: using current price
                return self.get_current_token_price(token_symbol)

            # Coinbase data format [timestamp, low, high, open, close, volume]
            # using closing price of the nearest candle
            closest_candle = min(candles, key=lambda x: abs(x[0] - timestamp))
            price = float(closest_candle[4])  # close price

            # caching the result
            self.price_cache[cache_key] = price

            log.debug(f"Historical price for {token_symbol} at {timestamp}: ${price:.4f}")
            return price

        except requests.exceptions.RequestException as e:
            log.warning(f"Network error getting historical price for {token_symbol}: {e}")
            return self.get_current_token_price(token_symbol)
        except Exception as e:
            log.warning(f"Error getting historical price for {token_symbol} at {timestamp}: {e}")
            return self.get_current_token_price(token_symbol)

    def get_current_token_price(self, token_symbol: str) -> float:
        """Getting current token's price as a fallback option using existing modules"""

        # preloaded prices first
        if token_symbol in self.current_token_prices:
            return self.current_token_prices[token_symbol]

        # mapping for the fallback prices
        token_symbol_upper = token_symbol.upper()
        if token_symbol_upper in self.current_token_prices:
            return self.current_token_prices[token_symbol_upper]

        # for WETH using ETH price
        if token_symbol_upper == 'WETH' and 'ETH' in self.current_token_prices:
            return self.current_token_prices['ETH']

        # trying metadata
        try:
            price = metadata.get_token_price_usd(token_symbol, str(int(time.time())))
            if price > 0:
                return price
        except Exception as e:
            log.debug(f"Metadata current price lookup failed for {token_symbol}: {e}")

        log.warning(f"Price not found for token {token_symbol}, using fallback")
        return self.current_token_prices.get('ETH', 2500.0)  # Fallback to ETH price

    def get_token_price_usd(self, token_symbol: str, timestamp: Optional[str] = None) -> float:
        """
        Getting token price in USD for a certaing timestamp
        
        Args:
            token_symbol: (ETH, WETH, USDC, etc.)
            timestamp: Unix timestamp (string или int)
            
        Returns:
            USD token price
        """
        if timestamp:
            try:
                # timestamp
                if isinstance(timestamp, str):
                    if timestamp.isdigit():
                        timestamp_int = int(timestamp)
                    else:
                        # ISO format
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp_int = int(dt.timestamp())
                else:
                    timestamp_int = int(timestamp)

                # historical price
                return self.get_historical_token_price(token_symbol, timestamp_int)

            except Exception as e:
                log.warning(f"Error parsing timestamp {timestamp}: {e}")

        # Fallback: current price
        return self.get_current_token_price(token_symbol)

    def calculate_transaction_value_usd(self, value_token: float, token_symbol: str, timestamp_str: str) -> tuple[float, Dict[str, float]]:
        """
        Getting transaction cost in USD at the time
        
        Returns:
            tuple: (value_usd, token_prices_used)
        """
        try:
            # getting historical price of a token at the moment
            token_price = self.get_token_price_usd(token_symbol, timestamp_str)

            # converting to USD
            value_usd = value_token * token_price

            prices_used = {token_symbol: token_price}

            return value_usd, prices_used

        except Exception as e:
            log.warning(f"Error calculating USD value for {value_token} {token_symbol} at {timestamp_str}: {e}")
            # Fallback: using current price
            token_price = self.get_current_token_price(token_symbol)
            return value_token * token_price, {token_symbol: token_price}

    def get_address_type_from_graph(self, graph: TransactionsGraph, address: str) -> str:
        """Defining address type from the graph"""

        # Сначала проверяем в узлах графа
        normalized_address = address.lower()

        if normalized_address in graph.nodes:
            node = graph.nodes[normalized_address]
            if node.type == NodeType.WALLET:
                return "wallet"
            elif node.type == NodeType.CONTRACT:
                return "contract"
            elif node.type == NodeType.ROOT:
                return "contract"  # ROOT nodes are usually SC

        # if not found in nodes, checking transactions
        for edge in graph.edges.values():
            for transaction in edge.transactions.values():
                if transaction.address_from.address.lower() == normalized_address:
                    return "wallet" if transaction.address_from.type == AddressType.WALLET else "contract"
                elif transaction.address_to.address.lower() == normalized_address:
                    return "wallet" if transaction.address_to.type == AddressType.WALLET else "contract"

        # assume Wallet by default
        return "wallet"

    def load_existing_graph(self) -> TransactionsGraph:
        """Loading existing graph file"""

        if not self.graph_file_path.exists():
            raise FileNotFoundError(
                f"Graph file not found: {self.graph_file_path}\n"
                f"Please ensure the graph file exists. You can create it using the existing "
                f"graph building scripts in the project."
            )

        try:
            log.info(f"Loading graph from {self.graph_file_path}")
            graph = load_graph_from_json(str(self.graph_file_path))
            log.info(f"Successfully loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in graph file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load graph: {e}")

    def identify_rocket_pool_contracts(self, graph: TransactionsGraph) -> Set[str]:
        """Determine Rocket Pool SC addresses in the graph"""

        rocket_pool_contracts = set()

        # ROOT nodes are usually main Rocket Pool SC
        for address, node in graph.nodes.items():
            if node.type == NodeType.ROOT:
                rocket_pool_contracts.add(address.lower())
                log.info(f"Found Rocket Pool ROOT contract: {address}")

        # Also searching for the known Rocket Pool SC 
        # (this list can be expanded with known Rocket Pool addresses)
        known_rp_contracts = {
            # Add known addresses here:
            # "0x...",  # RocketStorage
            # "0x...",  # RocketDepositPool
            # etc.
        }

        for contract_addr in known_rp_contracts:
            if contract_addr.lower() in graph.nodes:
                rocket_pool_contracts.add(contract_addr.lower())
                log.info(f"Found known Rocket Pool contract: {contract_addr}")

        log.info(f"Identified {len(rocket_pool_contracts)} Rocket Pool contracts")
        return rocket_pool_contracts

    def extract_rocket_pool_interacting_addresses(self, graph: TransactionsGraph) -> Set[str]:
        """Extraction ONLY addresses of the direct interaction with Rocket Pool contracts"""

        # Determine Rocket Pool contracts first
        rocket_pool_contracts = self.identify_rocket_pool_contracts(graph)

        if not rocket_pool_contracts:
            log.warning("No Rocket Pool contracts identified in the graph!")
            log.warning("Will analyze all addresses as fallback...")
            return self.extract_all_addresses_fallback(graph)

        rocket_pool_interacting_addresses = set()
        total_transactions_analyzed = 0
        rocket_pool_transactions = 0

        # Analysing all transactions and marking which interacted with the Rocket Pool
        for edge in graph.edges.values():
            for transaction in edge.transactions.values():
                total_transactions_analyzed += 1
                from_addr = transaction.address_from.address.lower()
                to_addr = transaction.address_to.address.lower()

                # Check if at least one Rocket Pool contract participates in transaction
                is_rp_transaction = (
                        from_addr in rocket_pool_contracts or
                        to_addr in rocket_pool_contracts
                )

                if is_rp_transaction:
                    rocket_pool_transactions += 1

                    # Adding both addresses and excluding Rocket Pool contract
                    if from_addr not in rocket_pool_contracts:
                        rocket_pool_interacting_addresses.add(from_addr)
                    if to_addr not in rocket_pool_contracts:
                        rocket_pool_interacting_addresses.add(to_addr)

        log.info(f"Analyzed {total_transactions_analyzed} total transactions")
        log.info(f"Found {rocket_pool_transactions} transactions involving Rocket Pool contracts")
        log.info(f"Extracted {len(rocket_pool_interacting_addresses)} unique addresses that interact with Rocket Pool")

        # logging addresses statstics
        wallet_count = 0
        contract_count = 0
        for addr in rocket_pool_interacting_addresses:
            if addr in graph.nodes:
                if graph.nodes[addr].type == NodeType.WALLET:
                    wallet_count += 1
                elif graph.nodes[addr].type == NodeType.CONTRACT:
                    contract_count += 1

        log.info(f"Address types: {wallet_count} wallets, {contract_count} contracts, {len(rocket_pool_interacting_addresses) - wallet_count - contract_count} unknown")

        return rocket_pool_interacting_addresses

    def extract_all_addresses_fallback(self, graph: TransactionsGraph) -> Set[str]:
        """Fallback method: extract all graph's addresses"""

        all_addresses = set()

        # Extracting all addresses from nodes (excluding ROOT nodes which are contracts)
        for address, node in graph.nodes.items():
            if node.type != NodeType.ROOT:  # excluding ROOT nodes
                all_addresses.add(address.lower())

        # Extraction of addresses from transactions
        for edge in graph.edges.values():
            for transaction in edge.transactions.values():
                from_addr = transaction.address_from.address.lower()
                to_addr = transaction.address_to.address.lower()

                # Make sure it's no ROOT nodes
                if from_addr in graph.nodes and graph.nodes[from_addr].type != NodeType.ROOT:
                    all_addresses.add(from_addr)
                if to_addr in graph.nodes and graph.nodes[to_addr].type != NodeType.ROOT:
                    all_addresses.add(to_addr)

        log.info(f"Extracted {len(all_addresses)} unique addresses (excluding ROOT contracts)")
        return all_addresses

    def get_wallet_statistics_etherscan_365d(self, address: str, graph: TransactionsGraph) -> WalletStatistics:
        """
        Obtain enhanced statistics via Etherscan API
        """

        # Determine address type
        address_type = self.get_address_type_from_graph(graph, address)

        # Etherscan API key
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not etherscan_api_key:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                error="ETHERSCAN_API_KEY not set in environment variables"
            )

        try:
            # Timespan - one year
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=365)
            start_block = 0  # Etherscan uses blocks but we use all blocks for simplicity

            # Transaction list for address
            url = "https://api.etherscan.io/api"
            params = {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": start_block,
                "endblock": 99999999,
                "page": 1,
                "offset": 100000,  # Transaction limit
                "sort": "asc",
                "apikey": etherscan_api_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data["status"] != "1":
                return WalletStatistics(
                    address=address,
                    address_type=address_type,
                    error=f"Etherscan API error: {data.get('message', 'Unknown error')}"
                )

            transactions = data["result"]

            if not transactions:
                return self._create_empty_wallet_stats(address, address_type)

            # Filtering transactions for the last 365 days
            start_timestamp = int(start_time.timestamp())
            filtered_transactions = [
                tx for tx in transactions
                if int(tx["timeStamp"]) >= start_timestamp
            ]

            if not filtered_transactions:
                return self._create_empty_wallet_stats(address, address_type)

            # Analysis of the transaction
            return self._analyze_etherscan_transactions_365d(address, address_type, filtered_transactions, transactions)

        except requests.exceptions.RequestException as e:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                error=f"Network error: {str(e)}"
            )
        except Exception as e:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                error=f"Unexpected error: {str(e)}"
            )

    def _analyze_etherscan_transactions_365d(self, address: str, address_type: str, transactions_365d: List[Dict], all_transactions: List[Dict]) -> WalletStatistics:
        """Etherscan transactions 365 days analysis"""

        if not transactions_365d:
            return self._create_empty_wallet_stats(address, address_type)

        volumes_usd_365d = []
        outgoing_tx_365d = []
        incoming_tx_365d = []
        all_prices_used = {}

        daily_volumes = {}
        monthly_volumes = {}
        gas_used_total = 0
        gas_fees_usd_total = 0.0
        unique_addresses = set()
        wallet_interactions = 0
        contract_interactions = 0

        for tx in transactions_365d:
            # wei -> ETH conversin
            value_wei = int(tx["value"])
            value_eth = value_wei / 10 ** 18

            timestamp = int(tx["timeStamp"])
            from_addr = tx["from"].lower()
            to_addr = tx["to"].lower()

            # Checking the direction of the transaction
            is_outgoing = from_addr == address.lower()
            is_incoming = to_addr == address.lower()

            if is_outgoing:
                outgoing_tx_365d.append(tx)
            if is_incoming:
                incoming_tx_365d.append(tx)

            # considering only outgoing transactions
            if is_outgoing and value_eth > 0:
                value_usd, prices_used = self.calculate_transaction_value_usd(
                    value_eth, 'ETH', str(timestamp)
                )
                volumes_usd_365d.append(value_usd)
                all_prices_used.update(prices_used)

                # daily and monthly activity
                tx_date = datetime.fromtimestamp(timestamp, timezone.utc).date()
                month_key = tx_date.strftime('%Y-%m')

                if tx_date not in daily_volumes:
                    daily_volumes[tx_date] = 0
                daily_volumes[tx_date] += value_usd

                if month_key not in monthly_volumes:
                    monthly_volumes[month_key] = 0
                monthly_volumes[month_key] += value_usd

            # Gas analysis (only onutgoing transactions)
            if is_outgoing:
                gas_used = int(tx.get("gasUsed", 0))
                gas_price = int(tx.get("gasPrice", 0))
                gas_used_total += gas_used

                # gas fee -> USD conversion
                gas_fee_eth = (gas_used * gas_price) / 10 ** 18
                gas_fee_usd, _ = self.calculate_transaction_value_usd(
                    gas_fee_eth, 'ETH', str(timestamp)
                )
                gas_fees_usd_total += gas_fee_usd

            # analysis of interaction
            if is_outgoing:
                other_address = to_addr
            else:
                other_address = from_addr

            unique_addresses.add(other_address)

            if len(other_address) == 42:  # Ethereum address
                if tx.get("input", "0x") == "0x":
                    wallet_interactions += 1
                else:
                    contract_interactions += 1

        # wallet information
        all_timestamps = [int(tx["timeStamp"]) for tx in all_transactions]
        first_timestamp = min(all_timestamps) if all_timestamps else None
        last_timestamp_365d = max([int(tx["timeStamp"]) for tx in transactions_365d])

        first_date = datetime.fromtimestamp(first_timestamp, timezone.utc) if first_timestamp else None
        wallet_age_days = (datetime.now(timezone.utc) - first_date).days if first_date else None

        # statistics for 365 days
        total_volume_usd_365d = sum(volumes_usd_365d)
        average_volume_usd_365d = total_volume_usd_365d / len(volumes_usd_365d) if volumes_usd_365d else 0.0
        max_volume_usd_365d = max(volumes_usd_365d) if volumes_usd_365d else 0.0
        median_volume_usd_365d = sorted(volumes_usd_365d)[len(volumes_usd_365d) // 2] if volumes_usd_365d else 0.0

        # activity
        active_days = len(daily_volumes)
        avg_daily_volume = sum(daily_volumes.values()) / len(daily_volumes) if daily_volumes else 0.0
        max_daily_volume = max(daily_volumes.values()) if daily_volumes else 0.0
        most_active_month = max(monthly_volumes.items(), key=lambda x: x[1])[0] if monthly_volumes else None

        # Gas stats
        avg_gas_price_gwei = 0.0
        if outgoing_tx_365d:
            total_gas_price_wei = sum(int(tx.get("gasPrice", 0)) for tx in outgoing_tx_365d)
            avg_gas_price_gwei = (total_gas_price_wei / len(outgoing_tx_365d)) / 10 ** 9

        return WalletStatistics(
            address=address,
            address_type=address_type,
            creation_date=first_date.isoformat() if first_date else None,
            first_transaction_date=first_date.isoformat() if first_date else None,
            last_transaction_date=datetime.fromtimestamp(last_timestamp_365d, timezone.utc).isoformat(),
            wallet_age_days=wallet_age_days,
            total_volume_usd_365d=round(total_volume_usd_365d, 2),
            average_volume_usd_365d=round(average_volume_usd_365d, 2),
            max_volume_usd_365d=round(max_volume_usd_365d, 2),
            median_volume_usd_365d=round(median_volume_usd_365d, 2),
            total_transactions_365d=len(transactions_365d),
            outgoing_transactions_365d=len(outgoing_tx_365d),
            incoming_transactions_365d=len(incoming_tx_365d),
            unique_addresses_interacted_365d=len(unique_addresses),
            wallet_interactions_365d=wallet_interactions,
            contract_interactions_365d=contract_interactions,
            active_days_365d=active_days,
            avg_daily_volume_usd_365d=round(avg_daily_volume, 2),
            max_daily_volume_usd_365d=round(max_daily_volume, 2),
            most_active_month_365d=most_active_month,
            total_gas_used_365d=gas_used_total,
            total_gas_fees_usd_365d=round(gas_fees_usd_total, 2),
            avg_gas_price_gwei_365d=round(avg_gas_price_gwei, 2),
            token_prices_used=all_prices_used
        )

    def get_wallet_statistics_rocket_pool_subgraph_365d(self, address: str, graph: TransactionsGraph) -> WalletStatistics:
        """
        Obtain statistics via The Graph Protocol subgraph for Rocket Pool over 365 days
        """

        # Determine address type
        address_type = self.get_address_type_from_graph(graph, address)

        # Time range - past 365 days
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=365)
        start_timestamp = int(start_time.timestamp())

        # Rocket Pool subgraph endpoint
        subgraph_url = "https://api.thegraph.com/subgraphs/name/rocket-pool/rocketpool"

        # Enhanced request with time filter
        query = """
        query GetWalletStats($address: String!, $startTimestamp: Int!) {
            user(id: $address) {
                id
                deposits(where: {timestamp_gte: $startTimestamp}, orderBy: timestamp, orderDirection: asc) {
                    id
                    amount
                    block
                    timestamp
                }
                withdrawals(where: {timestamp_gte: $startTimestamp}, orderBy: timestamp, orderDirection: asc) {
                    id
                    amount
                    block
                    timestamp
                }
            }
        }
        """

        try:
            response = requests.post(
                subgraph_url,
                json={"query": query, "variables": {"address": address.lower(), "startTimestamp": start_timestamp}},
                timeout=30
            )
            response.raise_for_status()

            data = response.json()

            if "errors" in data:
                return WalletStatistics(
                    address=address,
                    address_type=address_type,
                    error=f"Subgraph error: {data['errors']}"
                )

            user_data = data.get("data", {}).get("user")

            if not user_data:
                return self._create_empty_wallet_stats(address, address_type)

            # analysis of deposits and withdraws over 365 days
            all_transactions = []
            all_transactions.extend(user_data.get("deposits", []))
            all_transactions.extend(user_data.get("withdrawals", []))

            if not all_transactions:
                return self._create_empty_wallet_stats(address, address_type)

            # sorting
            all_transactions.sort(key=lambda x: int(x["timestamp"]))

            # We know less details for Rocket Pool, so using simplified analysis
            return self._analyze_rocket_pool_transactions_365d(address, address_type, all_transactions)

        except Exception as e:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                error=f"Subgraph error: {str(e)}"
            )

    def _analyze_rocket_pool_transactions_365d(self, address: str, address_type: str, transactions: List[Dict]) -> WalletStatistics:
        """365 days Rocket Pool transactions analysis"""

        if not transactions:
            return self._create_empty_wallet_stats(address, address_type)

        volumes_usd = []
        timestamps = [int(tx["timestamp"]) for tx in transactions]
        all_prices_used = {}

        daily_volumes = {}
        monthly_volumes = {}

        for tx in transactions:
            value_eth = float(tx["amount"]) / 10 ** 18  # wei to ETH
            value_usd, prices_used = self.calculate_transaction_value_usd(
                value_eth, 'ETH', tx["timestamp"]
            )
            volumes_usd.append(value_usd)
            all_prices_used.update(prices_used)

            # daily and monthly activity
            tx_date = datetime.fromtimestamp(int(tx["timestamp"]), timezone.utc).date()
            month_key = tx_date.strftime('%Y-%m')

            if tx_date not in daily_volumes:
                daily_volumes[tx_date] = 0
            daily_volumes[tx_date] += value_usd

            if month_key not in monthly_volumes:
                monthly_volumes[month_key] = 0
            monthly_volumes[month_key] += value_usd

        first_timestamp = min(timestamps)
        last_timestamp = max(timestamps)

        # calculation statistics
        total_volume_usd = sum(volumes_usd)
        average_volume_usd = total_volume_usd / len(volumes_usd)
        max_volume_usd = max(volumes_usd)
        median_volume_usd = sorted(volumes_usd)[len(volumes_usd) // 2]

        # Wallet age (simplified - from the first RP transaction)
        first_date = datetime.fromtimestamp(first_timestamp, timezone.utc)
        wallet_age_days = (datetime.now(timezone.utc) - first_date).days

        # Activity
        active_days = len(daily_volumes)
        avg_daily_volume = sum(daily_volumes.values()) / len(daily_volumes) if daily_volumes else 0.0
        max_daily_volume = max(daily_volumes.values()) if daily_volumes else 0.0
        most_active_month = max(monthly_volumes.items(), key=lambda x: x[1])[0] if monthly_volumes else None

        return WalletStatistics(
            address=address,
            address_type=address_type,
            creation_date=datetime.fromtimestamp(first_timestamp, timezone.utc).isoformat(),
            first_transaction_date=datetime.fromtimestamp(first_timestamp, timezone.utc).isoformat(),
            last_transaction_date=datetime.fromtimestamp(last_timestamp, timezone.utc).isoformat(),
            wallet_age_days=wallet_age_days,
            total_volume_usd_365d=round(total_volume_usd, 2),
            average_volume_usd_365d=round(average_volume_usd, 2),
            max_volume_usd_365d=round(max_volume_usd, 2),
            median_volume_usd_365d=round(median_volume_usd, 2),
            total_transactions_365d=len(transactions),
            outgoing_transactions_365d=len([tx for tx in transactions if "deposits" in str(tx)]),  # Simplification
            incoming_transactions_365d=len([tx for tx in transactions if "withdrawals" in str(tx)]),  # Simplification
            unique_addresses_interacted_365d=1,  # Only with Rocket Pool
            wallet_interactions_365d=0,  # Rocket Pool - is a contract
            contract_interactions_365d=len(transactions),  # All RP contract interactions
            active_days_365d=active_days,
            avg_daily_volume_usd_365d=round(avg_daily_volume, 2),
            max_daily_volume_usd_365d=round(max_daily_volume, 2),
            most_active_month_365d=most_active_month,
            total_gas_used_365d=0,  # data not available via subgraph
            total_gas_fees_usd_365d=0.0,
            avg_gas_price_gwei_365d=0.0,
            token_prices_used=all_prices_used
        )

    def _create_empty_wallet_stats(self, address: str, address_type: str) -> WalletStatistics:
        """Create empty statistics for an address without activity"""
        return WalletStatistics(
            address=address,
            address_type=address_type,
            creation_date=None,
            first_transaction_date=None,
            last_transaction_date=None,
            wallet_age_days=None,
            total_volume_usd_365d=0.0,
            average_volume_usd_365d=0.0,
            max_volume_usd_365d=0.0,
            median_volume_usd_365d=0.0,
            total_transactions_365d=0,
            outgoing_transactions_365d=0,
            incoming_transactions_365d=0,
            unique_addresses_interacted_365d=0,
            wallet_interactions_365d=0,
            contract_interactions_365d=0,
            active_days_365d=0,
            avg_daily_volume_usd_365d=0.0,
            max_daily_volume_usd_365d=0.0,
            most_active_month_365d=None,
            total_gas_used_365d=0,
            total_gas_fees_usd_365d=0.0,
            avg_gas_price_gwei_365d=0.0,
            token_prices_used={}
        )

    def get_wallet_statistics_batch(self, addresses: List[str], graph: TransactionsGraph) -> List[WalletStatistics]:
        """Getting enhanced statistics over 365 days for a batch of addresses using multithreading"""

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_address = {}

            for address in addresses:
                # First try Rocket Pool subgraph (365d), thenатем (365d)
                future1 = executor.submit(self.get_wallet_statistics_rocket_pool_subgraph_365d, address, graph)
                future_to_address[future1] = (address, "subgraph_365d")

                # Also run Etherscan in parallel (365d)
                future2 = executor.submit(self.get_wallet_statistics_etherscan_365d, address, graph)
                future_to_address[future2] = (address, "etherscan_365d")

            # Gathering results
            address_results = {}

            with tqdm(total=len(addresses) * 2, desc="Fetching 365-day address statistics") as pbar:
                for future in as_completed(future_to_address):
                    address, source = future_to_address[future]
                    pbar.update(1)

                    try:
                        result = future.result()

                        # Using Etherscan as a priority source (more complete data)
                        if address not in address_results:
                            address_results[address] = result
                        elif source == "etherscan_365d" and not result.error:
                            # Etherscan data has higher priority rather than subgraph
                            address_results[address] = result
                        elif address_results[address].error and not result.error:
                            address_results[address] = result

                    except Exception as e:
                        log.warning(f"Failed to get 365d statistics for {address} from {source}: {e}")
                        if address not in address_results:
                            address_results[address] = WalletStatistics(
                                address=address,
                                address_type="unknown",
                                error=f"Failed to fetch from {source}: {str(e)}"
                            )

            # Addind a small delay to preserve rate limits
            time.sleep(0.1)

        return list(address_results.values())

    def analyze_addresses(self, addresses: Set[str], graph: TransactionsGraph) -> List[WalletStatistics]:
        """Analysing list of addresses and obtaining their statistics"""

        log.info(f"Starting analysis of {len(addresses)} addresses...")

        # conversion to list for processing
        addresses_list = list(addresses)

        # Processing addresses in chunks
        batch_size = 100  # chunk size
        all_results = []

        for i in range(0, len(addresses_list), batch_size):
            batch = addresses_list[i:i + batch_size]
            log.info(f"Processing batch {i // batch_size + 1}/{(len(addresses_list) + batch_size - 1) // batch_size}")

            batch_results = self.get_wallet_statistics_batch(batch, graph)
            all_results.extend(batch_results)

            # adding a small delay
            if i + batch_size < len(addresses_list):
                time.sleep(1)

        log.info(f"Completed analysis of {len(all_results)} addresses")

        # Error statistics
        error_count = sum(1 for result in all_results if result.error)
        success_count = len(all_results) - error_count

        # Counting addresses types
        wallet_count = sum(1 for result in all_results if result.address_type == "wallet" and not result.error)
        contract_count = sum(1 for result in all_results if result.address_type == "contract" and not result.error)

        log.info(f"Success rate: {success_count}/{len(all_results)} ({success_count / len(all_results) * 100:.1f}%)")
        log.info(f"Address types - Wallets: {wallet_count}, Contracts: {contract_count}")

        return all_results

    def save_results(self, results: List[WalletStatistics], prefix: str = "rocket_pool_addresses"):
        """Saving the results in JSON and CSV"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSON
        json_file = self.output_dir / f"{prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json_data = [asdict(result) for result in results]
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        log.info(f"Saved JSON results to {json_file}")

        # CSV
        csv_file = self.output_dir / f"{prefix}_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
                writer.writeheader()
                for result in results:
                    writer.writerow(asdict(result))

        log.info(f"Saved CSV results to {csv_file}")

        # Generate a report
        self.create_summary_report(results, prefix, timestamp)

        return json_file, csv_file

    def create_summary_report(self, results: List[WalletStatistics], prefix: str, timestamp: str):
        """Generating report with enhances statistics over 365 days"""

        # Filtering successful results
        successful_results = [r for r in results if not r.error and r.total_transactions_365d and r.total_transactions_365d > 0]

        if not successful_results:
            log.warning("No successful results to create summary report")
            return

        # calculating aggregated metrics
        total_addresses = len(results)
        successful_addresses = len(successful_results)

        # address type separation
        wallets = [r for r in successful_results if r.address_type == "wallet"]
        contracts = [r for r in successful_results if r.address_type == "contract"]

        # 365 days metrics
        total_volumes_365d = [r.total_volume_usd_365d for r in successful_results if r.total_volume_usd_365d]
        avg_volumes_365d = [r.average_volume_usd_365d for r in successful_results if r.average_volume_usd_365d]
        max_volumes_365d = [r.max_volume_usd_365d for r in successful_results if r.max_volume_usd_365d]
        tx_counts_365d = [r.total_transactions_365d for r in successful_results if r.total_transactions_365d]

        # wallet age
        wallet_ages = [r.wallet_age_days for r in successful_results if r.wallet_age_days]

        # interactions
        unique_interactions = [r.unique_addresses_interacted_365d for r in successful_results if r.unique_addresses_interacted_365d]
        wallet_interactions = [r.wallet_interactions_365d for r in successful_results if r.wallet_interactions_365d]
        contract_interactions = [r.contract_interactions_365d for r in successful_results if r.contract_interactions_365d]

        # activity
        active_days = [r.active_days_365d for r in successful_results if r.active_days_365d]
        gas_fees = [r.total_gas_fees_usd_365d for r in successful_results if r.total_gas_fees_usd_365d]

        # getting token prices
        token_prices_sample = successful_results[0].token_prices_used if successful_results[0].token_prices_used else self.current_token_prices

        summary = {
            "analysis_timestamp": timestamp,
            "analysis_type": "all_addresses_365d" if self.analyze_all_addresses else "rocket_pool_direct_interactions_365d",
            "analysis_period": "365 days",
            "graph_source": str(self.graph_file_path),
            "token_prices_used": token_prices_sample,
            "total_addresses_analyzed": total_addresses,
            "analysis_scope": "All addresses in graph (365d analysis)" if self.analyze_all_addresses else "Only addresses that directly interact with Rocket Pool contracts (365d analysis)",
            "successful_analyses": successful_addresses,
            "success_rate_percent": round(successful_addresses / total_addresses * 100, 2),
            "address_types": {
                "wallets": len(wallets),
                "contracts": len(contracts),
                "unknown": successful_addresses - len(wallets) - len(contracts)
            },
            "volume_metrics_365d": {
                "total_volume_usd": {
                    "min": round(min(total_volumes_365d), 2) if total_volumes_365d else 0,
                    "max": round(max(total_volumes_365d), 2) if total_volumes_365d else 0,
                    "sum": round(sum(total_volumes_365d), 2) if total_volumes_365d else 0,
                    "mean": round(sum(total_volumes_365d) / len(total_volumes_365d), 2) if total_volumes_365d else 0
                },
                "average_volume_usd": {
                    "min": round(min(avg_volumes_365d), 2) if avg_volumes_365d else 0,
                    "max": round(max(avg_volumes_365d), 2) if avg_volumes_365d else 0,
                    "mean": round(sum(avg_volumes_365d) / len(avg_volumes_365d), 2) if avg_volumes_365d else 0
                },
                "max_volume_usd": {
                    "min": round(min(max_volumes_365d), 2) if max_volumes_365d else 0,
                    "max": round(max(max_volumes_365d), 2) if max_volumes_365d else 0,
                    "mean": round(sum(max_volumes_365d) / len(max_volumes_365d), 2) if max_volumes_365d else 0
                }
            },
            "transaction_metrics_365d": {
                "total_transactions": {
                    "min": min(tx_counts_365d) if tx_counts_365d else 0,
                    "max": max(tx_counts_365d) if tx_counts_365d else 0,
                    "sum": sum(tx_counts_365d) if tx_counts_365d else 0,
                    "mean": round(sum(tx_counts_365d) / len(tx_counts_365d), 2) if tx_counts_365d else 0
                }
            },
            "wallet_age_metrics": {
                "min_age_days": min(wallet_ages) if wallet_ages else 0,
                "max_age_days": max(wallet_ages) if wallet_ages else 0,
                "mean_age_days": round(sum(wallet_ages) / len(wallet_ages), 2) if wallet_ages else 0,
                "median_age_days": sorted(wallet_ages)[len(wallet_ages) // 2] if wallet_ages else 0
            },
            "interaction_metrics_365d": {
                "unique_addresses_interacted": {
                    "min": min(unique_interactions) if unique_interactions else 0,
                    "max": max(unique_interactions) if unique_interactions else 0,
                    "mean": round(sum(unique_interactions) / len(unique_interactions), 2) if unique_interactions else 0
                },
                "wallet_interactions": sum(wallet_interactions) if wallet_interactions else 0,
                "contract_interactions": sum(contract_interactions) if contract_interactions else 0
            },
            "activity_metrics_365d": {
                "active_days": {
                    "min": min(active_days) if active_days else 0,
                    "max": max(active_days) if active_days else 0,
                    "mean": round(sum(active_days) / len(active_days), 2) if active_days else 0
                },
                "total_gas_fees_usd": round(sum(gas_fees), 2) if gas_fees else 0,
                "mean_gas_fees_usd": round(sum(gas_fees) / len(gas_fees), 2) if gas_fees else 0
            }
        }

        # saving the summary
        summary_file = self.output_dir / f"{prefix}_{timestamp}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        log.info(f"Saved summary report to {summary_file}")

        # log output
        analysis_type_msg = "ALL addresses" if self.analyze_all_addresses else "addresses that interact directly with Rocket Pool contracts"
        log.info(f"=== ROCKET POOL 365-DAY ANALYSIS SUMMARY ===")
        log.info(f"Analyzed {total_addresses} {analysis_type_msg} over 365 days")
        log.info(f"Graph source: {self.graph_file_path.name}")
        log.info(f"Success rate: {successful_addresses}/{total_addresses} ({summary['success_rate_percent']}%)")
        log.info(f"Address types: {len(wallets)} wallets, {len(contracts)} contracts")
        if total_volumes_365d:
            log.info(f"Total volume (365d): ${summary['volume_metrics_365d']['total_volume_usd']['sum']:,.2f}")
            log.info(f"Average wallet age: {summary['wallet_age_metrics']['mean_age_days']:.0f} days")
            log.info(f"Total transactions (365d): {summary['transaction_metrics_365d']['total_transactions']['sum']:,}")
            log.info(f"Total gas fees (365d): ${summary['activity_metrics_365d']['total_gas_fees_usd']:,.2f}")

    def run_analysis(self) -> None:
        """Running complete analysis"""

        log.info("=" * 60)
        log.info("ROCKET POOL ADDRESS ANALYSIS STARTED")
        log.info("=" * 60)

        try:
            # 1. loading existing graph
            graph = self.load_existing_graph()

            # 2. extraction of the addresses depending on the set-up
            if self.analyze_all_addresses:
                log.info("Analyzing ALL addresses in the graph...")
                addresses_to_analyze = self.extract_all_addresses_fallback(graph)
            else:
                log.info("Analyzing only addresses that directly interact with Rocket Pool...")
                addresses_to_analyze = self.extract_rocket_pool_interacting_addresses(graph)

            if not addresses_to_analyze:
                log.error("No addresses found for analysis")
                return

            # 3. data analysis
            results = self.analyze_addresses(addresses_to_analyze, graph)

            # 4. saving results
            json_file, csv_file = self.save_results(results)

            log.info("=" * 60)
            log.info("ANALYSIS COMPLETED SUCCESSFULLY")
            log.info(f"Results saved to:")
            log.info(f"  JSON: {json_file}")
            log.info(f"  CSV: {csv_file}")
            log.info("=" * 60)

        except Exception as e:
            log.error(f"Analysis failed: {e}")
            raise


def main():
    """Main function"""

    import argparse

    # Getting project root directory for the main function
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent  # mintegrity/scripts/stats_vis/ -> mintegrity/

    parser = argparse.ArgumentParser(
        description="Rocket Pool Address Analyzer - analyzes 365-day activity of addresses that interact with Rocket Pool contracts (by default) or all addresses in graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rocket_pool_analyzer.py
  python rocket_pool_analyzer.py --graph-path /path/to/custom_graph.json
  python rocket_pool_analyzer.py --output-dir /path/to/my_results --max-workers 10
  python rocket_pool_analyzer.py --all-addresses  # Analyze ALL addresses, not just Rocket Pool interactions

Note: Graph contains 90-day interactions, but statistics are collected for 365 days per address.
Default paths are calculated relative to the project root directory.
        """
    )

    parser.add_argument(
        "--graph-path",
        default=None,
        help=f"Path to existing graph file (default: {project_root}/files/rocket_pool_full_graph_90_days.json)"
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Output directory for results (default: {project_root}/files/rocket_pool_analysis)"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum number of concurrent API requests (default: 5)"
    )

    parser.add_argument(
        "--all-addresses",
        action="store_true",
        help="Analyze ALL addresses in graph instead of only Rocket Pool interacting addresses"
    )

    args = parser.parse_args()

    try:
        analyzer = RocketPoolAnalyzer(
            graph_file_path=args.graph_path,
            output_dir=args.output_dir,
            max_workers=args.max_workers,
            analyze_all_addresses=args.all_addresses
        )

        analyzer.run_analysis()

    except KeyboardInterrupt:
        log.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
