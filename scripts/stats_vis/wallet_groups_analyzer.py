#!/usr/bin/env python3
"""
General Wallet Groups Analyzer
Provides generic functionality for analyzing groups of coordinated addresses
with complete statistics via APIs.

This module contains:
1. Data structures for wallet and group statistics
2. Built-in address analyzer with API functionality
3. General wallet groups analyzer with visualization capabilities
4. No blockchain-specific logic
"""

import os
import json
import csv
import pandas as pd
import numpy as np
import time
import requests
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure matplotlib for headless servers
import matplotlib
from dotenv import load_dotenv

matplotlib.use('Agg')  # Use backend without GUI
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

try:
    from tqdm import tqdm
except ImportError:
    # Create tqdm stub if not installed
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None):
            self.iterable = iterable
            self.total = total
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            self.n += n

        def __iter__(self):
            return iter(self.iterable)

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings('ignore')
plt.style.use('default')

# Try importing additional modules for full analysis
try:
    from scripts.commons import metadata
    from scripts.commons.tokens_metadata_scraper import fetch_current_token_prices
    from scripts.commons.known_token_list import ETH_TOKENS_WHITELIST

    FULL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Some API analysis modules not available: {e}")
    print("Will use basic graph-based analysis")
    FULL_ANALYSIS_AVAILABLE = False


@dataclass
class WalletStatistics:
    """Address statistics for 365 days"""
    address: str
    address_type: str = None
    total_volume_usd_365d: Optional[float] = None
    total_transactions_365d: Optional[int] = None
    outgoing_transactions_365d: Optional[int] = None
    incoming_transactions_365d: Optional[int] = None
    wallet_age_days: Optional[int] = None
    active_days_365d: Optional[int] = None
    most_active_month_365d: Optional[str] = None
    total_gas_fees_usd_365d: Optional[float] = None
    unique_addresses_interacted_365d: Optional[int] = None
    average_volume_usd_365d: Optional[float] = None
    max_volume_usd_365d: Optional[float] = None
    median_volume_usd_365d: Optional[float] = None
    wallet_interactions_365d: Optional[int] = None
    contract_interactions_365d: Optional[int] = None
    avg_daily_volume_usd_365d: Optional[float] = None
    max_daily_volume_usd_365d: Optional[float] = None
    total_gas_used_365d: Optional[int] = None
    avg_gas_price_gwei_365d: Optional[float] = None
    creation_date: Optional[str] = None
    first_transaction_date: Optional[str] = None
    last_transaction_date: Optional[str] = None
    token_prices_used: Optional[Dict[str, float]] = None
    error: Optional[str] = None


@dataclass
class GroupStatistics:
    """Group address statistics for 365 days"""
    group_id: int
    group_size: int
    addresses: List[str]

    # Aggregated statistics for 365 days
    total_volume_usd_365d: float = 0.0
    total_transactions_365d: int = 0
    total_outgoing_transactions_365d: int = 0
    total_incoming_transactions_365d: int = 0

    # Average values
    avg_volume_per_address_365d: float = 0.0
    avg_transactions_per_address_365d: float = 0.0

    # Maximum values in group
    max_volume_in_group_365d: float = 0.0
    max_transactions_in_group_365d: int = 0

    # Age and activity
    oldest_wallet_age_days: Optional[int] = None
    newest_wallet_age_days: Optional[int] = None
    avg_wallet_age_days: Optional[float] = None

    # Group patterns
    total_active_days_365d: int = 0
    unique_months_active: int = 0
    coordination_score_avg: float = 0.0

    # Role distribution in group
    layer_wallets_count: int = 0
    storage_wallets_count: int = 0
    regular_wallets_count: int = 0
    contracts_count: int = 0

    # Gas and fees
    total_gas_fees_usd_365d: float = 0.0
    avg_gas_fees_per_address_365d: float = 0.0

    # Interactions
    internal_transfers_count: int = 0  # Transfers within group
    external_unique_addresses: int = 0  # Unique external addresses

    # Additional information
    distance_to_root: Optional[int] = None
    error_addresses: List[str] = None


class BuiltInAddressAnalyzer:
    """Built-in address analyzer with full API functionality"""

    def __init__(self, max_workers: int = 5):
        global FULL_ANALYSIS_AVAILABLE
        self.max_workers = max_workers
        self.price_cache = {}  # Cache for historical prices
        self.current_token_prices = {}

        if FULL_ANALYSIS_AVAILABLE:
            try:
                # Initialize metadata
                metadata.init()
                self.current_token_prices = self._fetch_current_prices()
                print(f"Loaded fallback prices for {len(self.current_token_prices)} tokens")
            except Exception as e:
                print(f"Failed to initialize pricing: {e}")
                FULL_ANALYSIS_AVAILABLE = False

    def _fetch_current_prices(self) -> Dict[str, float]:
        """Gets current token prices"""
        try:
            # Use existing module to get prices
            token_prices_with_timestamps = fetch_current_token_prices(ETH_TOKENS_WHITELIST)

            current_prices = {}
            for token, (timestamp, price) in token_prices_with_timestamps.items():
                current_prices[token] = price

            return current_prices

        except Exception as e:
            print(f"Failed to fetch current prices via API: {e}")

            # Fallback: use prices from metadata
            fallback_prices = {}
            for token in ETH_TOKENS_WHITELIST:
                try:
                    price = metadata.get_token_price_usd(token, str(int(time.time())))
                    if price > 0:
                        fallback_prices[token] = price
                except:
                    pass

            return fallback_prices

    def get_historical_token_price(self, token_symbol: str, timestamp: int) -> float:
        """Gets historical token price"""
        cache_key = f"{token_symbol.upper()}-{timestamp}"

        if cache_key in self.price_cache:
            return self.price_cache[cache_key]

        # First try metadata
        try:
            price = metadata.get_token_price_usd(token_symbol, str(timestamp))
            if price > 0:
                self.price_cache[cache_key] = price
                return price
        except Exception:
            pass

        # External API (Coinbase)
        try:
            token_to_pair = {
                'ETH': 'ETH-USD', 'BTC': 'BTC-USD', 'WETH': 'ETH-USD',
                'USDT': 'USDT-USD', 'USDC': 'USDC-USD', 'DAI': 'DAI-USD',
                'LINK': 'LINK-USD', 'UNI': 'UNI-USD', 'AAVE': 'AAVE-USD'
            }

            pair = token_to_pair.get(token_symbol.upper(), 'ETH-USD')

            start_time = timestamp - 3600
            end_time = timestamp + 3600

            url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
            params = {
                'start': datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
                'end': datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
                'granularity': 3600
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            candles = response.json()
            if candles:
                closest_candle = min(candles, key=lambda x: abs(x[0] - timestamp))
                price = float(closest_candle[4])  # close price
                self.price_cache[cache_key] = price
                return price

        except Exception:
            pass

        # Fallback: current price
        return self.current_token_prices.get(token_symbol.upper(),
                                             self.current_token_prices.get('ETH', 2500.0))

    def get_wallet_statistics_etherscan(self, address: str, address_type: str) -> WalletStatistics:
        """Gets statistics via Etherscan API"""
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not etherscan_api_key:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                error="ETHERSCAN_API_KEY not set"
            )

        try:
            # Get transactions for 365 days
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=365)

            url = "https://api.etherscan.io/api"
            params = {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": 0,
                "endblock": 99999999,
                "page": 1,
                "offset": 10000,
                "sort": "asc",
                "apikey": etherscan_api_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data["status"] != "1":
                error_msg = data.get('message', 'Unknown error')
                return WalletStatistics(
                    address=address,
                    address_type=address_type,
                    error=f"Etherscan API error: {error_msg}"
                )

            transactions = data["result"]
            if not transactions:
                return self._create_empty_stats(address, address_type)

            # Filter for 365 days
            start_timestamp = int(start_time.timestamp())
            filtered_transactions = [
                tx for tx in transactions
                if int(tx["timeStamp"]) >= start_timestamp
            ]

            if not filtered_transactions:
                return self._create_empty_stats(address, address_type)

            return self._analyze_transactions(address, address_type, filtered_transactions, transactions)

        except Exception as e:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                error=f"API error: {str(e)}"
            )

    def _analyze_transactions(self, address: str, address_type: str,
                              transactions_365d: List[Dict], all_transactions: List[Dict]) -> WalletStatistics:
        """Analyzes transactions"""

        volumes_usd = []
        outgoing_txs = []
        incoming_txs = []
        daily_volumes = {}
        monthly_volumes = {}
        gas_used_total = 0
        gas_fees_usd_total = 0.0
        unique_addresses = set()

        for tx in transactions_365d:
            value_wei = int(tx["value"])
            value_eth = value_wei / 10 ** 18
            timestamp = int(tx["timeStamp"])
            from_addr = tx["from"].lower()
            to_addr = tx["to"].lower()

            is_outgoing = from_addr == address.lower()
            is_incoming = to_addr == address.lower()

            if is_outgoing:
                outgoing_txs.append(tx)
            if is_incoming:
                incoming_txs.append(tx)

            # Analyze outgoing transactions
            if is_outgoing and value_eth > 0:
                # Get ETH price at transaction time
                eth_price = self.get_historical_token_price('ETH', timestamp)
                value_usd = value_eth * eth_price
                volumes_usd.append(value_usd)

                # Daily and monthly activity
                tx_date = datetime.fromtimestamp(timestamp, timezone.utc).date()
                month_key = tx_date.strftime('%Y-%m')

                daily_volumes[tx_date] = daily_volumes.get(tx_date, 0) + value_usd
                monthly_volumes[month_key] = monthly_volumes.get(month_key, 0) + value_usd

            # Gas analysis
            if is_outgoing:
                gas_used = int(tx.get("gasUsed", 0))
                gas_price = int(tx.get("gasPrice", 0))
                gas_used_total += gas_used

                gas_fee_eth = (gas_used * gas_price) / 10 ** 18
                eth_price = self.get_historical_token_price('ETH', timestamp)
                gas_fees_usd_total += gas_fee_eth * eth_price

            # Unique addresses
            other_address = to_addr if is_outgoing else from_addr
            unique_addresses.add(other_address)

        # Wallet age
        all_timestamps = [int(tx["timeStamp"]) for tx in all_transactions]
        first_timestamp = min(all_timestamps) if all_timestamps else None

        first_date = None
        wallet_age_days = None
        if first_timestamp:
            first_date = datetime.fromtimestamp(first_timestamp, timezone.utc)
            wallet_age_days = (datetime.now(timezone.utc) - first_date).days

        # Statistics
        total_volume = sum(volumes_usd)
        avg_volume = total_volume / len(volumes_usd) if volumes_usd else 0
        max_volume = max(volumes_usd) if volumes_usd else 0
        median_volume = sorted(volumes_usd)[len(volumes_usd) // 2] if volumes_usd else 0

        active_days = len(daily_volumes)
        avg_daily_volume = sum(daily_volumes.values()) / len(daily_volumes) if daily_volumes else 0
        max_daily_volume = max(daily_volumes.values()) if daily_volumes else 0
        most_active_month = max(monthly_volumes.items(), key=lambda x: x[1])[0] if monthly_volumes else None

        # Gas statistics
        avg_gas_price_gwei = 0.0
        if outgoing_txs:
            total_gas_price_wei = sum(int(tx.get("gasPrice", 0)) for tx in outgoing_txs)
            avg_gas_price_gwei = (total_gas_price_wei / len(outgoing_txs)) / 10 ** 9

        return WalletStatistics(
            address=address,
            address_type=address_type,
            creation_date=first_date.isoformat() if first_date else None,
            first_transaction_date=first_date.isoformat() if first_date else None,
            last_transaction_date=datetime.fromtimestamp(max([int(tx["timeStamp"]) for tx in transactions_365d]), timezone.utc).isoformat(),
            wallet_age_days=wallet_age_days,
            total_volume_usd_365d=round(total_volume, 2),
            average_volume_usd_365d=round(avg_volume, 2),
            max_volume_usd_365d=round(max_volume, 2),
            median_volume_usd_365d=round(median_volume, 2),
            total_transactions_365d=len(transactions_365d),
            outgoing_transactions_365d=len(outgoing_txs),
            incoming_transactions_365d=len(incoming_txs),
            unique_addresses_interacted_365d=len(unique_addresses),
            active_days_365d=active_days,
            avg_daily_volume_usd_365d=round(avg_daily_volume, 2),
            max_daily_volume_usd_365d=round(max_daily_volume, 2),
            most_active_month_365d=most_active_month,
            total_gas_used_365d=gas_used_total,
            total_gas_fees_usd_365d=round(gas_fees_usd_total, 2),
            avg_gas_price_gwei_365d=round(avg_gas_price_gwei, 2),
            wallet_interactions_365d=0,  # Simplified
            contract_interactions_365d=0,  # Simplified
            token_prices_used={'ETH': self.current_token_prices.get('ETH', 0)}
        )

    def _create_empty_stats(self, address: str, address_type: str) -> WalletStatistics:
        """Creates empty statistics"""
        return WalletStatistics(
            address=address,
            address_type=address_type,
            total_volume_usd_365d=0.0,
            total_transactions_365d=0,
            outgoing_transactions_365d=0,
            incoming_transactions_365d=0
        )

    def analyze_addresses_batch(self, addresses: List[str], graph) -> List[WalletStatistics]:
        """Analyzes batch of addresses"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_address = {}

            for address in addresses:
                # Determine address type from graph
                address_type = "wallet"
                if address in graph.nodes:
                    node = graph.nodes[address]
                    if hasattr(node, 'type') and str(node.type).upper() == "CONTRACT":
                        address_type = "contract"

                future = executor.submit(self.get_wallet_statistics_etherscan, address, address_type)
                future_to_address[future] = address

            with tqdm(total=len(addresses), desc="Analyzing addresses") as pbar:
                for future in as_completed(future_to_address):
                    pbar.update(1)
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        address = future_to_address[future]
                        print(f"Failed to analyze {address}: {e}")
                        results.append(WalletStatistics(
                            address=address,
                            address_type="unknown",
                            error=str(e)
                        ))

        return results


class WalletGroupsAnalyzer:
    """General wallet groups analyzer with visualization capabilities"""

    def __init__(self, output_dir: str = "files/groups_analysis", max_workers: int = 5):
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)

        self.individual_stats = {}  # Individual address statistics
        self.group_stats = []  # Group statistics

    def analyze_addresses_with_full_stats(self, addresses: List[str], graph):
        """Analyzes addresses with full statistics via built-in analyzer"""
        print(f"Starting full 365-day analysis for {len(addresses)} addresses...")
        print("This will fetch detailed statistics via Etherscan API with historical prices")
        print("This may take several minutes depending on the number of addresses...")

        # Create built-in analyzer
        analyzer = BuiltInAddressAnalyzer(max_workers=self.max_workers)

        try:
            # Analyze addresses in batches
            batch_size = 50
            all_results = []

            for i in range(0, len(addresses), batch_size):
                batch = addresses[i:i + batch_size]
                print(f"Processing batch {i // batch_size + 1}/{(len(addresses) + batch_size - 1) // batch_size}")

                batch_results = analyzer.analyze_addresses_batch(batch, graph)
                all_results.extend(batch_results)

                # Pause between batches
                if i + batch_size < len(addresses):
                    time.sleep(1)

            # Convert results
            for result in all_results:
                if not result.error:
                    self.individual_stats[result.address] = result
                else:
                    print(f"Failed to analyze {result.address}: {result.error}")

            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_file = self.output_dir / "data" / f"detailed_addresses_analysis_{timestamp}.json"

            with open(detailed_file, 'w', encoding='utf-8') as f:
                json_data = [asdict(result) for result in all_results]
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            success_count = len([r for r in all_results if not r.error])
            print(f"Completed detailed analysis: {success_count}/{len(all_results)} addresses successfully analyzed")
            print(f"Detailed results saved to: {detailed_file}")

        except Exception as e:
            print(f"Failed to perform detailed analysis: {e}")
            print("Falling back to simplified graph-based statistics...")
            self.create_mock_statistics_from_graph(addresses, graph)

    def create_mock_statistics_from_graph(self, addresses: List[str], graph):
        """Creates simplified statistics from graph data"""
        for address in addresses:
            # Simple counting from graph
            total_transactions = 0
            total_volume = 0.0

            # Count outgoing transactions
            for (from_addr, to_addr), edge in graph.edges.items():
                if from_addr == address:
                    total_transactions += len(edge.transactions)
                    for tx in edge.transactions.values():
                        total_volume += tx.value_usd

            # Determine address type
            address_type = "wallet"
            if address in graph.nodes:
                node = graph.nodes[address]
                # Check node type
                if hasattr(node, 'type'):
                    if hasattr(node.type, 'name'):
                        address_type = node.type.name.lower()
                    elif str(node.type).upper() == "CONTRACT":
                        address_type = "contract"

            self.individual_stats[address] = WalletStatistics(
                address=address,
                address_type=address_type,
                total_volume_usd_365d=total_volume,
                total_transactions_365d=total_transactions,
                outgoing_transactions_365d=total_transactions,
                incoming_transactions_365d=0,
                wallet_age_days=None,
                active_days_365d=None,
                most_active_month_365d=None,
                total_gas_fees_usd_365d=0.0,
                unique_addresses_interacted_365d=None
            )

        print(f"Created mock statistics for {len(self.individual_stats)} addresses")

    def calculate_group_statistics(self, wallet_groups: List[List[str]], graph):
        """Calculates aggregated statistics for each group"""
        print("Calculating group statistics...")

        self.group_stats = []

        for group_id, group_addresses in enumerate(wallet_groups, 1):
            print(f"Processing group {group_id} with {len(group_addresses)} addresses")

            # Filter addresses that have statistics
            valid_addresses = []
            error_addresses = []

            for addr in group_addresses:
                if addr in self.individual_stats:
                    valid_addresses.append(addr)
                else:
                    error_addresses.append(addr)

            if not valid_addresses:
                print(f"No valid statistics for group {group_id}")
                continue

            # Aggregate statistics
            group_stat = self._aggregate_group_statistics(
                group_id, list(group_addresses), valid_addresses, error_addresses, graph
            )

            self.group_stats.append(group_stat)

        print(f"Calculated statistics for {len(self.group_stats)} groups")

    def _aggregate_group_statistics(self, group_id: int, all_addresses: List[str],
                                    valid_addresses: List[str], error_addresses: List[str], graph) -> GroupStatistics:
        """Aggregates statistics for one group"""

        # Get statistics for valid addresses
        stats_list = [self.individual_stats[addr] for addr in valid_addresses]

        # Basic information
        group_size = len(all_addresses)

        # Aggregated volumes and transactions
        total_volume = sum(s.total_volume_usd_365d or 0 for s in stats_list)
        total_transactions = sum(s.total_transactions_365d or 0 for s in stats_list)
        total_outgoing = sum(s.outgoing_transactions_365d or 0 for s in stats_list)
        total_incoming = sum(s.incoming_transactions_365d or 0 for s in stats_list)

        # Average values
        avg_volume_per_address = total_volume / len(valid_addresses) if valid_addresses else 0
        avg_transactions_per_address = total_transactions / len(valid_addresses) if valid_addresses else 0

        # Maximum values
        max_volume = max((s.total_volume_usd_365d or 0 for s in stats_list), default=0)
        max_transactions = max((s.total_transactions_365d or 0 for s in stats_list), default=0)

        # Wallet age
        ages = [s.wallet_age_days for s in stats_list if s.wallet_age_days]
        oldest_age = max(ages) if ages else None
        newest_age = min(ages) if ages else None
        avg_age = sum(ages) / len(ages) if ages else None

        # Activity
        total_active_days = sum(s.active_days_365d or 0 for s in stats_list)

        # Count unique active months
        unique_months = set()
        for s in stats_list:
            if s.most_active_month_365d:
                unique_months.add(s.most_active_month_365d)

        # Wallet type distribution
        contracts_count = sum(1 for s in stats_list if s.address_type == "contract")
        regular_count = len(stats_list) - contracts_count

        # Gas fees
        total_gas_fees = sum(s.total_gas_fees_usd_365d or 0 for s in stats_list)
        avg_gas_fees = total_gas_fees / len(valid_addresses) if valid_addresses else 0

        # Internal transfers
        internal_transfers = self._count_internal_transfers(valid_addresses, graph)

        # External interactions
        external_addresses = set()
        for s in stats_list:
            if s.unique_addresses_interacted_365d:
                external_addresses.add(s.address)

        return GroupStatistics(
            group_id=group_id,
            group_size=group_size,
            addresses=all_addresses,
            total_volume_usd_365d=total_volume,
            total_transactions_365d=total_transactions,
            total_outgoing_transactions_365d=total_outgoing,
            total_incoming_transactions_365d=total_incoming,
            avg_volume_per_address_365d=avg_volume_per_address,
            avg_transactions_per_address_365d=avg_transactions_per_address,
            max_volume_in_group_365d=max_volume,
            max_transactions_in_group_365d=max_transactions,
            oldest_wallet_age_days=oldest_age,
            newest_wallet_age_days=newest_age,
            avg_wallet_age_days=avg_age,
            total_active_days_365d=total_active_days,
            unique_months_active=len(unique_months),
            regular_wallets_count=regular_count,
            contracts_count=contracts_count,
            total_gas_fees_usd_365d=total_gas_fees,
            avg_gas_fees_per_address_365d=avg_gas_fees,
            internal_transfers_count=internal_transfers,
            external_unique_addresses=len(external_addresses),
            error_addresses=error_addresses
        )

    def _count_internal_transfers(self, addresses: List[str], graph) -> int:
        """Counts number of transfers within group"""
        internal_count = 0
        address_set = set(addresses)

        # Analyze graph edges
        for (from_addr, to_addr), edge in graph.edges.items():
            if from_addr in address_set and to_addr in address_set:
                internal_count += len(edge.transactions)

        return internal_count

    def create_group_volume_distribution(self):
        """Creates group volume distribution"""
        print("Creating group volume distribution...")

        if not self.group_stats:
            print("No group statistics available")
            return

        volumes = [group.total_volume_usd_365d for group in self.group_stats]

        # Define bins for groups
        bins = [
            (0, 10_000, "$0-$10K"),
            (10_000, 100_000, "$10K-$100K"),
            (100_000, 1_000_000, "$100K-$1M"),
            (1_000_000, 10_000_000, "$1M-$10M"),
            (10_000_000, float('inf'), "$10M+")
        ]

        bin_counts, bin_labels = self._calculate_bins(volumes, bins)

        # Create chart
        plt.figure(figsize=(12, 8))
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        bars = plt.bar(bin_labels, bin_counts, color=colors, alpha=0.8, edgecolor='black')

        # Add values on bars
        for bar, count in zip(bars, bin_counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bin_counts) * 0.01,
                         str(count), ha='center', va='bottom', fontweight='bold')

        plt.title('Distribution of Groups by Total Volume (365 days)', fontsize=16, fontweight='bold')
        plt.xlabel('Group Volume Range (USD)', fontsize=12)
        plt.ylabel('Number of Groups', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        total = len(self.group_stats)
        plt.figtext(0.02, 0.98, f'Total Groups: {total}', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "group_volume_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_group_size_analysis(self):
        """Creates group size analysis"""
        print("Creating group size analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Group Size Analysis', fontsize=16, fontweight='bold')

        # 1. Distribution by group sizes
        group_sizes = [group.group_size for group in self.group_stats]
        unique_sizes = sorted(set(group_sizes))
        size_counts = [group_sizes.count(size) for size in unique_sizes]

        axes[0, 0].bar(unique_sizes, size_counts, color='#3498db', alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('Distribution by Group Size')
        axes[0, 0].set_xlabel('Group Size (addresses)')
        axes[0, 0].set_ylabel('Number of Groups')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Volume vs group size
        volumes = [group.total_volume_usd_365d for group in self.group_stats]
        axes[0, 1].scatter(group_sizes, volumes, color='#e74c3c', alpha=0.7, s=50)
        axes[0, 1].set_title('Group Volume vs Size')
        axes[0, 1].set_xlabel('Group Size (addresses)')
        axes[0, 1].set_ylabel('Total Volume (USD)')
        if any(v > 0 for v in volumes):
            axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Average volume per address in group
        avg_volumes = [group.avg_volume_per_address_365d for group in self.group_stats]
        axes[1, 0].scatter(group_sizes, avg_volumes, color='#2ecc71', alpha=0.7, s=50)
        axes[1, 0].set_title('Average Volume per Address vs Group Size')
        axes[1, 0].set_xlabel('Group Size (addresses)')
        axes[1, 0].set_ylabel('Average Volume per Address (USD)')
        if any(v > 0 for v in avg_volumes):
            axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Group efficiency
        efficiency = np.array(avg_volumes)
        if any(efficiency > 0):
            axes[1, 1].hist(efficiency[efficiency > 0], bins=15, color='#f39c12', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Distribution of Group Efficiency')
            axes[1, 1].set_xlabel('Average Volume per Address (USD)')
            axes[1, 1].set_ylabel('Number of Groups')
            axes[1, 1].set_xscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'No volume data available',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "group_size_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_top_groups_analysis(self):
        """Creates top groups analysis"""
        print("Creating top groups analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Top Groups Analysis', fontsize=16, fontweight='bold')

        n_top = min(10, len(self.group_stats))

        # 1. Top by volume
        top_volume = sorted(self.group_stats, key=lambda x: x.total_volume_usd_365d, reverse=True)[:n_top]
        group_ids = [f"Group {g.group_id}\n({g.group_size} addr)" for g in top_volume]
        volumes = [g.total_volume_usd_365d for g in top_volume]

        y_pos = np.arange(len(top_volume))
        axes[0, 0].barh(y_pos, volumes, color='#e74c3c', alpha=0.8)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(group_ids, fontsize=9)
        axes[0, 0].set_xlabel('Total Volume (USD)')
        axes[0, 0].set_title(f'Top {n_top} Groups by Volume')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Top by transaction count
        top_tx = sorted(self.group_stats, key=lambda x: x.total_transactions_365d, reverse=True)[:n_top]
        group_ids_tx = [f"Group {g.group_id}\n({g.group_size} addr)" for g in top_tx]
        transactions = [g.total_transactions_365d for g in top_tx]

        y_pos = np.arange(len(top_tx))
        axes[0, 1].barh(y_pos, transactions, color='#3498db', alpha=0.8)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(group_ids_tx, fontsize=9)
        axes[0, 1].set_xlabel('Total Transactions')
        axes[0, 1].set_title(f'Top {n_top} Groups by Transactions')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Group efficiency
        top_efficiency = sorted(self.group_stats, key=lambda x: x.avg_volume_per_address_365d, reverse=True)[:n_top]
        group_ids_eff = [f"Group {g.group_id}\n({g.group_size} addr)" for g in top_efficiency]
        avg_volumes = [g.avg_volume_per_address_365d for g in top_efficiency]

        y_pos = np.arange(len(top_efficiency))
        axes[1, 0].barh(y_pos, avg_volumes, color='#2ecc71', alpha=0.8)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(group_ids_eff, fontsize=9)
        axes[1, 0].set_xlabel('Average Volume per Address (USD)')
        axes[1, 0].set_title(f'Top {n_top} Most Efficient Groups')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Internal group coordination
        coordination_data = [(g.group_id, g.group_size, g.internal_transfers_count) for g in self.group_stats]
        coordination_data.sort(key=lambda x: x[2], reverse=True)

        if coordination_data:
            top_coord = coordination_data[:n_top]
            group_ids_coord = [f"Group {g[0]}\n({g[1]} addr)" for g in top_coord]
            internal_transfers = [g[2] for g in top_coord]

            y_pos = np.arange(len(top_coord))
            axes[1, 1].barh(y_pos, internal_transfers, color='#9b59b6', alpha=0.8)
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(group_ids_coord, fontsize=9)
            axes[1, 1].set_xlabel('Internal Transfers Count')
            axes[1, 1].set_title(f'Top {n_top} Groups by Internal Coordination')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "top_groups_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_groups_vs_individuals_comparison(self, wallet_groups: List[List[str]]):
        """Creates groups vs individual addresses comparison"""
        print("Creating groups vs individuals comparison...")

        # Get addresses that are NOT in groups
        all_group_addresses = set()
        for group in wallet_groups:
            all_group_addresses.update(group)

        individual_addresses = []
        for addr, stats in self.individual_stats.items():
            if addr not in all_group_addresses:
                individual_addresses.append(stats)

        if not individual_addresses:
            print("No individual addresses found for comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Groups vs Individual Addresses Comparison', fontsize=16, fontweight='bold')

        # 1. Volume distribution
        group_volumes = [g.total_volume_usd_365d for g in self.group_stats if g.total_volume_usd_365d > 0]
        individual_volumes = [s.total_volume_usd_365d for s in individual_addresses if s.total_volume_usd_365d and s.total_volume_usd_365d > 0]

        axes[0, 0].hist([group_volumes, individual_volumes], bins=20, label=['Groups', 'Individuals'],
                        alpha=0.7, color=['red', 'blue'], edgecolor='black')
        axes[0, 0].set_xlabel('Total Volume (USD)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Volume Distribution: Groups vs Individuals')
        if group_volumes or individual_volumes:
            axes[0, 0].set_xscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Box plot volume comparison
        data_to_plot = []
        labels = []
        if group_volumes:
            data_to_plot.append(group_volumes)
            labels.append('Groups')
        if individual_volumes:
            data_to_plot.append(individual_volumes)
            labels.append('Individuals')

        if data_to_plot:
            axes[0, 1].boxplot(data_to_plot, labels=labels)
            axes[0, 1].set_ylabel('Total Volume (USD)')
            axes[0, 1].set_title('Volume Distribution Comparison')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Average values for groups vs individuals
        group_avg_volumes = [g.avg_volume_per_address_365d for g in self.group_stats if g.avg_volume_per_address_365d > 0]

        if group_avg_volumes and individual_volumes:
            axes[1, 0].hist([group_avg_volumes, individual_volumes], bins=15,
                            label=['Groups (avg per address)', 'Individuals'],
                            alpha=0.7, color=['orange', 'blue'], edgecolor='black')
            axes[1, 0].set_xlabel('Volume per Address (USD)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Volume per Address: Groups vs Individuals')
            axes[1, 0].set_xscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Statistics
        group_total_volume = sum(group_volumes) if group_volumes else 0
        individual_total_volume = sum(individual_volumes) if individual_volumes else 0
        total_addresses_in_groups = sum(g.group_size for g in self.group_stats)

        stats_text = f"""
        Groups Statistics:
        • Count: {len(self.group_stats)}
        • Total Volume: ${group_total_volume:,.0f}
        • Avg Volume per Group: ${group_total_volume / len(self.group_stats) if self.group_stats else 0:,.0f}
        
        Individual Statistics:
        • Count: {len(individual_addresses)}
        • Total Volume: ${individual_total_volume:,.0f}
        • Avg Volume per Individual: ${individual_total_volume / len(individual_addresses) if individual_addresses else 0:,.0f}
        
        Efficiency:
        • Groups control {group_total_volume / (group_total_volume + individual_total_volume) * 100 if (group_total_volume + individual_total_volume) > 0 else 0:.1f}% of volume
        • With {total_addresses_in_groups}/{total_addresses_in_groups + len(individual_addresses) * 100 if (total_addresses_in_groups + len(individual_addresses)) > 0 else 0:.1f}% of addresses
        """

        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "groups_vs_individuals_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_transaction_volume_bins_for_all_groups(self):
        """Creates bin plot by total transaction volume for ALL groups"""
        print("Creating transaction volume bins for all groups...")

        if not self.group_stats:
            print("No group statistics available")
            return

        # Get transaction volume data for all groups
        transaction_volumes = [group.total_transactions_365d for group in self.group_stats]

        # Define bins by transaction count
        bins = [
            (0, 100, "0-100 tx"),
            (100, 500, "100-500 tx"),
            (500, 1000, "500-1K tx"),
            (1000, 5000, "1K-5K tx"),
            (5000, 10000, "5K-10K tx"),
            (10000, float('inf'), "10K+ tx")
        ]

        bin_counts, bin_labels = self._calculate_bins(transaction_volumes, bins)

        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Transaction Volume Distribution for All Groups (365 days)', fontsize=16, fontweight='bold')

        # 1. Bin chart
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
        bars = ax1.bar(bin_labels, bin_counts, color=colors, alpha=0.8, edgecolor='black')

        # Add values on bars
        for bar, count in zip(bars, bin_counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bin_counts) * 0.01,
                         str(count), ha='center', va='bottom', fontweight='bold')

        ax1.set_title('Groups Distribution by Transaction Count')
        ax1.set_xlabel('Transaction Count Range')
        ax1.set_ylabel('Number of Groups')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Cumulative distribution
        sorted_volumes = sorted(transaction_volumes, reverse=True)
        cumulative_percent = [(i + 1) / len(sorted_volumes) * 100 for i in range(len(sorted_volumes))]

        ax2.plot(sorted_volumes, cumulative_percent, 'o-', color='#e74c3c', linewidth=2, markersize=4)
        ax2.set_title('Cumulative Transaction Distribution')
        ax2.set_xlabel('Total Transactions (365d)')
        ax2.set_ylabel('Cumulative Percentage of Groups (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')

        # Add statistics
        total_groups = len(self.group_stats)
        total_transactions = sum(transaction_volumes)
        avg_transactions = total_transactions / total_groups if total_groups > 0 else 0
        median_transactions = sorted(transaction_volumes)[len(transaction_volumes) // 2] if transaction_volumes else 0

        stats_text = f"""
        Total Groups: {total_groups}
        Total Transactions: {total_transactions:,}
        Average per Group: {avg_transactions:,.0f}
        Median per Group: {median_transactions:,}
        """

        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "transaction_volume_bins_all_groups.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Print statistics to log
        self._print_bins_stats("Transaction Volume", bin_labels, bin_counts, np.array(transaction_volumes))

    def _print_bins_stats(self, prefix, bin_labels, bin_counts, data):
        """Prints bin statistics"""
        total = sum(bin_counts)
        print(f"📊 {prefix} distribution:")
        for label, count in zip(bin_labels, bin_counts):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {label}: {count} groups ({percentage:.1f}%)")

        print(f"📈 {prefix} statistics:")
        print(f"  Total: {data.sum():,.0f}")
        print(f"  Average: {data.mean():,.0f}")
        print(f"  Median: {np.median(data):,.0f}")
        print(f"  Max: {data.max():,.0f}")
        print(f"  Min: {data.min():,.0f}")

    def generate_groups_report(self):
        """Generates basic group statistics summary"""
        print("Generating groups statistics summary...")

        if not self.group_stats:
            print("No group statistics for report")
            return {}

        df_groups = pd.DataFrame([asdict(group) for group in self.group_stats])

        # Basic statistics
        total_groups = len(self.group_stats)
        total_addresses_in_groups = df_groups['group_size'].sum()
        total_volume = df_groups['total_volume_usd_365d'].sum()
        total_transactions = df_groups['total_transactions_365d'].sum()
        avg_group_size = df_groups['group_size'].mean()
        avg_volume_per_group = df_groups['total_volume_usd_365d'].mean()

        # Group efficiency
        largest_group = df_groups['group_size'].max()
        most_active_group_id = df_groups.loc[df_groups['total_transactions_365d'].idxmax(), 'group_id']
        highest_volume_group_id = df_groups.loc[df_groups['total_volume_usd_365d'].idxmax(), 'group_id']

        stats = {
            'total_groups': total_groups,
            'total_addresses_in_groups': total_addresses_in_groups,
            'avg_group_size': avg_group_size,
            'largest_group_size': largest_group,
            'total_volume': total_volume,
            'avg_volume_per_group': avg_volume_per_group,
            'total_transactions': total_transactions,
            'most_active_group_id': most_active_group_id,
            'highest_volume_group_id': highest_volume_group_id
        }

        # Output statistics to log
        print("=" * 50)
        print("📊 GROUPS ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total Groups: {stats['total_groups']}")
        print(f"Total Addresses in Groups: {stats['total_addresses_in_groups']}")
        print(f"Average Group Size: {stats['avg_group_size']:.1f}")
        print(f"Largest Group Size: {stats['largest_group_size']}")
        print(f"Total Volume: ${stats['total_volume']:,.0f}")
        print(f"Average Volume per Group: ${stats['avg_volume_per_group']:,.0f}")
        print(f"Total Transactions: {stats['total_transactions']:,.0f}")
        print(f"Most Active Group: {stats['most_active_group_id']}")
        print(f"Highest Volume Group: {stats['highest_volume_group_id']}")
        print("=" * 50)

        # Top 5 groups in log
        print("🏆 TOP 5 GROUPS BY VOLUME:")
        top_5_groups = df_groups.nlargest(5, 'total_volume_usd_365d')
        for i, (_, group) in enumerate(top_5_groups.iterrows(), 1):
            print(f"{i}. Group {group['group_id']} - {group['group_size']} addresses")
            print(f"   Volume: ${group['total_volume_usd_365d']:,.0f} | Transactions: {group['total_transactions_365d']:,}")

        print("=" * 50)

        return stats

    def save_groups_data(self):
        """Saves group data to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine prefix based on analysis type
        if FULL_ANALYSIS_AVAILABLE:
            prefix = "groups_full_analysis_365d"
        else:
            prefix = "groups_analysis"

        # Save to JSON
        json_file = self.output_dir / "data" / f"{prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json_data = [asdict(group) for group in self.group_stats]
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Save to CSV
        csv_file = self.output_dir / "data" / f"{prefix}_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.group_stats:
                writer = csv.DictWriter(f, fieldnames=asdict(self.group_stats[0]).keys())
                writer.writeheader()
                for group in self.group_stats:
                    writer.writerow(asdict(group))

        print(f"Groups data saved to: {json_file} and {csv_file}")
        return json_file, csv_file

    def _calculate_bins(self, data, bins):
        """Counts elements in bins"""
        bin_counts = []
        bin_labels = []

        for min_val, max_val, label in bins:
            if max_val == float('inf'):
                count = len([x for x in data if x >= min_val])
            else:
                count = len([x for x in data if min_val <= x < max_val])

            bin_counts.append(count)
            bin_labels.append(label)

        return bin_counts, bin_labels
