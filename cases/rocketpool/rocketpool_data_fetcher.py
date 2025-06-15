#!/usr/bin/env python3
"""
Rocket Pool Data Fetcher

This script handles all data retrieval operations for Rocket Pool analysis:
1. Loads pre-built transaction graph
2. Extracts addresses that interact with Rocket Pool contracts  
3. Fetches transaction data from Etherscan API (365 days)
4. Fetches data from Rocket Pool Subgraph (365 days)
5. Retrieves historical token prices
6. Saves raw data in structured format for later analysis

The output can be used by the analysis script in scripts/stats_vis/
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set, Optional

import requests
from tqdm import tqdm

from scripts.commons import metadata
from scripts.commons.known_token_list import ETH_TOKENS_WHITELIST
from scripts.commons.logging_config import get_logger
from scripts.commons.model import AddressType
from scripts.commons.tokens_metadata_scraper import fetch_current_token_prices
from scripts.graph.model.transactions_graph import TransactionsGraph, NodeType
from scripts.graph.util.transactions_graph_json import load_graph_from_json

log = get_logger()


@dataclass
class AddressRawData:
    """Raw data fetched for a single address"""
    address: str
    address_type: str
    etherscan_data: Optional[Dict] = None
    subgraph_data: Optional[Dict] = None
    fetch_errors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.fetch_errors is None:
            self.fetch_errors = []


@dataclass
class FetchMetadata:
    """Metadata about the data fetching operation"""
    fetch_timestamp: str
    graph_source: str
    addresses_analyzed: int
    analysis_mode: str  # "rocket_pool_only" or "all_addresses"
    api_sources: List[str]
    success_rate: float
    time_period_days: int = 365


class RocketPoolDataFetcher:
    """Fetches raw data from Rocket Pool related APIs"""
    
    def __init__(self,
                 graph_file_path: str,
                 output_dir: str = "files/rocket_pool_data",
                 max_workers: int = 3,
                 analyze_all_addresses: bool = False):
        
        self.graph_file_path = Path(graph_file_path)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.analyze_all_addresses = analyze_all_addresses
        self.price_cache = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata
        metadata.init()
        
        # Get current token prices
        self.current_token_prices = self._fetch_current_prices()
        
        log.info(f"Initialized Rocket Pool Data Fetcher")
        log.info(f"Graph file: {self.graph_file_path}")
        log.info(f"Output directory: {self.output_dir}")
        log.info(f"Mode: {'All addresses' if analyze_all_addresses else 'Rocket Pool interactions only'}")

    def _fetch_current_prices(self) -> Dict[str, float]:
        """Fetch current token prices as fallback"""
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
            
            fallback_prices = {}
            for token in ETH_TOKENS_WHITELIST:
                price = metadata.get_token_price_usd(token, str(int(time.time())))
                if price > 0:
                    fallback_prices[token] = price
                    
            return fallback_prices

    def get_historical_token_price(self, token_symbol: str, timestamp: int) -> float:
        """Get historical token price at specific timestamp"""
        cache_key = f"{token_symbol.upper()}-{timestamp}"
        
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        # Try metadata first
        try:
            price = metadata.get_token_price_usd(token_symbol, str(timestamp))
            if price > 0:
                self.price_cache[cache_key] = price
                return price
        except Exception as e:
            log.debug(f"Metadata price lookup failed for {token_symbol}: {e}")
        
        # Use external API
        try:
            token_to_pair = {
                'ETH': 'ETH-USD', 'BTC': 'BTC-USD', 'WETH': 'ETH-USD',
                'USDT': 'USDT-USD', 'USDC': 'USDC-USD', 'DAI': 'DAI-USD',
                'LINK': 'LINK-USD', 'UNI': 'UNI-USD', 'AAVE': 'AAVE-USD',
                'MKR': 'MKR-USD', 'CRV': 'CRV-USD', 'COMP': 'COMP-USD',
                'SNX': 'SNX-USD', 'GRT': 'GRT-USD', 'LDO': 'LDO-USD',
                'MATIC': 'MATIC-USD', 'SHIB': 'SHIB-USD'
            }
            
            pair = token_to_pair.get(token_symbol.upper())
            if not pair:
                log.warning(f"Token {token_symbol} not supported, using ETH price as fallback")
                if token_symbol.upper() != 'ETH':
                    return self.get_historical_token_price('ETH', timestamp)
                pair = 'ETH-USD'
            
            # Coinbase API
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
            
            if not candles:
                # Try wider range
                start_time = timestamp - 86400
                end_time = timestamp + 86400
                params.update({
                    'start': datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
                    'end': datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
                    'granularity': 86400
                })
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                candles = response.json()
            
            if not candles:
                return self.current_token_prices.get(token_symbol, self.current_token_prices.get('ETH', 2500.0))
            
            closest_candle = min(candles, key=lambda x: abs(x[0] - timestamp))
            price = float(closest_candle[4])  # close price
            
            self.price_cache[cache_key] = price
            return price
            
        except Exception as e:
            log.warning(f"Error getting historical price for {token_symbol}: {e}")
            return self.current_token_prices.get(token_symbol, self.current_token_prices.get('ETH', 2500.0))

    def load_graph(self) -> TransactionsGraph:
        """Load transaction graph from file"""
        if not self.graph_file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.graph_file_path}")
        
        try:
            log.info(f"Loading graph from {self.graph_file_path}")
            graph = load_graph_from_json(str(self.graph_file_path))
            log.info(f"Successfully loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
        except Exception as e:
            raise RuntimeError(f"Failed to load graph: {e}")

    def identify_rocket_pool_contracts(self, graph: TransactionsGraph) -> Set[str]:
        """Identify Rocket Pool contract addresses in the graph"""
        rocket_pool_contracts = set()
        
        # ROOT nodes are usually main Rocket Pool contracts
        for address, node in graph.nodes.items():
            if node.type == NodeType.ROOT:
                rocket_pool_contracts.add(address.lower())
                log.info(f"Found Rocket Pool ROOT contract: {address}")
        
        log.info(f"Identified {len(rocket_pool_contracts)} Rocket Pool contracts")
        return rocket_pool_contracts

    def extract_target_addresses(self, graph: TransactionsGraph) -> Set[str]:
        """Extract addresses to analyze based on configuration"""
        if self.analyze_all_addresses:
            return self._extract_all_addresses(graph)
        else:
            return self._extract_rocket_pool_interacting_addresses(graph)

    def _extract_rocket_pool_interacting_addresses(self, graph: TransactionsGraph) -> Set[str]:
        """Extract addresses that directly interact with Rocket Pool contracts"""
        rocket_pool_contracts = self.identify_rocket_pool_contracts(graph)
        
        if not rocket_pool_contracts:
            log.warning("No Rocket Pool contracts identified! Using all addresses as fallback")
            return self._extract_all_addresses(graph)
        
        interacting_addresses = set()
        rp_transactions = 0
        
        for edge in graph.edges.values():
            for transaction in edge.transactions.values():
                from_addr = transaction.address_from.address.lower()
                to_addr = transaction.address_to.address.lower()
                
                is_rp_transaction = (from_addr in rocket_pool_contracts or 
                                   to_addr in rocket_pool_contracts)
                
                if is_rp_transaction:
                    rp_transactions += 1
                    if from_addr not in rocket_pool_contracts:
                        interacting_addresses.add(from_addr)
                    if to_addr not in rocket_pool_contracts:
                        interacting_addresses.add(to_addr)
        
        log.info(f"Found {rp_transactions} Rocket Pool transactions")
        log.info(f"Extracted {len(interacting_addresses)} addresses that interact with Rocket Pool")
        return interacting_addresses

    def _extract_all_addresses(self, graph: TransactionsGraph) -> Set[str]:
        """Extract all addresses from graph (excluding ROOT contracts)"""
        all_addresses = set()
        
        for address, node in graph.nodes.items():
            if node.type != NodeType.ROOT:
                all_addresses.add(address.lower())
        
        for edge in graph.edges.values():
            for transaction in edge.transactions.values():
                from_addr = transaction.address_from.address.lower()
                to_addr = transaction.address_to.address.lower()
                
                if from_addr in graph.nodes and graph.nodes[from_addr].type != NodeType.ROOT:
                    all_addresses.add(from_addr)
                if to_addr in graph.nodes and graph.nodes[to_addr].type != NodeType.ROOT:
                    all_addresses.add(to_addr)
        
        log.info(f"Extracted {len(all_addresses)} unique addresses (excluding ROOT contracts)")
        return all_addresses

    def get_address_type(self, graph: TransactionsGraph, address: str) -> str:
        """Determine address type from graph"""
        normalized_address = address.lower()
        
        if normalized_address in graph.nodes:
            node = graph.nodes[normalized_address]
            if node.type == NodeType.WALLET:
                return "wallet"
            elif node.type in [NodeType.CONTRACT, NodeType.ROOT]:
                return "contract"
        
        # Check in transactions
        for edge in graph.edges.values():
            for transaction in edge.transactions.values():
                if transaction.address_from.address.lower() == normalized_address:
                    return "wallet" if transaction.address_from.type == AddressType.WALLET else "contract"
                elif transaction.address_to.address.lower() == normalized_address:
                    return "wallet" if transaction.address_to.type == AddressType.WALLET else "contract"
        
        return "wallet"  # Default assumption

    def fetch_etherscan_data(self, address: str) -> Optional[Dict]:
        """Fetch raw transaction data from Etherscan API"""
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not etherscan_api_key:
            return None
        
        try:
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
                "offset": 100000,
                "sort": "asc",
                "apikey": etherscan_api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data["status"] != "1":
                return {"error": f"Etherscan API error: {data.get('message', 'Unknown error')}"}
            
            # Filter for 365 days
            start_timestamp = int(start_time.timestamp())
            transactions = [tx for tx in data["result"] 
                          if int(tx["timeStamp"]) >= start_timestamp]
            
            return {
                "transactions": transactions,
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                "period_days": 365
            }
            
        except Exception as e:
            return {"error": f"Etherscan fetch error: {str(e)}"}

    def fetch_subgraph_data(self, address: str) -> Optional[Dict]:
        """Fetch data from Rocket Pool Subgraph"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=365)
            start_timestamp = int(start_time.timestamp())
            
            subgraph_url = "https://api.thegraph.com/subgraphs/name/rocket-pool/rocketpool"
            
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
            
            response = requests.post(
                subgraph_url,
                json={"query": query, "variables": {"address": address.lower(), "startTimestamp": start_timestamp}},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "errors" in data:
                return {"error": f"Subgraph error: {data['errors']}"}
            
            user_data = data.get("data", {}).get("user")
            if not user_data:
                return {"transactions": [], "fetch_timestamp": datetime.now(timezone.utc).isoformat()}
            
            # Combine deposits and withdrawals
            all_transactions = []
            all_transactions.extend(user_data.get("deposits", []))
            all_transactions.extend(user_data.get("withdrawals", []))
            
            return {
                "transactions": sorted(all_transactions, key=lambda x: int(x["timestamp"])),
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                "period_days": 365
            }
            
        except Exception as e:
            return {"error": f"Subgraph fetch error: {str(e)}"}

    def fetch_address_data(self, address: str, graph: TransactionsGraph) -> AddressRawData:
        """Fetch all data for a single address"""
        address_type = self.get_address_type(graph, address)
        raw_data = AddressRawData(address=address, address_type=address_type)
        
        # Fetch Etherscan data
        etherscan_data = self.fetch_etherscan_data(address)
        if etherscan_data:
            if "error" in etherscan_data:
                raw_data.fetch_errors.append(f"Etherscan: {etherscan_data['error']}")
            else:
                raw_data.etherscan_data = etherscan_data
        else:
            raw_data.fetch_errors.append("Etherscan: API key not available")
        
        # Fetch Subgraph data
        subgraph_data = self.fetch_subgraph_data(address)
        if subgraph_data:
            if "error" in subgraph_data:
                raw_data.fetch_errors.append(f"Subgraph: {subgraph_data['error']}")
            else:
                raw_data.subgraph_data = subgraph_data
        
        return raw_data

    def fetch_all_data(self, addresses: Set[str], graph: TransactionsGraph) -> List[AddressRawData]:
        """Fetch data for all addresses using multithreading"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_address = {
                executor.submit(self.fetch_address_data, address, graph): address 
                for address in addresses
            }
            
            with tqdm(total=len(addresses), desc="Fetching address data") as pbar:
                for future in as_completed(future_to_address):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        address = future_to_address[future]
                        log.warning(f"Failed to fetch data for {address}: {e}")
                        error_data = AddressRawData(
                            address=address,
                            address_type="unknown",
                            fetch_errors=[f"Fetch failed: {str(e)}"]
                        )
                        results.append(error_data)
                    finally:
                        pbar.update(1)
        
        return results

    def save_raw_data(self, addresses_data: List[AddressRawData], graph_source: str) -> str:
        """Save raw data to file for later analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate success rate
        successful_fetches = sum(1 for data in addresses_data 
                               if not data.fetch_errors or 
                               (data.etherscan_data and "error" not in data.etherscan_data) or
                               (data.subgraph_data and "error" not in data.subgraph_data))
        
        success_rate = successful_fetches / len(addresses_data) if addresses_data else 0.0
        
        # Create metadata
        metadata_obj = FetchMetadata(
            fetch_timestamp=datetime.now(timezone.utc).isoformat(),
            graph_source=str(self.graph_file_path),
            addresses_analyzed=len(addresses_data),
            analysis_mode="all_addresses" if self.analyze_all_addresses else "rocket_pool_only",
            api_sources=["etherscan", "subgraph"],
            success_rate=success_rate
        )
        
        # Prepare output data
        output_data = {
            "metadata": asdict(metadata_obj),
            "addresses": {data.address: asdict(data) for data in addresses_data},
            "token_prices": self.current_token_prices
        }
        
        # Save to file
        output_file = self.output_dir / f"rocket_pool_raw_data_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        log.info(f"Saved raw data to {output_file}")
        log.info(f"Success rate: {successful_fetches}/{len(addresses_data)} ({success_rate*100:.1f}%)")
        
        return str(output_file)

    def run_fetch(self) -> str:
        """Run the complete data fetching process"""
        log.info("=" * 60)
        log.info("ROCKET POOL DATA FETCHING STARTED")
        log.info("=" * 60)
        
        try:
            # Load graph
            graph = self.load_graph()
            
            # Extract target addresses
            addresses = self.extract_target_addresses(graph)
            
            if not addresses:
                raise ValueError("No addresses found to analyze")
            
            # Fetch all data
            log.info(f"Fetching data for {len(addresses)} addresses...")
            addresses_data = self.fetch_all_data(addresses, graph)
            
            # Save raw data
            output_file = self.save_raw_data(addresses_data, str(self.graph_file_path))
            
            log.info("=" * 60)
            log.info("DATA FETCHING COMPLETED")
            log.info(f"Raw data saved to: {output_file}")
            log.info("Use the wallet statistics analyzer to process this data")
            log.info("=" * 60)
            
            return output_file
            
        except Exception as e:
            log.error(f"Data fetching failed: {e}")
            raise


def main():
    """Main function with hardcoded parameters"""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent.parent  # cases/rocketpool/ -> mintegrity/
    
    # Hardcoded configuration
    graph_path = str(project_root / "files" / "rocket_pool_full_graph_90_days.json")
    output_dir = str(project_root / "files" / "rocket_pool_data")
    max_workers = 3
    analyze_all_addresses = False  # Set to True to analyze all addresses instead of just Rocket Pool interactions
    
    try:
        fetcher = RocketPoolDataFetcher(
            graph_file_path=graph_path,
            output_dir=output_dir,
            max_workers=max_workers,
            analyze_all_addresses=analyze_all_addresses
        )
        
        output_file = fetcher.run_fetch()
        print(f"\nSuccess! Raw data saved to: {output_file}")
        print("Next step: Use wallet_statistics_analyzer.py to analyze this data")
        
    except KeyboardInterrupt:
        log.info("Data fetching interrupted by user")
        return 1
    except Exception as e:
        log.error(f"Data fetching failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
