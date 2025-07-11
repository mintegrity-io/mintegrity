#!/usr/bin/env python3
"""
Rocket Pool Data Fetcher

This script handles all data retrieval operations for Rocket Pool analysis:
1. Loads pre-built transaction graph
2. Extracts addresses that interact with Rocket Pool contracts  
3. Fetches transaction data from Etherscan API (365 days)
4. Fetches data from Rocket Pool Subgraph (365 days)
5. Retrieves historical token prices using pre-loaded prices
6. Saves raw data in structured format for later analysis

The output can be used by the analysis script in scripts/stats_vis/
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Set, Optional

import requests
from tqdm import tqdm

from scripts.commons import prices
from scripts.commons.logging_config import get_logger
from scripts.commons.model import AddressType
from scripts.graph.model.transactions_graph import TransactionsGraph, NodeType
from scripts.graph.util.transactions_graph_json import load_graph_from_json
from scripts.subgraph.client import SubgraphClient
# Use existing functions instead of duplicating
from scripts.eth_transactions_scraper import get_address_types, _save_address_type_cache

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
    api_sources: List[str]
    success_rate: float
    time_period_days: int = 365


class RocketPoolDataFetcher:
    """Fetches raw data from Rocket Pool related APIs"""
    
    def __init__(self,
                 graph_file_path: str,
                 output_dir: str = "files/rocket_pool_data",
                 max_workers: int = 3):
        
        self.graph_file_path = Path(graph_file_path)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.price_cache = {}
        
        # Initialize subgraph client
        self.subgraph_client = SubgraphClient()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata prices
        prices.init()
        
        log.info(f"Initialized Rocket Pool Data Fetcher")
        log.info(f"Graph file: {self.graph_file_path}")
        log.info(f"Output directory: {self.output_dir}")
        log.info(f"Mode: Rocket Pool interactions only")

    def get_historical_token_price(self, token_symbol: str, timestamp: int) -> float:
        """Get historical token price using pre-loaded prices from prices.py"""
        cache_key = f"{token_symbol.upper()}-{timestamp}"
        
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        try:
            price = prices.get_token_price_usd(token_symbol, str(timestamp))
            if price > 0:
                self.price_cache[cache_key] = price
                return price
        except Exception as e:
            log.debug(f"Price lookup failed for {token_symbol}: {e}")
        
        # Fallback to ETH price if token not found
        if token_symbol.upper() != 'ETH':
            try:
                eth_price = prices.get_token_price_usd('ETH', str(timestamp))
                if eth_price > 0:
                    log.warning(f"Token {token_symbol} not found, using ETH price as fallback: ${eth_price}")
                    self.price_cache[cache_key] = eth_price
                    return eth_price
            except Exception as e:
                log.debug(f"ETH fallback price lookup failed: {e}")
        
        # Final fallback
        fallback_price = prices.CURRENT_TOKEN_PRICES.get('ETH', 2500.0)
        log.warning(f"Using fallback price for {token_symbol}: ${fallback_price}")
        self.price_cache[cache_key] = fallback_price
        return fallback_price

    def get_current_token_prices(self) -> Dict[str, float]:
        """Get current token prices using pre-loaded prices from prices.py"""
        return prices.CURRENT_TOKEN_PRICES.copy()

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
        
        for address, node in graph.nodes.items():
            if node.type == NodeType.ROOT:
                rocket_pool_contracts.add(address.lower())
                log.info(f"Found Rocket Pool ROOT contract: {address}")
        
        log.info(f"Identified {len(rocket_pool_contracts)} Rocket Pool contracts")
        return rocket_pool_contracts

    def extract_rocket_pool_interacting_addresses(self, graph: TransactionsGraph) -> Set[str]:
        """Extract addresses that directly interact with Rocket Pool contracts"""
        rocket_pool_contracts = self.identify_rocket_pool_contracts(graph)
        
        if not rocket_pool_contracts:
            raise ValueError("No Rocket Pool contracts identified in the graph!")
        
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
        
        if not interacting_addresses:
            raise ValueError("No addresses found that interact with Rocket Pool contracts!")
        
        return interacting_addresses

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

    def fetch_address_data(self, address: str, address_type: str) -> AddressRawData:
        """Fetch all data for a single address"""
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
        subgraph_data = self.subgraph_client.fetch_user_data(address)
        if subgraph_data:
            if "error" in subgraph_data:
                raw_data.fetch_errors.append(f"Subgraph: {subgraph_data['error']}")
            else:
                raw_data.subgraph_data = subgraph_data
        
        return raw_data

    def fetch_all_data(self, addresses: Set[str]) -> List[AddressRawData]:
        """Fetch data for all addresses using batch address type detection"""
        addresses_list = list(addresses)
        
        # Use efficient batch processing from eth_transactions_scraper
        log.info(f"Getting address types for {len(addresses_list)} addresses...")
        address_types_enum = get_address_types(addresses_list, network="eth-mainnet")
        
        # Convert to strings
        address_types = {}
        for addr, addr_type in address_types_enum.items():
            address_types[addr] = "wallet" if addr_type == AddressType.WALLET else "contract"
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_address = {
                executor.submit(
                    self.fetch_address_data, 
                    address, 
                    address_types.get(address, "wallet")
                ): address 
                for address in addresses_list
            }
            
            with tqdm(total=len(addresses_list), desc="Fetching address data") as pbar:
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
        
        # Save cache from eth_transactions_scraper
        _save_address_type_cache()
        
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
        
        # Create metadata (simplified without analysis_mode)
        metadata_obj = FetchMetadata(
            fetch_timestamp=datetime.now(timezone.utc).isoformat(),
            graph_source=str(self.graph_file_path),
            addresses_analyzed=len(addresses_data),
            api_sources=["etherscan", "subgraph"],
            success_rate=success_rate
        )
        
        # Get current token prices
        token_prices = self.get_current_token_prices()
        
        # Prepare output data
        output_data = {
            "metadata": asdict(metadata_obj),
            "addresses": {data.address: asdict(data) for data in addresses_data},
            "token_prices": token_prices
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
            graph = self.load_graph()
            addresses = self.extract_rocket_pool_interacting_addresses(graph)
            
            log.info(f"Fetching data for {len(addresses)} Rocket Pool users...")
            addresses_data = self.fetch_all_data(addresses)
            
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
    project_root = current_dir.parent.parent.parent
    
    graph_path = str(project_root / "files" / "rocket_pool_full_graph_90_days.json")
    output_dir = str(project_root / "files" / "rocket_pool_data")
    max_workers = 3
    
    try:
        fetcher = RocketPoolDataFetcher(
            graph_file_path=graph_path,
            output_dir=output_dir,
            max_workers=max_workers
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
