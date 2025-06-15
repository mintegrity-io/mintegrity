#!/usr/bin/env python3
"""
Wallet Statistics Analyzer

This script analyzes raw wallet data fetched by the Rocket Pool Data Fetcher:
1. Loads raw data from JSON file created by the data fetcher
2. Performs statistical analysis and calculations
3. Generates comprehensive reports and summaries
4. Saves analysis results in JSON and CSV formats

This script works with data in the format saved by cases/rocketpool/rocketpool_data_fetcher.py
"""

import csv
import json
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from scripts.commons.logging_config import get_logger

log = get_logger()


@dataclass
class WalletStatistics:
    """Analyzed address statistics for 365 days"""
    address: str
    address_type: str = None
    
    # Basic info
    creation_date: Optional[str] = None
    first_transaction_date: Optional[str] = None
    last_transaction_date: Optional[str] = None
    wallet_age_days: Optional[int] = None
    
    # 365 day volume metrics
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
    
    # Data sources used
    data_sources: Optional[List[str]] = None
    analysis_errors: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = []
        if self.analysis_errors is None:
            self.analysis_errors = []


class WalletStatisticsAnalyzer:
    """Analyzes raw wallet data and generates statistics"""
    
    def __init__(self, output_dir: str = "files/wallet_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.token_prices = {}
        
        log.info(f"Initialized Wallet Statistics Analyzer")
        log.info(f"Output directory: {self.output_dir}")

    def load_raw_data(self, raw_data_file: str) -> Dict[str, Any]:
        """Load raw data from file created by data fetcher"""
        raw_data_path = Path(raw_data_file)
        
        if not raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
        
        try:
            with open(raw_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            log.info(f"Loaded raw data from {raw_data_path}")
            log.info(f"Metadata: {data.get('metadata', {})}")
            log.info(f"Number of addresses: {len(data.get('addresses', {}))}")
            
            # Store token prices for calculations
            self.token_prices = data.get('token_prices', {})
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in raw data file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load raw data: {e}")

    def get_token_price_usd(self, token_symbol: str, timestamp: Optional[str] = None) -> float:
        """Get token price in USD (simplified version for analysis)"""
        # For analysis, we'll use current prices as stored in the raw data
        # In a more sophisticated version, this could use historical prices
        token_symbol_upper = token_symbol.upper()
        
        if token_symbol_upper in self.token_prices:
            return self.token_prices[token_symbol_upper]
        
        # Fallback mappings
        if token_symbol_upper == 'WETH' and 'ETH' in self.token_prices:
            return self.token_prices['ETH']
        
        # Default fallback
        return self.token_prices.get('ETH', 2500.0)

    def calculate_transaction_value_usd(self, value_token: float, token_symbol: str, timestamp_str: str) -> float:
        """Calculate transaction value in USD"""
        try:
            token_price = self.get_token_price_usd(token_symbol, timestamp_str)
            return value_token * token_price
        except Exception as e:
            log.warning(f"Error calculating USD value for {value_token} {token_symbol}: {e}")
            return value_token * self.get_token_price_usd('ETH')

    def analyze_etherscan_data(self, address: str, address_type: str, etherscan_data: Dict) -> WalletStatistics:
        """Analyze Etherscan transaction data"""
        if "error" in etherscan_data:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                analysis_errors=[f"Etherscan data error: {etherscan_data['error']}"]
            )
        
        transactions = etherscan_data.get("transactions", [])
        if not transactions:
            return self._create_empty_wallet_stats(address, address_type, ["etherscan"])
        
        volumes_usd_365d = []
        outgoing_tx_365d = []
        incoming_tx_365d = []
        
        daily_volumes = {}
        monthly_volumes = {}
        gas_used_total = 0
        gas_fees_usd_total = 0.0
        unique_addresses = set()
        wallet_interactions = 0
        contract_interactions = 0
        
        for tx in transactions:
            # Convert wei to ETH
            value_wei = int(tx["value"])
            value_eth = value_wei / 10 ** 18
            
            timestamp = int(tx["timeStamp"])
            from_addr = tx["from"].lower()
            to_addr = tx["to"].lower()
            
            # Determine transaction direction
            is_outgoing = from_addr == address.lower()
            is_incoming = to_addr == address.lower()
            
            if is_outgoing:
                outgoing_tx_365d.append(tx)
            if is_incoming:
                incoming_tx_365d.append(tx)
            
            # Analyze outgoing transactions for volume
            if is_outgoing and value_eth > 0:
                value_usd = self.calculate_transaction_value_usd(value_eth, 'ETH', str(timestamp))
                volumes_usd_365d.append(value_usd)
                
                # Daily and monthly activity
                tx_date = datetime.fromtimestamp(timestamp, timezone.utc).date()
                month_key = tx_date.strftime('%Y-%m')
                
                if tx_date not in daily_volumes:
                    daily_volumes[tx_date] = 0
                daily_volumes[tx_date] += value_usd
                
                if month_key not in monthly_volumes:
                    monthly_volumes[month_key] = 0
                monthly_volumes[month_key] += value_usd
            
            # Gas analysis (only outgoing transactions)
            if is_outgoing:
                gas_used = int(tx.get("gasUsed", 0))
                gas_price = int(tx.get("gasPrice", 0))
                gas_used_total += gas_used
                
                # Convert gas fee to USD
                gas_fee_eth = (gas_used * gas_price) / 10 ** 18
                gas_fee_usd = self.calculate_transaction_value_usd(gas_fee_eth, 'ETH', str(timestamp))
                gas_fees_usd_total += gas_fee_usd
            
            # Interaction analysis
            if is_outgoing:
                other_address = to_addr
            else:
                other_address = from_addr
            
            unique_addresses.add(other_address)
            
            # Simple heuristic: no input data = wallet, with input data = contract
            if len(other_address) == 42:  # Valid Ethereum address
                if tx.get("input", "0x") == "0x":
                    wallet_interactions += 1
                else:
                    contract_interactions += 1
        
        # Calculate wallet age and dates
        all_timestamps = [int(tx["timeStamp"]) for tx in transactions]
        first_timestamp = min(all_timestamps) if all_timestamps else None
        last_timestamp = max(all_timestamps) if all_timestamps else None
        
        first_date = datetime.fromtimestamp(first_timestamp, timezone.utc) if first_timestamp else None
        last_date = datetime.fromtimestamp(last_timestamp, timezone.utc) if last_timestamp else None
        wallet_age_days = (datetime.now(timezone.utc) - first_date).days if first_date else None
        
        # Calculate volume statistics
        total_volume_usd_365d = sum(volumes_usd_365d)
        average_volume_usd_365d = total_volume_usd_365d / len(volumes_usd_365d) if volumes_usd_365d else 0.0
        max_volume_usd_365d = max(volumes_usd_365d) if volumes_usd_365d else 0.0
        median_volume_usd_365d = statistics.median(volumes_usd_365d) if volumes_usd_365d else 0.0
        
        # Activity patterns
        active_days = len(daily_volumes)
        avg_daily_volume = sum(daily_volumes.values()) / len(daily_volumes) if daily_volumes else 0.0
        max_daily_volume = max(daily_volumes.values()) if daily_volumes else 0.0
        most_active_month = max(monthly_volumes.items(), key=lambda x: x[1])[0] if monthly_volumes else None
        
        # Gas statistics
        avg_gas_price_gwei = 0.0
        if outgoing_tx_365d:
            total_gas_price_wei = sum(int(tx.get("gasPrice", 0)) for tx in outgoing_tx_365d)
            avg_gas_price_gwei = (total_gas_price_wei / len(outgoing_tx_365d)) / 10 ** 9
        
        return WalletStatistics(
            address=address,
            address_type=address_type,
            creation_date=first_date.isoformat() if first_date else None,
            first_transaction_date=first_date.isoformat() if first_date else None,
            last_transaction_date=last_date.isoformat() if last_date else None,
            wallet_age_days=wallet_age_days,
            total_volume_usd_365d=round(total_volume_usd_365d, 2),
            average_volume_usd_365d=round(average_volume_usd_365d, 2),
            max_volume_usd_365d=round(max_volume_usd_365d, 2),
            median_volume_usd_365d=round(median_volume_usd_365d, 2),
            total_transactions_365d=len(transactions),
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
            data_sources=["etherscan"]
        )

    def analyze_subgraph_data(self, address: str, address_type: str, subgraph_data: Dict) -> WalletStatistics:
        """Analyze Rocket Pool Subgraph data"""
        if "error" in subgraph_data:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                analysis_errors=[f"Subgraph data error: {subgraph_data['error']}"]
            )
        
        transactions = subgraph_data.get("transactions", [])
        if not transactions:
            return self._create_empty_wallet_stats(address, address_type, ["subgraph"])
        
        volumes_usd = []
        timestamps = [int(tx["timestamp"]) for tx in transactions]
        
        daily_volumes = {}
        monthly_volumes = {}
        
        for tx in transactions:
            value_eth = float(tx["amount"]) / 10 ** 18  # wei to ETH
            value_usd = self.calculate_transaction_value_usd(value_eth, 'ETH', tx["timestamp"])
            volumes_usd.append(value_usd)
            
            # Daily and monthly activity
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
        
        # Calculate statistics
        total_volume_usd = sum(volumes_usd)
        average_volume_usd = total_volume_usd / len(volumes_usd)
        max_volume_usd = max(volumes_usd)
        median_volume_usd = statistics.median(volumes_usd)
        
        # Wallet age (from first RP transaction)
        first_date = datetime.fromtimestamp(first_timestamp, timezone.utc)
        last_date = datetime.fromtimestamp(last_timestamp, timezone.utc)
        wallet_age_days = (datetime.now(timezone.utc) - first_date).days
        
        # Activity patterns
        active_days = len(daily_volumes)
        avg_daily_volume = sum(daily_volumes.values()) / len(daily_volumes) if daily_volumes else 0.0
        max_daily_volume = max(daily_volumes.values()) if daily_volumes else 0.0
        most_active_month = max(monthly_volumes.items(), key=lambda x: x[1])[0] if monthly_volumes else None
        
        return WalletStatistics(
            address=address,
            address_type=address_type,
            creation_date=first_date.isoformat(),
            first_transaction_date=first_date.isoformat(),
            last_transaction_date=last_date.isoformat(),
            wallet_age_days=wallet_age_days,
            total_volume_usd_365d=round(total_volume_usd, 2),
            average_volume_usd_365d=round(average_volume_usd, 2),
            max_volume_usd_365d=round(max_volume_usd, 2),
            median_volume_usd_365d=round(median_volume_usd, 2),
            total_transactions_365d=len(transactions),
            outgoing_transactions_365d=len(transactions),  # Simplified - all are outgoing to RP
            incoming_transactions_365d=0,
            unique_addresses_interacted_365d=1,  # Only with Rocket Pool
            wallet_interactions_365d=0,  # Rocket Pool is a contract
            contract_interactions_365d=len(transactions),
            active_days_365d=active_days,
            avg_daily_volume_usd_365d=round(avg_daily_volume, 2),
            max_daily_volume_usd_365d=round(max_daily_volume, 2),
            most_active_month_365d=most_active_month,
            total_gas_used_365d=0,  # Not available in subgraph
            total_gas_fees_usd_365d=0.0,
            avg_gas_price_gwei_365d=0.0,
            data_sources=["subgraph"]
        )

    def _create_empty_wallet_stats(self, address: str, address_type: str, data_sources: List[str]) -> WalletStatistics:
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
            data_sources=data_sources
        )

    def analyze_single_address(self, address: str, raw_address_data: Dict) -> WalletStatistics:
        """Analyze a single address from raw data"""
        address_type = raw_address_data.get("address_type", "unknown")
        etherscan_data = raw_address_data.get("etherscan_data")
        subgraph_data = raw_address_data.get("subgraph_data")
        fetch_errors = raw_address_data.get("fetch_errors", [])
        
        # Priority: Etherscan data (more comprehensive) > Subgraph data
        result = None
        
        if etherscan_data and "error" not in etherscan_data and etherscan_data.get("transactions"):
            result = self.analyze_etherscan_data(address, address_type, etherscan_data)
        elif subgraph_data and "error" not in subgraph_data and subgraph_data.get("transactions"):
            result = self.analyze_subgraph_data(address, address_type, subgraph_data)
        
        if result is None:
            # No usable data found
            errors = fetch_errors.copy()
            if etherscan_data and "error" in etherscan_data:
                errors.append(f"Etherscan: {etherscan_data['error']}")
            if subgraph_data and "error" in subgraph_data:
                errors.append(f"Subgraph: {subgraph_data['error']}")
            
            result = WalletStatistics(
                address=address,
                address_type=address_type,
                analysis_errors=errors if errors else ["No usable data available"]
            )
        
        return result

    def analyze_all_addresses(self, raw_data: Dict[str, Any]) -> List[WalletStatistics]:
        """Analyze all addresses from raw data"""
        addresses_data = raw_data.get("addresses", {})
        results = []
        
        log.info(f"Analyzing {len(addresses_data)} addresses...")
        
        for address, raw_address_data in addresses_data.items():
            try:
                result = self.analyze_single_address(address, raw_address_data)
                results.append(result)
            except Exception as e:
                log.warning(f"Failed to analyze address {address}: {e}")
                error_result = WalletStatistics(
                    address=address,
                    address_type=raw_address_data.get("address_type", "unknown"),
                    analysis_errors=[f"Analysis failed: {str(e)}"]
                )
                results.append(error_result)
        
        # Log summary
        error_count = sum(1 for result in results if result.analysis_errors)
        success_count = len(results) - error_count
        
        wallet_count = sum(1 for result in results if result.address_type == "wallet" and not result.analysis_errors)
        contract_count = sum(1 for result in results if result.address_type == "contract" and not result.analysis_errors)
        
        log.info(f"Analysis completed: {success_count}/{len(results)} successful ({success_count/len(results)*100:.1f}%)")
        log.info(f"Address types - Wallets: {wallet_count}, Contracts: {contract_count}")
        
        return results

    def create_summary_report(self, results: List[WalletStatistics], raw_metadata: Dict, prefix: str, timestamp: str):
        """Generate comprehensive summary report"""
        successful_results = [r for r in results if not r.analysis_errors and r.total_transactions_365d and r.total_transactions_365d > 0]
        
        if not successful_results:
            log.warning("No successful results to create summary report")
            return
        
        # Separate by address type
        wallets = [r for r in successful_results if r.address_type == "wallet"]
        contracts = [r for r in successful_results if r.address_type == "contract"]
        
        # Calculate metrics
        def safe_stats(values: List[float]) -> Dict:
            if not values:
                return {"min": 0, "max": 0, "sum": 0, "mean": 0, "median": 0}
            return {
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "sum": round(sum(values), 2),
                "mean": round(statistics.mean(values), 2),
                "median": round(statistics.median(values), 2)
            }
        
        # Volume metrics
        total_volumes = [r.total_volume_usd_365d for r in successful_results if r.total_volume_usd_365d]
        avg_volumes = [r.average_volume_usd_365d for r in successful_results if r.average_volume_usd_365d]
        max_volumes = [r.max_volume_usd_365d for r in successful_results if r.max_volume_usd_365d]
        
        # Transaction metrics
        tx_counts = [r.total_transactions_365d for r in successful_results if r.total_transactions_365d]
        
        # Age metrics
        wallet_ages = [r.wallet_age_days for r in successful_results if r.wallet_age_days]
        
        # Interaction metrics
        unique_interactions = [r.unique_addresses_interacted_365d for r in successful_results if r.unique_addresses_interacted_365d]
        wallet_interactions = [r.wallet_interactions_365d for r in successful_results if r.wallet_interactions_365d]
        contract_interactions = [r.contract_interactions_365d for r in successful_results if r.contract_interactions_365d]
        
        # Activity metrics
        active_days = [r.active_days_365d for r in successful_results if r.active_days_365d]
        gas_fees = [r.total_gas_fees_usd_365d for r in successful_results if r.total_gas_fees_usd_365d]
        
        summary = {
            "analysis_timestamp": timestamp,
            "source_metadata": raw_metadata,
            "total_addresses_analyzed": len(results),
            "successful_analyses": len(successful_results),
            "success_rate_percent": round(len(successful_results) / len(results) * 100, 2),
            "address_types": {
                "wallets": len(wallets),
                "contracts": len(contracts),
                "unknown": len(successful_results) - len(wallets) - len(contracts)
            },
            "volume_metrics_365d": {
                "total_volume_usd": safe_stats(total_volumes),
                "average_volume_usd": safe_stats(avg_volumes),
                "max_volume_usd": safe_stats(max_volumes)
            },
            "transaction_metrics_365d": {
                "total_transactions": safe_stats(tx_counts)
            },
            "wallet_age_metrics": safe_stats(wallet_ages),
            "interaction_metrics_365d": {
                "unique_addresses_interacted": safe_stats(unique_interactions),
                "wallet_interactions_total": sum(wallet_interactions) if wallet_interactions else 0,
                "contract_interactions_total": sum(contract_interactions) if contract_interactions else 0
            },
            "activity_metrics_365d": {
                "active_days": safe_stats(active_days),
                "total_gas_fees_usd": safe_stats(gas_fees)
            }
        }
        
        # Save summary
        summary_file = self.output_dir / f"{prefix}_{timestamp}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        log.info(f"Saved summary report to {summary_file}")
        
        # Log key metrics
        log.info("=== ANALYSIS SUMMARY ===")
        log.info(f"Analyzed {len(results)} addresses")
        log.info(f"Success rate: {len(successful_results)}/{len(results)} ({summary['success_rate_percent']}%)")
        log.info(f"Address types: {len(wallets)} wallets, {len(contracts)} contracts")
        if total_volumes:
            log.info(f"Total volume (365d): ${summary['volume_metrics_365d']['total_volume_usd']['sum']:,.2f}")
            log.info(f"Average wallet age: {summary['wallet_age_metrics']['mean']:.0f} days")
            log.info(f"Total transactions (365d): {summary['transaction_metrics_365d']['total_transactions']['sum']:,}")

    def save_results(self, results: List[WalletStatistics], raw_metadata: Dict, prefix: str = "wallet_analysis") -> tuple[str, str]:
        """Save analysis results to JSON and CSV"""
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
        
        # Generate summary report
        self.create_summary_report(results, raw_metadata, prefix, timestamp)
        
        return str(json_file), str(csv_file)

    def run_analysis(self, raw_data_file: str) -> tuple[str, str]:
        """Run complete analysis on raw data file"""
        log.info("=" * 60)
        log.info("WALLET STATISTICS ANALYSIS STARTED")
        log.info("=" * 60)
        
        try:
            # Load raw data
            raw_data = self.load_raw_data(raw_data_file)
            
            # Analyze all addresses
            results = self.analyze_all_addresses(raw_data)
            
            # Save results
            json_file, csv_file = self.save_results(results, raw_data.get('metadata', {}))
            
            log.info("=" * 60)
            log.info("ANALYSIS COMPLETED SUCCESSFULLY")
            log.info(f"Results saved to:")
            log.info(f"  JSON: {json_file}")
            log.info(f"  CSV: {csv_file}")
            log.info("=" * 60)
            
            return json_file, csv_file
            
        except Exception as e:
            log.error(f"Analysis failed: {e}")
            raise


def main():
    """Main function with hardcoded parameters"""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent  # scripts/stats_vis/ -> mintegrity/
    
    # Hardcoded configuration
    output_dir = str(project_root / "files" / "wallet_analysis")
    
    # You need to specify the raw data file path here
    # This should be the output from rocketpool_data_fetcher.py
    # Example: raw_data_file = str(project_root / "files" / "rocket_pool_data" / "rocket_pool_raw_data_20241215_143022.json")
    
    # Find the most recent raw data file automatically
    raw_data_dir = project_root / "files" / "rocket_pool_data"
    if not raw_data_dir.exists():
        print(f"Error: Raw data directory not found: {raw_data_dir}")
        print("Please run rocketpool_data_fetcher.py first to generate raw data")
        return 1
    
    # Find most recent raw data file
    raw_data_files = list(raw_data_dir.glob("rocket_pool_raw_data_*.json"))
    if not raw_data_files:
        print(f"Error: No raw data files found in {raw_data_dir}")
        print("Please run rocketpool_data_fetcher.py first to generate raw data")
        return 1
    
    # Use the most recent file
    raw_data_file = str(max(raw_data_files, key=lambda x: x.stat().st_mtime))
    print(f"Using raw data file: {raw_data_file}")
    
    try:
        analyzer = WalletStatisticsAnalyzer(output_dir=output_dir)
        json_file, csv_file = analyzer.run_analysis(raw_data_file)
        
        print(f"\nAnalysis completed successfully!")
        print(f"JSON results: {json_file}")
        print(f"CSV results: {csv_file}")
        
    except KeyboardInterrupt:
        log.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
