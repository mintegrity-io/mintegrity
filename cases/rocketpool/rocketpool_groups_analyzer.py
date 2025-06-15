#!/usr/bin/env python3
"""
Rocket Pool Groups Analyzer
Rocket Pool specific implementation for analyzing coordinated groups of addresses.

EXECUTION:
cd mintegrity
python cases/rocketpool/rocketpool_groups_analyzer.py

Functionality:
1. Loads existing Rocket Pool graph
2. Identifies groups of coordinated addresses using Rocket Pool specific algorithms
3. Uses general wallet groups analyzer for statistics and visualizations
4. Creates detailed reports and saves results

REQUIREMENTS:
• ETHERSCAN_API_KEY in .env file for full analysis
• Internet connection for API requests
• files/rocket_pool_full_graph_90_days.json
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import Rocket Pool specific modules
    from scripts.graph.util.transactions_graph_json import load_graph_from_json
    from scripts.graph.analysis.wallet_groups.wallet_grouping import (
        detect_wallet_coordination,
        identify_wallet_groups
    )
    
    # Import general wallet groups analyzer
    from scripts.stats_vis.wallet_groups_analyzer import WalletGroupsAnalyzer, FULL_ANALYSIS_AVAILABLE
    
    print("✅ Successfully imported all required modules")
except ImportError as e:
    print(f"❌ Could not import required modules: {e}")
    print("Make sure you are running from the mintegrity directory and all modules exist")
    sys.exit(1)


class RocketPoolGroupsAnalyzer:
    """Rocket Pool specific groups analyzer"""
    
    # Hardcoded configuration - no CLI
    GRAPH_FILE_PATH = "files/rocket_pool_full_graph_90_days.json"
    OUTPUT_DIR = "files/rocket_pool_groups_analysis"
    COORDINATION_THRESHOLD = 5.0
    MIN_GROUP_SIZE = 2
    MAX_WORKERS = 5
    ADDRESSES_FILE_PATH = None  # Optional: path to existing addresses analysis
    
    def __init__(self):
        print("🚀 Initializing Rocket Pool Groups Analyzer")
        print(f"Graph file: {self.GRAPH_FILE_PATH}")
        print(f"Output directory: {self.OUTPUT_DIR}")
        print(f"Coordination threshold: {self.COORDINATION_THRESHOLD}")
        print(f"Minimum group size: {self.MIN_GROUP_SIZE}")
        print(f"Max workers: {self.MAX_WORKERS}")
        
        # Initialize general analyzer
        self.analyzer = WalletGroupsAnalyzer(
            output_dir=self.OUTPUT_DIR,
            max_workers=self.MAX_WORKERS
        )
        
        self.graph = None
        self.wallet_groups = []
        
        # Show available analysis capabilities
        if FULL_ANALYSIS_AVAILABLE:
            print("🚀 Full API analysis available (365-day detailed statistics via Etherscan + historical prices)")
        else:
            print("📊 Basic analysis available (graph-based statistics only)")
            
        # Check API key
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if etherscan_api_key:
            masked_key = etherscan_api_key[:8] + "..." + etherscan_api_key[-4:] if len(etherscan_api_key) > 12 else "***"
            print(f"✅ ETHERSCAN_API_KEY found: {masked_key}")
        else:
            print("⚠️  ETHERSCAN_API_KEY not set")
            print("   Will use basic graph-based analysis instead")
            print("   Add ETHERSCAN_API_KEY=your_key to .env file for full functionality")
    
    def load_rocket_pool_graph(self):
        """Loads Rocket Pool graph from file"""
        graph_path = Path(self.GRAPH_FILE_PATH)
        
        if not graph_path.exists():
            print(f"❌ Graph file not found: {graph_path}")
            print("Please ensure the Rocket Pool graph file exists")
            sys.exit(1)
        
        print(f"📂 Loading Rocket Pool graph from {graph_path}")
        self.graph = load_graph_from_json(str(graph_path))
        print(f"✅ Successfully loaded graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
    
    def detect_rocket_pool_wallet_groups(self):
        """Identifies coordinated address groups using Rocket Pool specific algorithms"""
        print("🔍 Detecting coordinated wallet groups in Rocket Pool data...")
        
        # Use Rocket Pool specific coordination detection
        coordination_scores = detect_wallet_coordination(self.graph)
        
        # Import wallet_metrics from module
        from scripts.graph.analysis.wallet_groups.wallet_grouping import wallet_metrics
        
        self.wallet_groups = identify_wallet_groups(
            coordination_scores,
            wallet_metrics,
            threshold=self.COORDINATION_THRESHOLD
        )
        
        # Filter groups by minimum size
        self.wallet_groups = [group for group in self.wallet_groups if len(group) >= self.MIN_GROUP_SIZE]
        
        print(f"✅ Found {len(self.wallet_groups)} groups with {self.MIN_GROUP_SIZE}+ addresses")
        for i, group in enumerate(self.wallet_groups):
            print(f"   Group {i + 1}: {len(group)} addresses")
    
    def load_or_analyze_individual_addresses(self):
        """Loads or analyzes individual address statistics"""
        
        # Check if there's an existing addresses file
        if self.ADDRESSES_FILE_PATH and Path(self.ADDRESSES_FILE_PATH).exists():
            print(f"📂 Loading existing addresses analysis from {self.ADDRESSES_FILE_PATH}")
            
            import json
            from scripts.stats_vis.wallet_groups_analyzer import WalletStatistics
            
            with open(self.ADDRESSES_FILE_PATH, 'r') as f:
                addresses_data = json.load(f)
            
            # Convert to analyzer's individual_stats format
            for addr_data in addresses_data:
                if not addr_data.get('error'):
                    self.analyzer.individual_stats[addr_data['address']] = WalletStatistics(**addr_data)
            
            print(f"✅ Loaded statistics for {len(self.analyzer.individual_stats)} addresses")
            return
        
        # Collect all addresses from groups
        all_group_addresses = set()
        for group in self.wallet_groups:
            all_group_addresses.update(group)
        
        if not all_group_addresses:
            print("⚠️  No addresses found in groups")
            return
        
        # Analyze addresses
        if FULL_ANALYSIS_AVAILABLE:
            print("🚀 Performing full 365-day analysis via APIs...")
            self.analyzer.analyze_addresses_with_full_stats(list(all_group_addresses), self.graph)
        else:
            print("📊 Creating simplified statistics from graph data...")
            self.analyzer.create_mock_statistics_from_graph(list(all_group_addresses), self.graph)
    
    def run_full_analysis(self):
        """Runs complete Rocket Pool groups analysis"""
        print("=" * 60)
        print("🚀 ROCKET POOL GROUPS ANALYSIS STARTED")
        print("=" * 60)
        
        try:
            # 1. Load Rocket Pool graph
            self.load_rocket_pool_graph()
            
            # 2. Detect Rocket Pool specific wallet groups
            self.detect_rocket_pool_wallet_groups()
            
            if not self.wallet_groups:
                print("❌ No wallet groups found")
                return
            
            # 3. Load or analyze individual address statistics
            self.load_or_analyze_individual_addresses()
            
            # 4. Calculate group statistics using general analyzer
            self.analyzer.calculate_group_statistics(self.wallet_groups, self.graph)
            
            if not self.analyzer.group_stats:
                print("❌ No group statistics calculated")
                return
            
            # 5. Create visualizations using general analyzer
            print("📊 Creating visualizations...")
            self.analyzer.create_transaction_volume_bins_for_all_groups()
            self.analyzer.create_group_volume_distribution()
            self.analyzer.create_group_size_analysis()
            self.analyzer.create_top_groups_analysis()
            self.analyzer.create_groups_vs_individuals_comparison(self.wallet_groups)
            
            # 6. Save data using general analyzer
            json_file, csv_file = self.analyzer.save_groups_data()
            
            # 7. Generate report using general analyzer
            stats = self.analyzer.generate_groups_report()
            
            print("=" * 60)
            print("✅ ROCKET POOL GROUPS ANALYSIS COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"📊 {stats['total_groups']} groups analyzed")
            print(f"👥 {stats['total_addresses_in_groups']} addresses in groups")
            print(f"💰 ${stats['total_volume']:,.0f} total volume")
            print(f"🔄 {stats['total_transactions']:,.0f} total transactions")
            print(f"📁 Charts saved to: {self.analyzer.output_dir}/plots/")
            print(f"📁 Data saved to: {json_file} and {csv_file}")
            print("📈 Generated 5 PNG charts with comprehensive analysis")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Rocket Pool groups analysis failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main function"""
    print("🚀 Starting Rocket Pool Groups Analysis")
    print(f"Working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}")
    
    try:
        analyzer = RocketPoolGroupsAnalyzer()
        analyzer.run_full_analysis()
        return 0
        
    except KeyboardInterrupt:
        print("⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("If you're getting import errors, make sure you're running from the mintegrity project root")
        return 1


if __name__ == "__main__":
    exit(main())
