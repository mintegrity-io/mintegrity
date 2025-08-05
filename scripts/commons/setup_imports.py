#!/usr/bin/env python3
"""
Setup Imports Utility
Extracted from RocketPoolGroupsAnalyzer for reuse across all protocol analyzers

Location: scripts/commons/setup_imports.py
"""

import sys
from pathlib import Path

def setup_project_imports():
    """
    Setup project imports - reusable for all protocols
    
    Adds project root to Python path and imports required modules with error handling.
    
    Returns:
        bool: True if all imports successful, False otherwise
    """
    # Add project root to path for imports
    # Since this file is in scripts/commons/, project root is 2 levels up
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
        return True
        
    except ImportError as e:
        print(f"❌ Could not import required modules: {e}")
        print("Make sure you are running from the mintegrity directory and all modules exist")
        return False

def get_project_root():
    """
    Get project root directory
    
    Returns:
        Path: Path to project root directory
    """
    return Path(__file__).parent.parent.parent
