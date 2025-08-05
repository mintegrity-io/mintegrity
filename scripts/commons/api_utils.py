#!/usr/bin/env python3
"""
API Utils
Extracted from RocketPoolGroupsAnalyzer for reuse across all protocol analyzers

Location: scripts/commons/api_utils.py
"""

import os

def check_api_capabilities():
    """
    Check API capabilities and show available analysis options
    
    Checks for FULL_ANALYSIS_AVAILABLE flag and ETHERSCAN_API_KEY environment variable.
    Prints status information about available analysis capabilities.
    
    Returns:
        tuple: (full_analysis_available: bool, etherscan_api_key: str or None)
    """
    try:
        from scripts.stats_vis.wallet_groups_analyzer import FULL_ANALYSIS_AVAILABLE
    except ImportError:
        print("⚠️  Could not import FULL_ANALYSIS_AVAILABLE")
        FULL_ANALYSIS_AVAILABLE = False
    
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
    
    return FULL_ANALYSIS_AVAILABLE, etherscan_api_key

def is_full_analysis_available():
    """
    Check if full analysis is available
    
    Returns:
        bool: True if full analysis capabilities are available
    """
    try:
        from scripts.stats_vis.wallet_groups_analyzer import FULL_ANALYSIS_AVAILABLE
        return FULL_ANALYSIS_AVAILABLE
    except ImportError:
        return False

def get_etherscan_api_key():
    """
    Get Etherscan API key from environment
    
    Returns:
        str or None: API key if found, None otherwise
    """
    return os.getenv("ETHERSCAN_API_KEY")

def mask_api_key(api_key):
    """
    Mask API key for safe logging
    
    Args:
        api_key (str): API key to mask
        
    Returns:
        str: Masked API key
    """
    if not api_key:
        return "Not set"
    
    if len(api_key) > 12:
        return api_key[:8] + "..." + api_key[-4:]
    else:
        return "***"
