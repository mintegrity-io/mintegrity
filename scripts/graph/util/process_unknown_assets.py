#!/usr/bin/env python3
"""
Script to parse logs.txt and count occurrences of unknown assets that were found in transactions
but not included in the whitelist.

This script helps identify popular assets that should potentially be added to the whitelist.
"""

import re
import os
from collections import Counter
from typing import Dict, List, Tuple


def count_unknown_assets(log_file_path: str) -> Dict[str, int]:
    """
    Parse logs.txt file and count occurrences of assets mentioned in warnings about unknown assets.

    Args:
        log_file_path: Path to the logs.txt file

    Returns:
        Dictionary mapping asset symbols to their occurrence counts
    """
    # Regex pattern to extract asset names from warning logs
    pattern = r"Transfer with unknown asset ([A-Za-z0-9]+) found, but this asset is not in the whitelist"

    asset_counts = Counter()

    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    asset = match.group(1)
                    asset_counts[asset] += 1
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return {}
    except Exception as e:
        print(f"Error processing log file: {str(e)}")
        return {}

    return dict(asset_counts)


def get_top_assets(asset_counts: Dict[str, int], limit: int = None) -> List[Tuple[str, int]]:
    """
    Get assets sorted by occurrence count in descending order.

    Args:
        asset_counts: Dictionary mapping asset symbols to their occurrence counts
        limit: Maximum number of assets to return (None for all)

    Returns:
        List of (asset_symbol, count) tuples sorted by count in descending order
    """
    sorted_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)
    if limit:
        return sorted_assets[:limit]
    return sorted_assets


if __name__ == "__main__":
    # Path to the logs file
    logs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs.txt")

    # Count occurrences of unknown assets
    asset_counts = count_unknown_assets(logs_path)

    if not asset_counts:
        print("No unknown assets found in the logs file.")
    else:
        # Get assets sorted by occurrence (most frequent first)
        sorted_assets = get_top_assets(asset_counts)

        print(f"Found {len(sorted_assets)} unique unknown assets in the logs file.")
        print("\nTop unknown assets by occurrence:")
        print("-" * 40)
        print(f"{'Asset':<10} | {'Occurrences':<10}")
        print("-" * 40)

        for asset, count in sorted_assets:
            print(f"{asset:<10} | {count:<10}")

        print("\nConsider adding frequently occurring assets to ETH_TOKENS_WHITELIST in known_token_list.py")
