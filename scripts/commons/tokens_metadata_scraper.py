import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from requests import RequestException

from scripts.commons.logging_config import get_logger

log = get_logger()

# Load environment variables from .env file
load_dotenv()

# Get the Alchemy API key from the environment variable
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")


def fetch_current_token_prices(token_symbols: list[str]) -> dict[str, tuple[int, float]]:
    """
    Fetch token prices from Alchemy API in batches

    Args:
        token_symbols: List of token symbols (e.g., ['ETH', 'BTC'])

    Returns:
        Dictionary mapping token symbol to (timestamp, price) tuple
        Price will be 0 for unknown tokens
    """
    results = {}
    batch_size = 25  # Alchemy API limit

    # Process tokens in batches of 25
    for i in range(0, len(token_symbols), batch_size):
        batch = token_symbols[i:i + batch_size]
        url = f"https://api.g.alchemy.com/prices/v1/{ALCHEMY_API_KEY}/tokens/by-symbol"

        params = {
            "symbols": batch
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()["data"]

            # Create a dictionary to easily look up token data
            token_data_map = {item["symbol"]: item for item in data if "symbol" in item}

            # Process each token in the batch
            for token_symbol in batch:
                if token_symbol in token_data_map and "prices" in token_data_map[token_symbol]:
                    token_data = token_data_map[token_symbol]
                    for price_entry in token_data["prices"]:
                        if price_entry.get("currency") == "usd" and "value" in price_entry and "lastUpdatedAt" in price_entry:
                            iso_timestamp = price_entry["lastUpdatedAt"]
                            timestamp = int(datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00')).timestamp())
                            current_price = float(price_entry["value"])
                            results[token_symbol] = (timestamp, current_price)
                            break
                    else:
                        # No USD price found
                        log.error(f"No USD price found for token {token_symbol}")
                        results[token_symbol] = (int(datetime.now().timestamp()), 0.0)
                else:
                    # Token not found in response
                    log.error(f"Unknown token or missing price data: {token_symbol}")
                    results[token_symbol] = (int(datetime.now().timestamp()), 0.0)
        else:
            raise RequestException(f"Error fetching price from Alchemy for tokens {batch}: {response.json()}")

    return results
