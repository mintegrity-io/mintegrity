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


def fetch_current_token_price(token_symbol: str) -> (int, float):
    """
    Fetch token price from Alchemy API

    Args:
        token_symbol: Token symbol (e.g., 'ETH', 'BTC')

    Returns:
        int timestamp
        float price_usd
    """
    # Convert datetime to timestamp if needed
    # TODO put in one api call, Alchemy has cap of 300 request per hours
    url = f"https://api.g.alchemy.com/prices/v1/{ALCHEMY_API_KEY}/tokens/by-symbol"

    params = {
        "symbols": [token_symbol]
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()["data"]

        # Check if we have price data for the token
        for token_data in data:
            if token_data.get("symbol") == token_symbol and "prices" in token_data:
                for price_entry in token_data["prices"]:
                    if price_entry.get("currency") == "usd" and "value" in price_entry and "lastUpdatedAt" in price_entry:
                        # Convert ISO timestamp to Unix timestamp
                        iso_timestamp = price_entry["lastUpdatedAt"]
                        timestamp = int(datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00')).timestamp())
                        current_price = float(price_entry["value"])
                        return timestamp, current_price

        # If we reach here, it means we didn't find the expected data
        raise RequestException(f"Error fetching price from Alchemy for token {token_symbol}: expected parameters not found: {response.json()}")
    else:
        raise RequestException(f"Error fetching price from Alchemy  for token {token_symbol}: {response.json()}")

