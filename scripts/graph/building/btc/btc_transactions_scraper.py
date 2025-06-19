import os
import time
import json
import requests
from typing import Dict, Optional, List, Tuple, Set
from dotenv import load_dotenv
from dataclasses import dataclass

from scripts.commons.model import *

log = get_logger()

# Load environment variables from .env file
load_dotenv()

# Get the Alchemy API key from the environment variable
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

if not ALCHEMY_API_KEY:
    raise ValueError("ALCHEMY_API_KEY is not set in the environment variables.")

# Constants for Bitcoin networks
BTC_MAINNET = "btc-mainnet"
BTC_TESTNET = "btc-testnet"

# Define BTC token symbol
BTC_TOKEN_SYMBOL = "BTC"

# Cache for address information
_address_info_cache = {}

# Path to the persistent cache file
_CACHE_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "btc_address_cache.json")

def _load_address_cache():
    """Load address cache from file if it exists"""
    if os.path.exists(_CACHE_FILE_PATH):
        try:
            with open(_CACHE_FILE_PATH, 'r') as f:
                global _address_info_cache
                _address_info_cache = json.load(f)
            log.info(f"Loaded {len(_address_info_cache)} BTC addresses from cache file")
        except Exception as e:
            log.warning(f"Failed to load BTC address cache: {e}")

def _save_address_cache():
    """Save address cache to file"""
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE_PATH), exist_ok=True)
        with open(_CACHE_FILE_PATH, 'w') as f:
            json.dump(_address_info_cache, f)
        log.info(f"Saved {len(_address_info_cache)} BTC addresses to cache file")
    except Exception as e:
        log.warning(f"Failed to save BTC address cache: {e}")

# Load the cache at module initialization
_load_address_cache()

def print_timestamp(timestamp: int) -> str:
    """
    Converts a Unix timestamp to a human-readable format.

    :param timestamp: Unix timestamp in seconds
    :return: Unix timestamp + Human-readable date string
    """
    return f"{timestamp} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(timestamp))})"

def convert_btc_to_eth_address_format(btc_address: str) -> str:
    """
    Converts a BTC address to a pseudo-ETH address format for compatibility with the model.
    Prefixes with 0x and pads to ensure it's the right length.

    :param btc_address: Bitcoin address
    :return: Ethereum-like address format
    """
    # Simple approach: prefix with 0x and pad/truncate to 42 chars
    pseudo_eth = "0x" + btc_address.rjust(40, '0')
    if len(pseudo_eth) > 42:
        pseudo_eth = pseudo_eth[:42]
    return pseudo_eth.lower()

def get_address_type(address: str, network: str = BTC_MAINNET) -> AddressType:
    """
    For Bitcoin, all addresses are considered wallets as there's no concept of smart contracts.
    This function is included for API compatibility with the ETH scraper.

    :param address: Bitcoin address
    :param network: Bitcoin network
    :return: Always AddressType.WALLET for Bitcoin
    """
    return AddressType.WALLET

def satoshi_to_btc(satoshi_value: int) -> float:
    """
    Convert satoshi value to BTC

    :param satoshi_value: Value in satoshi
    :return: Value in BTC
    """
    return satoshi_value / 100000000.0

def get_block_by_timestamp(timestamp: int, network: str = BTC_MAINNET) -> str:
    """
    Gets Bitcoin block height by timestamp using Alchemy's utility endpoint.

    :param timestamp: Unix timestamp in seconds
    :param network: The Bitcoin network
    :return: Block height as a string
    """
    url = f"https://api.g.alchemy.com/data/v1/{ALCHEMY_API_KEY}/utility/blocks/by-timestamp"
    log.info(f"Fetching block number for timestamp: {print_timestamp(timestamp)} on network: {network}")

    params = {
        "timestamp": str(timestamp),
        "networks": [network],
        "direction": "AFTER"
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if "block" in data["data"][0]:
            block_info = data["data"][0]["block"]
            height = block_info["number"]
            timestamp_from_block = block_info["timestamp"]
            log.debug(f"Block height: {height}, Timestamp: {timestamp_from_block}")
            return str(height)
        else:
            log.error(f"Error in block-by-timestamp response: {data}")
            raise Exception(f"Error in block-by-timestamp response: {data}")
    else:
        log.error(f"Error getting block by timestamp: {response.status_code}, {response.text}")
        raise Exception(f"Error: {response.status_code}, {response.text}")

def get_address_transactions(address: str, from_block: str, to_block: str, network: str = BTC_MAINNET) -> dict:
    """
    Fetches transactions for a Bitcoin address between two block heights.

    :param address: Bitcoin address
    :param from_block: Start block height
    :param to_block: End block height
    :param network: Bitcoin network
    :return: Dictionary containing transaction data
    """
    url = f"https://{network}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "alchemy_getAssetTransfers",
        "params": [{
            "fromBlock": from_block,
            "toBlock": to_block,
            "toAddress": address,
            "withMetadata": True,
            "excludeZeroValue": False,
            "category": ["external"]
        }]
    }

    incoming_txs = []
    page_key = None

    # Fetch incoming transactions with pagination
    while True:
        if page_key:
            payload["params"][0]["pageKey"] = page_key

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                incoming_txs.extend(data["result"].get("transfers", []))
                page_key = data["result"].get("pageKey")
                if not page_key:
                    break
            elif "error" in data:
                log.error(f"Error from Alchemy API: {data['error']}")
                break
        else:
            log.error(f"Request failed with status code {response.status_code}: {response.text}")
            break

    # Now fetch outgoing transactions
    payload["params"][0].pop("toAddress", None)
    payload["params"][0]["fromAddress"] = address

    outgoing_txs = []
    page_key = None

    while True:
        if page_key:
            payload["params"][0]["pageKey"] = page_key
        else:
            payload["params"][0].pop("pageKey", None)

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                outgoing_txs.extend(data["result"].get("transfers", []))
                page_key = data["result"].get("pageKey")
                if not page_key:
                    break
            elif "error" in data:
                log.error(f"Error from Alchemy API: {data['error']}")
                break
        else:
            log.error(f"Request failed with status code {response.status_code}: {response.text}")
            break

    return {
        "incoming": incoming_txs,
        "outgoing": outgoing_txs
    }

def get_address_interactions(
        target_address: Address,
        from_timestamp: int,
        to_timestamp: int,
        network: str = BTC_MAINNET,
        limit: int = 1000
) -> set[tuple[InteractionDirection, Transaction]]:
    """
    Fetches unique addresses that interacted with a Bitcoin address during a time period.

    :param target_address: The address to get interactions for incoming and outgoing transactions
    :param from_timestamp: Start timestamp in Unix seconds
    :param to_timestamp: End timestamp in Unix seconds
    :param network: The Bitcoin network (e.g., 'btc-mainnet', 'btc-testnet')
    :param limit: Maximum number of transactions to return
    :return: A set of tuples containing the interaction direction and transaction details
    """
    log.info(f"Fetching interactions for address: {target_address.address} from {print_timestamp(from_timestamp)} to {print_timestamp(to_timestamp)} on network: {network}")

    # Convert timestamps to block heights
    from_block = get_block_by_timestamp(from_timestamp, network)
    to_block = get_block_by_timestamp(to_timestamp, network)

    log.info(f"Converted timestamps to blocks: {from_timestamp} -> {from_block}, {to_timestamp} -> {to_block}")

    # Get all transactions for this address
    transactions_data = get_address_transactions(target_address.address, from_block, to_block, network)

    interactions: set[tuple[InteractionDirection, Transaction]] = set()
    tx_count = 0

    # Process incoming transactions
    for tx in transactions_data["incoming"]:
        if tx_count >= limit:
            break

        # Skip transactions without required data
        if not all(key in tx for key in ["hash", "from", "to", "value", "metadata"]):
            continue

        if tx["to"] != target_address.address:
            continue

        # Create an ETH-compatible address for the model
        from_addr_eth = convert_btc_to_eth_address_format(tx["from"])
        to_addr_eth = convert_btc_to_eth_address_format(tx["to"])

        # Create transaction object
        transaction = Transaction(
            transaction_hash=tx["hash"],
            address_from=Address(from_addr_eth, AddressType.WALLET),
            address_to=Address(to_addr_eth, AddressType.WALLET),
            value=satoshi_to_btc(int(float(tx["value"]) * 100000000)),  # Convert to BTC
            timestamp=tx["metadata"]["blockTimestamp"],
            token_symbol=BTC_TOKEN_SYMBOL
        )

        interactions.add((InteractionDirection.INCOMING, transaction))
        tx_count += 1

    # Process outgoing transactions
    for tx in transactions_data["outgoing"]:
        if tx_count >= limit:
            break

        # Skip transactions without required data
        if not all(key in tx for key in ["hash", "from", "to", "value", "metadata"]):
            continue

        if tx["from"] != target_address.address:
            continue

        # Create an ETH-compatible address for the model
        from_addr_eth = convert_btc_to_eth_address_format(tx["from"])
        to_addr_eth = convert_btc_to_eth_address_format(tx["to"])

        # Create transaction object
        transaction = Transaction(
            transaction_hash=tx["hash"],
            address_from=Address(from_addr_eth, AddressType.WALLET),
            address_to=Address(to_addr_eth, AddressType.WALLET),
            value=satoshi_to_btc(int(float(tx["value"]) * 100000000)),  # Convert to BTC
            timestamp=tx["metadata"]["blockTimestamp"],
            token_symbol=BTC_TOKEN_SYMBOL
        )

        interactions.add((InteractionDirection.OUTGOING, transaction))
        tx_count += 1

    log.info(f"Found {len(interactions)} unique transactions for address {target_address.address} between {from_timestamp} and {to_timestamp}")
    return interactions
