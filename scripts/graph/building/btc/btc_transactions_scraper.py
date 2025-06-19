import os
import time
import json
import requests
from typing import Dict, Optional, List, Tuple, Set
from dotenv import load_dotenv
from dataclasses import dataclass
import datetime

from scripts.commons.model import *

log = get_logger()

# Load environment variables from .env file
load_dotenv()

# Constants for Bitcoin networks
BTC_MAINNET = "mainnet"
BTC_TESTNET = "testnet"

# Base URLs for mempool.space API
MEMPOOL_API_BASE_URL = "https://mempool.space/api"
MEMPOOL_TESTNET_API_BASE_URL = "https://mempool.space/testnet/api"

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

def get_api_base_url(network: str = BTC_MAINNET) -> str:
    """
    Returns the base URL for the mempool.space API based on the network

    :param network: Bitcoin network (mainnet or testnet)
    :return: Base URL for the API
    """
    if network == BTC_TESTNET:
        return MEMPOOL_TESTNET_API_BASE_URL
    return MEMPOOL_API_BASE_URL

def get_block_by_timestamp(timestamp: int, network: str = BTC_MAINNET) -> str:
    """
    Gets Bitcoin block height by timestamp using mempool.space's API.

    :param timestamp: Unix timestamp in seconds
    :param network: The Bitcoin network
    :return: Block height as a string
    """
    base_url = get_api_base_url(network)

    # Get blocks within a reasonable range around the timestamp
    # First, fetch the current block height
    try:
        blocks_url = f"{base_url}/blocks/tip/height"
        response = requests.get(blocks_url)
        response.raise_for_status()
        current_height = int(response.text)

        # Estimate the block height for the timestamp (Bitcoin averages ~10 min per block)
        # Calculate how many blocks back we need to go (rough estimate)
        current_time = int(time.time())
        time_diff = current_time - timestamp
        blocks_back = time_diff // 600  # 600 seconds = 10 minutes

        if blocks_back > current_height:
            blocks_back = current_height - 1

        target_height = max(0, current_height - blocks_back)

        # Now we'll search for the exact block by timestamp
        found_block = None
        search_range = 1000  # We'll look at a range of blocks

        # Start at our estimated height and search backward
        start_height = min(current_height, target_height + search_range // 2)
        end_height = max(0, start_height - search_range)

        log.info(f"Searching for block with timestamp near {print_timestamp(timestamp)} in range {end_height} to {start_height}")

        # Get blocks in batches to find the right one
        batch_size = 10
        for height in range(start_height, end_height, -batch_size):
            blocks_url = f"{base_url}/v1/blocks/{height}"
            response = requests.get(blocks_url)
            response.raise_for_status()
            blocks = response.json()

            for block in blocks:
                block_time = block['timestamp']
                if block_time >= timestamp:
                    found_block = block
                elif found_block is not None:
                    # We've already found a block after the timestamp, and now we're at blocks before
                    # Return the block height we found
                    log.debug(f"Found block height: {found_block['height']} with timestamp: {found_block['timestamp']}")
                    return str(found_block['height'])

        # If we looked through all blocks and didn't find a transition, use the last one we checked
        if found_block is not None:
            log.debug(f"Found block height: {found_block['height']} with timestamp: {found_block['timestamp']}")
            return str(found_block['height'])

        # If we couldn't find a suitable block, return the estimated height
        log.warning(f"Could not find exact block for timestamp {print_timestamp(timestamp)}, using estimate: {target_height}")
        return str(target_height)

    except Exception as e:
        log.error(f"Error finding block by timestamp: {e}")
        raise Exception(f"Error finding block by timestamp: {e}")

def get_address_transactions(address: str, from_block: str, to_block: str, network: str = BTC_MAINNET) -> dict:
    """
    Fetches transactions for a Bitcoin address between two block heights using mempool.space API.

    :param address: Bitcoin address
    :param from_block: Start block height
    :param to_block: End block height
    :param network: Bitcoin network
    :return: Dictionary containing transaction data
    """
    base_url = get_api_base_url(network)
    from_block_int = int(from_block)
    to_block_int = int(to_block)

    log.info(f"Fetching transactions for address {address} from block {from_block_int} to {to_block_int}")

    # Initialize results
    incoming_txs = []
    outgoing_txs = []

    try:
        # Fetch transactions for the address
        # First, get transaction history
        txs_url = f"{base_url}/address/{address}/txs"
        response = requests.get(txs_url)
        response.raise_for_status()

        transactions = response.json()
        log.info(f"Found {len(transactions)} total transactions for address {address}")

        # Process each transaction
        for tx in transactions:
            # Get detailed transaction info
            tx_url = f"{base_url}/tx/{tx['txid']}"
            tx_response = requests.get(tx_url)
            tx_response.raise_for_status()
            tx_details = tx_response.json()

            # Check if transaction is within our block height range
            if 'status' in tx_details and 'block_height' in tx_details['status']:
                block_height = tx_details['status']['block_height']
                if block_height < from_block_int or block_height > to_block_int:
                    continue

                # Get the block timestamp as a string
                block_timestamp = str(tx_details['status'].get('block_time', 0))

                # In Bitcoin's UTXO model, we need to check if our address is primarily an input or output
                # A transaction is outgoing if our address is in inputs (we're spending)
                # A transaction is incoming if our address is only in outputs (we're receiving)

                # First, determine if we're in inputs (sending)
                address_in_inputs = False
                for vin in tx_details.get('vin', []):
                    if 'prevout' in vin and 'scriptpubkey_address' in vin['prevout'] and vin['prevout']['scriptpubkey_address'] == address:
                        address_in_inputs = True
                        break

                # Format transaction into our expected format
                formatted_tx = {
                    'hash': tx_details['txid'],
                    'metadata': {
                        'blockTimestamp': block_timestamp,
                        'blockHeight': block_height
                    }
                }

                # Handle outgoing transactions (if we're in the inputs)
                if address_in_inputs:
                    # This is an outgoing transaction
                    # Find who received the funds from our address (excluding change back to our address)
                    receiver = None
                    for vout in tx_details.get('vout', []):
                        if 'scriptpubkey_address' in vout and vout['scriptpubkey_address'] != address:
                            receiver = vout['scriptpubkey_address']
                            break

                    # Calculate how much was sent from our address
                    value = 0
                    for vin in tx_details.get('vin', []):
                        if 'prevout' in vin and 'scriptpubkey_address' in vin['prevout'] and vin['prevout']['scriptpubkey_address'] == address:
                            value += vin['prevout'].get('value', 0)

                    # Account for change sent back to our address
                    for vout in tx_details.get('vout', []):
                        if 'scriptpubkey_address' in vout and vout['scriptpubkey_address'] == address:
                            value -= vout.get('value', 0)

                    # Ensure value is positive
                    value = max(0, value)

                    formatted_tx['from'] = address
                    formatted_tx['to'] = receiver if receiver else 'unknown'
                    formatted_tx['value'] = value
                    outgoing_txs.append(formatted_tx)
                else:
                    # Check if we're in the outputs (receiving)
                    address_in_outputs = False
                    received_value = 0
                    for vout in tx_details.get('vout', []):
                        if 'scriptpubkey_address' in vout and vout['scriptpubkey_address'] == address:
                            address_in_outputs = True
                            received_value += vout.get('value', 0)

                    # Only process as incoming if we're in outputs but not in inputs
                    if address_in_outputs:
                        # Find who sent the funds to our address
                        sender = None
                        for vin in tx_details.get('vin', []):
                            if 'prevout' in vin and 'scriptpubkey_address' in vin['prevout']:
                                sender = vin['prevout']['scriptpubkey_address']
                                break

                        formatted_tx['from'] = sender if sender else 'unknown'
                        formatted_tx['to'] = address
                        formatted_tx['value'] = received_value
                        incoming_txs.append(formatted_tx)

        log.info(f"Processed {len(incoming_txs)} incoming and {len(outgoing_txs)} outgoing transactions within block range")

        return {
            'incoming': incoming_txs,
            'outgoing': outgoing_txs
        }

    except Exception as e:
        log.error(f"Error fetching address transactions: {e}")
        raise Exception(f"Error fetching address transactions: {e}")

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
    :param network: The Bitcoin network (e.g., 'mainnet', 'testnet')
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

        if tx["to"] != target_address:
            continue

        # Create transaction object
        transaction = Transaction(
            transaction_hash=tx["hash"],
            address_from=Address(tx["from"], AddressType.WALLET),
            address_to=Address(tx["to"], AddressType.WALLET),
            value=satoshi_to_btc(tx["value"]),
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(int(tx["metadata"]["blockTimestamp"]))),
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

        # Create transaction object
        transaction = Transaction(
            transaction_hash=tx["hash"],
            address_from=Address(tx["from"], AddressType.WALLET),
            address_to=Address(tx["to"], AddressType.WALLET),
            value=satoshi_to_btc(tx["value"]),
            timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(int(tx["metadata"]["blockTimestamp"]))),
            token_symbol=BTC_TOKEN_SYMBOL
        )

        interactions.add((InteractionDirection.OUTGOING, transaction))
        tx_count += 1

    log.info(f"Found {len(interactions)} unique transactions for address {target_address.address} between {from_timestamp} and {to_timestamp}")
    return interactions
