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

# Maximum number of consecutive retry attempts
MAX_RETRY_ATTEMPTS = 5
# Initial delay between retries for 429 errors (in seconds)
INITIAL_RETRY_DELAY_SECONDS = 2
# Connection timeout in seconds
CONNECTION_TIMEOUT_SECONDS = 5

def request_with_retry(url: str) -> requests.Response:
    """
    Performs an HTTP GET request with retry logic for rate limiting (429 errors).

    Uses exponential backoff strategy for retries.

    :param url: URL to fetch
    :return: Response object
    :raises Exception: After MAX_RETRY_ATTEMPTS consecutive failures
    """
    attempt = 0
    retry_delay = INITIAL_RETRY_DELAY_SECONDS

    while attempt < MAX_RETRY_ATTEMPTS:
        attempt += 1
        try:
            log.debug(f"Requesting URL: {url}, attempt {attempt}/{MAX_RETRY_ATTEMPTS}")
            response = requests.get(url, timeout=CONNECTION_TIMEOUT_SECONDS)

            # If we get a 429 error, wait and retry with exponential backoff
            if response.status_code == 429:
                log.warning(f"Request for {url} failed -> Rate limit exceeded (429). Attempt {attempt}/{MAX_RETRY_ATTEMPTS}. "
                           f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                # Double the delay for the next retry (exponential backoff)
                retry_delay = min(retry_delay * 2, 60)  # Cap at 60 seconds
                continue

            # For other errors, just raise the exception
            response.raise_for_status()

            # If request is successful, return the response
            log.debug(f"Request to {url} succeeded")
            return response

        except requests.exceptions.Timeout:
            log.warning(f"Request to {url} timed out. Attempt {attempt}/{MAX_RETRY_ATTEMPTS}. Retrying...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

        except requests.exceptions.ConnectionError:
            log.warning(f"Connection error when requesting {url}. Attempt {attempt}/{MAX_RETRY_ATTEMPTS}. Retrying...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429:
                log.warning(f"Rate limit exceeded (429). Attempt {attempt}/{MAX_RETRY_ATTEMPTS}. "
                           f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                # For other exceptions, just re-raise
                log.error(f"Request error: {e}")
                raise

    # If we've exhausted our retries
    log.error(f"Failed after {MAX_RETRY_ATTEMPTS} retry attempts: {url}")
    raise Exception(f"Failed after {MAX_RETRY_ATTEMPTS} retry attempts")

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

    Uses the dedicated /api/v1/mining/blocks/timestamp/:timestamp endpoint that directly
    returns the block closest to the given timestamp.

    :param timestamp: Unix timestamp in seconds
    :param network: The Bitcoin network
    :return: Block height as a string
    """
    base_url = get_api_base_url(network)

    try:
        # Use the dedicated timestamp-to-block endpoint which is much more efficient
        blocks_url = f"{base_url}/v1/mining/blocks/timestamp/{timestamp}"
        log.info(f"Looking up block for timestamp {print_timestamp(timestamp)} using direct endpoint")

        response = request_with_retry(blocks_url)
        block_data = response.json()

        if 'height' in block_data:
            log.debug(f"Found block height: {block_data['height']} with hash: {block_data.get('hash', 'unknown')}")
            return str(block_data['height'])
        else:
            log.warning(f"Unexpected response format from timestamp endpoint: {block_data}")
            raise Exception(f"Invalid response from mempool API: {block_data}")

    except Exception as e:
        log.error(f"Error finding block by timestamp: {e}")
        raise Exception(f"Error finding block by timestamp: {e}")

def get_address_transactions(address: str, from_block: str, to_block: str, network: str = BTC_MAINNET, limit: int = None) -> dict:
    """
    Fetches transactions for a Bitcoin address between two block heights using mempool.space API.
    Handles pagination to get all transactions within the specified block range.

    :param address: Bitcoin address
    :param from_block: Start block height
    :param to_block: End block height
    :param network: Bitcoin network
    :param limit: Maximum number of transactions to return (both incoming and outgoing combined)
    :return: Dictionary containing transaction data
    """
    base_url = get_api_base_url(network)
    from_block_int = int(from_block)
    to_block_int = int(to_block)

    log.info(f"Fetching transactions for address {address} from block {from_block_int} to {to_block_int}" +
             (f" with limit {limit}" if limit else ""))

    # Initialize results
    incoming_txs = []
    outgoing_txs = []

    # Track all transactions to process
    all_transactions = []
    total_transactions_processed = 0

    try:
        # Fetch transactions for the address with pagination
        # The API returns up to 50 mempool transactions + 25 confirmed transactions per request
        last_txid = None
        has_more = True
        reached_from_block = False
        reached_limit = False

        while has_more and not reached_from_block and not reached_limit:
            # Construct URL with pagination parameter if needed
            txs_url = f"{base_url}/address/{address}/txs/chain"
            if last_txid:
                txs_url = f"{txs_url}/{last_txid}"

            log.info(f"Fetching transactions from: {txs_url}")
            response = request_with_retry(txs_url)

            # Check if the request was successful
            if response.status_code != 200:
                log.error(f"API Error: {response.status_code} - {response.text}")
                if response.status_code == 400:
                    log.warning(f"Bad request for address: {address}. This might not be a valid Bitcoin address.")
                    break
                response.raise_for_status()

            batch_txs = response.json()
            log.debug(f"Retrieved {len(batch_txs)} transactions in this batch")

            if not batch_txs:
                has_more = False
                continue

            # Process each transaction in this batch
            for tx_details in batch_txs:
                # Store the last txid for pagination
                last_txid = tx_details['txid']

                # Get block height from the transaction data
                block_height = tx_details['status'].get('block_height')

                # Skip unconfirmed transactions or those outside our range
                if block_height is None or block_height > to_block_int:
                    log.info(f"Skipping transaction {tx_details['txid']} - block height {block_height} is outside range {from_block_int} to {to_block_int}")
                    continue

                if block_height < from_block_int:
                    # We've gone past our target range, can stop fetching
                    reached_from_block = True
                    break

                # This transaction is within our range, add to the list to process
                all_transactions.append(tx_details)
                total_transactions_processed += 1

                # Check if we've reached the transaction limit
                if limit and total_transactions_processed >= limit:
                    log.info(f"Reached transaction limit of {limit}. Stopping transaction fetching.")
                    reached_limit = True
                    break

            # If we got fewer than expected transactions, we've reached the end
            # mempool.space returns 50 mempool + 25 confirmed transactions per request
            if len(batch_txs) < 25:
                has_more = False

        log.info(f"Found {len(all_transactions)} total transactions for address {address} within block range")

        # Process collected transactions
        for tx_details in all_transactions:
            # Get the block timestamp as a string
            block_timestamp = str(tx_details['status'].get('block_time', 0))
            block_height = tx_details['status']['block_height']

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

        log.info(f"Processed {len(incoming_txs)} incoming and {len(outgoing_txs)} outgoing transactions within block range for address {address}")

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
    log.info(f"Fetching interactions for address: {target_address.address} from {print_timestamp(from_timestamp)} to {print_timestamp(to_timestamp)} on network: {network} with limit {limit}")

    # Convert timestamps to block heights
    from_block = get_block_by_timestamp(from_timestamp, network)
    to_block = get_block_by_timestamp(to_timestamp, network)

    log.info(f"Converted timestamps to blocks: {from_timestamp} -> {from_block}, {to_timestamp} -> {to_block}")

    # Get all transactions for this address with the limit applied at the API level
    transactions_data = get_address_transactions(target_address.address, from_block, to_block, network, limit=limit)

    interactions: set[tuple[InteractionDirection, Transaction]] = set()
    tx_count = 0

    # Process incoming transactions
    for tx in transactions_data["incoming"]:
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
