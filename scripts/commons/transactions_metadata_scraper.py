import os
import time
import json
import requests
from typing import Dict, Optional, List
from dotenv import load_dotenv

from scripts.commons.known_token_list import ETH_TOKENS_WHITELIST, ETH_TOKENS_BLACKLIST
from scripts.commons.model import *

log = get_logger()

# Load environment variables from .env file
load_dotenv()

# Get the Alchemy API key from the environment variable
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

if not ALCHEMY_API_KEY:
    raise ValueError("ALCHEMY_API_KEY is not set in the environment variables.")

# Initialize address type cache with hardcoded values
_address_type_cache = {
    # Well-known contracts
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2:eth-mainnet": AddressType.CONTRACT,  # WETH
    "0x6b175474e89094c44da98b954eedeac495271d0f:eth-mainnet": AddressType.CONTRACT,  # DAI
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48:eth-mainnet": AddressType.CONTRACT,  # USDC
    "0xdac17f958d2ee523a2206206994597c13d831ec7:eth-mainnet": AddressType.CONTRACT,  # USDT
    # Well-known wallets (e.g. exchanges)
    "0x28c6c06298d514db089934071355e5743bf21d60:eth-mainnet": AddressType.WALLET,  # Binance 14
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549:eth-mainnet": AddressType.WALLET,  # Binance Cold Wallet
}

# Path to the persistent cache file
_CACHE_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "graph", "cache", "address_type_cache.json")


# Load cache from file if it exists
def _load_address_type_cache():
    if os.path.exists(_CACHE_FILE_PATH):
        try:
            with open(_CACHE_FILE_PATH, 'r') as f:
                cache_data = json.load(f)
                # Convert string enum values back to enum objects
                for key, value in cache_data.items():
                    _address_type_cache[key] = AddressType[value]
            log.info(f"Loaded {len(cache_data)} address types from cache file")
        except Exception as e:
            log.warning(f"Failed to load address type cache: {e}")


# Save cache to file
def _save_address_type_cache():
    try:
        # Convert enum objects to strings for JSON serialization
        cache_data = {k: v.name for k, v in _address_type_cache.items()}
        os.makedirs(os.path.dirname(_CACHE_FILE_PATH), exist_ok=True)
        with open(_CACHE_FILE_PATH, 'w') as f:
            json.dump(cache_data, f)
        log.info(f"Saved {len(cache_data)} address types to cache file")
    except Exception as e:
        log.warning(f"Failed to save address type cache: {e}")


# Load the cache at module initialization
_load_address_type_cache()

# Keep track of addresses pending batch processing
_address_batch = {}
_batch_size = 50
_batch_timer = time.time()
_batch_timeout = 5  # seconds


def get_address_types_batch(addresses: List[str], network: str = "eth-mainnet") -> Dict[str, AddressType]:
    """
    Get address types for multiple addresses in a batch to reduce API calls.

    :param addresses: List of Ethereum addresses to check
    :param network: Ethereum network
    :return: Dictionary mapping addresses to their types
    """
    if not addresses:
        return {}

    url = f"https://{network}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
    results = {}

    # Process in chunks to avoid exceeding API limits
    chunk_size = 20
    for i in range(0, len(addresses), chunk_size):
        chunk = addresses[i:i + chunk_size]
        batch_payload = []

        for idx, addr in enumerate(chunk):
            batch_payload.append({
                "jsonrpc": "2.0",
                "method": "eth_getCode",
                "params": [addr, "latest"],
                "id": idx
            })

        try:
            response = requests.post(url, json=batch_payload)
            if response.status_code == 200:
                responses = response.json()

                # Process each response in the batch
                for resp in responses:
                    idx = resp.get("id")
                    if idx is not None and 0 <= idx < len(chunk):
                        addr = chunk[idx]
                        cache_key = f"{addr.lower()}:{network}"

                        if "result" in resp:
                            # Determine address type based on code
                            if resp["result"] == "0x":
                                results[addr] = AddressType.WALLET
                            else:
                                results[addr] = AddressType.CONTRACT

                            # Update cache
                            _address_type_cache[cache_key] = results[addr]
                            log.debug(f"Batch: Found address type for {addr} on {network}: {results[addr].name}")
            else:
                log.error(f"Batch request error: {response.status_code}, {response.text}")
        except Exception as e:
            log.error(f"Error in batch address type request: {str(e)}")

    return results


def get_address_type(address: str, network: str = "eth-mainnet") -> AddressType:
    """
    Determines if an address is a contract or a wallet by checking if it has code.
    Uses caching and batch processing to reduce API calls.

    :param address: Ethereum address to check
    :param network: Ethereum network
    :return: AddressType.CONTRACT if it's a contract, AddressType.WALLET if it's a wallet
    """
    # Normalize address for consistent cache keys
    normalized_address = address.lower()
    cache_key = f"{normalized_address}:{network}"

    # Check if result is already in cache
    if cache_key in _address_type_cache:
        log.debug(f"Using cached address type for {address} on {network}")
        return _address_type_cache[cache_key]

    global _address_batch, _batch_timer

    # Add to batch for processing if batch mode is enabled
    if _batch_size > 1:
        _address_batch[normalized_address] = network
        current_time = time.time()

        # Process the batch if it's full or if the timeout has elapsed
        if len(_address_batch) >= _batch_size or (current_time - _batch_timer >= _batch_timeout and _address_batch):
            addresses = list(_address_batch.keys())
            batch_results = get_address_types_batch(addresses, network)
            _address_batch = {}
            _batch_timer = current_time

            # If the current address was processed in the batch, return the result
            if normalized_address in batch_results:
                return batch_results[normalized_address]

    # If not processed in batch or batch mode disabled, use individual API call
    url = f"https://{network}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getCode",
        "params": [address, "latest"],
        "id": 1
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            # If it's a wallet, it will return "0x" (no code)
            # If it's a contract, it will return the bytecode
            if data["result"] == "0x":
                result = AddressType.WALLET
            else:
                result = AddressType.CONTRACT

            # Store result in cache and save to persistent storage
            _address_type_cache[cache_key] = result
            if len(_address_type_cache) % 100 == 0:  # Save periodically to avoid frequent writes
                _save_address_type_cache()

            log.debug(f"Found address type for {address} on {network}: {result.name}")
            return result
        else:
            log.error(f"Error checking if address is contract: {response.status_code}, {response.text}")
            raise Exception(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        log.error(f"Exception in get_address_type: {str(e)}")
        raise


def get_smart_contracts_by_issuer(wallet_address: Address, network: str = "eth-mainnet") -> set[SmartContract]:
    """
    Fetches a list of smart contracts created by a specific wallet

    :param wallet_address: The wallet address to query
    :param network: The Ethereum network (e.g., 'eth-mainnet', 'eth-goerli')
    :return: A list of smart contract addresses issued by the wallet_address
    """

    log.info(f"Fetching smart contracts for wallet: {wallet_address.address} on network: {network}")

    # Alchemy API endpoint
    url = f"https://{network}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

    # Initialize variables for pagination
    transfers = []
    page_key = None

    # Paginate through the results
    while True:
        payload = {
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [
                {
                    "fromBlock": "0x0",
                    "toBlock": "latest",
                    "fromAddress": wallet_address.address,
                    "excludeZeroValue": False,
                    "category": ["external"],
                    **({"pageKey": page_key} if page_key else {})
                }
            ],
            "id": 1
        }

        # Make the API request
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                transfers.extend(data["result"]["transfers"])
                page_key = data["result"].get("pageKey")
                if not page_key:
                    break
            elif "error" in data:
                log.error("Error from Alchemy API")
                raise Exception(f"Error: {data['error']}")
            else:
                raise Exception(f"Response cannot be parsed: {data}")
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    # Filter transfers to only include contract deployments (where 'to' is null)
    deployments = [tx for tx in transfers if tx["to"] is None]
    tx_hashes = [deployment["hash"] for deployment in deployments]

    # Fetch transaction receipts to get contract addresses
    contract_addresses: set[Address] = set()
    for tx_hash in tx_hashes:
        receipt_payload = {
            "jsonrpc": "2.0",
            "method": "eth_getTransactionReceipt",
            "params": [tx_hash],
            "id": 1
        }
        receipt_response = requests.post(url, json=receipt_payload)
        if receipt_response.status_code == 200:
            receipt_data = receipt_response.json()
            if "result" in receipt_data and receipt_data["result"]:
                contract_address: Address = Address(receipt_data["result"].get("contractAddress"), AddressType.CONTRACT)
                if contract_address:
                    contract_addresses.add(contract_address)
        else:
            raise Exception(f"Failed to fetch receipt for transaction {tx_hash}")

    log.info(f"Found {len(contract_addresses)} smart contracts deployed by {wallet_address.address}")
    log.debug(f"Smart contracts: {contract_addresses}")

    contracts: set[SmartContract] = {SmartContract(address=address) for address in contract_addresses}
    return contracts


def print_timestamp(timestamp: int) -> str:
    """
    Converts a Unix timestamp to a human-readable format.

    :param timestamp: Unix timestamp in seconds
    :return: Unix timestamp + Human-readable date string
    """
    return f"{timestamp} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(timestamp))})"


_timestamp_to_block_cache = {}


def get_block_by_timestamp(timestamp: int, network: str = "eth-mainnet") -> str:
    """
    Gets block number by timestamp using Alchemy's utility endpoint.

    :param timestamp: Unix timestamp in seconds
    :param network: The Ethereum network
    :return: Block number in hex format (0x...)
    """
    # Create cache key from timestamp and network
    cache_key = f"{timestamp}:{network}"

    # Check if result is already in cache
    if cache_key in _timestamp_to_block_cache:
        log.info(f"Using cached block number for timestamp: {print_timestamp(timestamp)} on network: {network}")
        return _timestamp_to_block_cache[cache_key]

    # If not in cache, fetch from API
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
            # Convert decimal to hex with '0x' prefix
            block_info = data["data"][0]["block"]
            number = block_info["number"]
            hex_number = hex(number)
            timestamp_from_block = block_info["timestamp"]
            log.debug(f"Block number (dec): {number}, Timestamp: {timestamp_from_block}")
            log.debug(f"Block number (hex): {hex_number}")
            log.info(f"Block number for timestamp: {print_timestamp(timestamp)} on network: {network} was successfully fetched")
            # Store result in cache
            _timestamp_to_block_cache[cache_key] = hex_number
            return hex(number)
        else:
            log.error(f"Error in block-by-timestamp response: {data}")
            raise Exception(f"Error in block-by-timestamp response: {data}")
    else:
        log.error(f"Error getting block by timestamp: {response.status_code}, {response.text}")
        raise Exception(f"Error: {response.status_code}, {response.text}")


def get_address_interactions(
        target_address: Address,
        from_timestamp: int,
        to_timestamp: int,
        network: str = "eth-mainnet",
        limit: int = 1000
) -> set[tuple[InteractionDirection, Transaction]]:
    """
    Fetches unique addresses that interacted with a address contract during a time period. Can be used for both wallets and contracts.

    :param target_address: The address get interactions for incoming and outgoing transactions
    :param from_timestamp: Start timestamp in Unix seconds
    :param to_timestamp: End timestamp in Unix seconds
    :param network: The Ethereum network (e.g., 'eth-mainnet', 'eth-goerli')
    :return: A set of unique addresses that interacted with the contract
    """
    log.info(f"Fetching interactions for address: {target_address.address} from {print_timestamp(from_timestamp)} to {print_timestamp(to_timestamp)} on network: {network}")

    # Convert timestamps to block numbers using the blocks-by-timestamp endpoint
    from_block = get_block_by_timestamp(from_timestamp, network)
    to_block = get_block_by_timestamp(to_timestamp, network)

    log.info(f"Converted timestamps to blocks: {from_timestamp} -> {from_block}, {to_timestamp} -> {to_block}")

    interactions: set[tuple[InteractionDirection, Transaction]] = set()

    # Get incoming transactions (to the address)
    incoming_interactions = get_interacting_addresses(
        target_address, network, from_block, to_block, InteractionDirection.INCOMING, ["external", "internal", "erc20", "erc721", "erc1155"], limit
    )
    interactions.update(incoming_interactions)

    # Get outgoing transactions (from the address)
    outgoing_interactions = get_interacting_addresses(
        target_address, network, from_block, to_block, InteractionDirection.OUTGOING, ["external", "internal", "erc20", "erc721", "erc1155"], limit
    )
    interactions.update(outgoing_interactions)

    log.info(f"Found {len(interactions)} unique addresses interacting with address {target_address} between {from_timestamp} and {to_timestamp}")
    log.debug(interactions)
    return interactions


def get_interacting_addresses(
        target_address: Address,
        network: str,
        from_block: str,
        to_block: str,
        direction: InteractionDirection,
        categories: list[str],
        limit: int = 1000
) -> set[tuple[InteractionDirection, Transaction]]:
    """
    Get addresses interacting with a contract in a specific direction.

    :param target_address: The target address (contract or wallet) to get interactions for
    :param network: The Ethereum network
    :param from_block: Start block in hex
    :param to_block: End block in hex
    :param direction: 'from' or 'to' indicating direction of interaction
    :param categories: List of transaction categories to query
    :return: A set of unique addresses that interacted with the target address
    """

    # Alchemy API endpoint
    url = f"https://{network}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

    interactions: set[tuple[InteractionDirection, Transaction]] = set()
    page_key = None

    while True:
        params = {
            "fromBlock": from_block,
            "toBlock": to_block,
            "excludeZeroValue": False,
            "category": categories,
            "withMetadata": True,
        }

        # Set direction parameter (fromAddress or toAddress)
        if direction == InteractionDirection.OUTGOING:
            params["fromAddress"] = target_address.address
        elif direction == InteractionDirection.INCOMING:
            params["toAddress"] = target_address.address
        else:
            raise ValueError("Invalid direction. Supported directions: " + ", ".join([d.name for d in InteractionDirection]))

        if page_key:
            params["pageKey"] = page_key

        payload = {
            "jsonrpc": "2.0",
            "method": "alchemy_getAssetTransfers",
            "params": [params],
            "id": 1
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            if "result" in data:
                transfers = data["result"]["transfers"]
                log.debug(f"Retrieved {len(transfers)} transfers for {direction} direction")

                # Extract the addresses from the opposite direction
                for transfer in transfers:
                    if transfer.get("hash") and transfer.get("from") and "value" in transfer and "metadata" in transfer and "asset" in transfer:
                        # For contract creation transactions, "to" will be None
                        if transfer.get("to") is None:
                            # For contract creation, we need to get the contract address from the transaction receipt
                            log.warning(f"Transaction with to=null found. Skipping. Transaction hash: {transfer.get('hash')}")
                            continue
                        if transfer["value"] is None:
                            # Probably token transfers (like ERC-721 NFTs), ignore for now
                            log.warning(f"Transaction with value=null found. Skipping. Transaction hash: {transfer.get('hash')}")
                            continue
                        if transfer.get("asset") in ETH_TOKENS_BLACKLIST:
                            log.debug(f"Skipping transfer with blacklisted asset {transfer.get('asset')}")
                            continue
                        if transfer.get("asset") not in ETH_TOKENS_WHITELIST:
                            # Only ETH transfers are supported for now
                            # Uncomment the next line to log unknown assets
                            # log.warning(f"Transfer with unknown asset {transfer.get('asset')} found, but this asset is not in the whitelist. Skipping. Transaction hash: {transfer.get('hash')}")
                            continue
                        # Determine address types
                        from_address_type: AddressType = get_address_type(transfer["from"])
                        to_address_type: AddressType = get_address_type(transfer["to"])

                        # Create Interaction object with all required fields
                        interaction = Transaction(
                            transaction_hash=transfer["hash"],
                            address_from=Address(transfer["from"].lower(), from_address_type),
                            address_to=Address(transfer["to"].lower(), to_address_type),
                            value=float(transfer["value"]),
                            timestamp=transfer["metadata"]["blockTimestamp"],
                            token_symbol=transfer["asset"]
                        )
                        interactions.add((direction, interaction))
                        if (len(interactions) >= limit):
                            log.info(f"Reached transaction limit of {limit} for {direction} direction. Stopping further processing.")
                            return interactions
                    else:
                        raise ValueError("Invalid transfer: missing required fields", transfer)

                # Check for more pages
                page_key = data["result"].get("pageKey")
                if not page_key:
                    break
            elif "error" in data:
                log.error(f"Error from Alchemy API: {data['error']}")
                if data["error"]["code"] in [-3200, -32603]:  # Alchemy internal errors, skip
                    log.error(f"Skipping transfer because of Alchemy internal error: {data}")
                    continue
                else:
                    raise Exception(f"Error: {data['error']}")
            else:
                raise Exception(f"Response cannot be parsed: {data}")
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    log.info(f"Found {len(interactions)} addresses for {direction} direction")
    return interactions
