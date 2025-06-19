import os
import time
import json
import requests
from typing import Dict, Optional, List
from dotenv import load_dotenv

from scripts.commons.known_token_list import TOKENS_WHITELIST, ETH_TOKENS_BLACKLIST
from scripts.commons.model import *

log = get_logger()

# Load environment variables from .env file
load_dotenv()

# Get the Alchemy API key from the environment variable
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

if not ALCHEMY_API_KEY:
    raise ValueError("ALCHEMY_API_KEY is not set in the environment variables.")

# Initialize address type cache
_address_type_cache = {}

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


def get_address_types(addresses: List[str], network: str = "eth-mainnet") -> Dict[str, AddressType]:
    """
    Determines if multiple addresses are contracts or wallets by checking if they have code.
    Uses caching and batch processing to efficiently handle large numbers of addresses.

    :param addresses: List of Ethereum addresses to check
    :param network: Ethereum network
    :return: Dictionary mapping addresses to their types (AddressType.CONTRACT or AddressType.WALLET)
    """
    if not addresses:
        return {}

    # Check cache first for all addresses
    results = {}
    addresses_to_check = []

    for address in addresses:
        normalized_address = address.lower()
        cache_key = f"{normalized_address}:{network}"

        # If in cache, use cached result
        if cache_key in _address_type_cache:
            results[address] = _address_type_cache[cache_key]
            log.debug(f"Using cached address type for {address} on {network}")
        else:
            # If not in cache, add to list for batch processing
            addresses_to_check.append(normalized_address)

    # If there are addresses not in cache, process them in batch
    if addresses_to_check:
        batch_results = get_address_types_batch(addresses_to_check, network)

        # Add batch results to the final results
        for addr, addr_type in batch_results.items():
            # Find the original address with original casing
            original_addr = next((a for a in addresses if a.lower() == addr.lower()), addr)
            results[original_addr] = addr_type

    return results


def get_address_type(address: str, network: str = "eth-mainnet") -> AddressType:
    """
    Determines if an address is a contract or a wallet by checking if it has code.
    Uses caching and batch processing to reduce API calls.

    :param address: Ethereum address to check
    :param network: Ethereum network
    :return: AddressType.CONTRACT if it's a contract, AddressType.WALLET if it's a wallet
    """
    # Use the new get_address_types function for a single address
    result = get_address_types([address], network)

    # Return the result for the address or raise an exception if not found
    if address in result:
        return result[address]

    # If the address wasn't found in the result (which shouldn't happen),
    # make a direct API call as a last resort
    log.warning(f"Address {address} not found in batch results, making direct API call")

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

            # Store result in cache
            cache_key = f"{address.lower()}:{network}"
            _address_type_cache[cache_key] = result

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

    contracts: set[SmartContract] = {SmartContract(address.address) for address in contract_addresses}
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


def _is_alchemy_internal_error(error_data) -> bool:
    """
    Check if the error from Alchemy API is an internal error that should be skipped.

    :param error_data: The error data from Alchemy API response
    :return: True if it's an internal error that should be skipped, False otherwise
    """
    ALCHEMY_INTERNAL_ERROR_CODES = [-3200, -32603, -32000, 1002]  # Alchemy internal error codes
    if "error" in error_data:
        # Handle case where error_data["error"] is a dictionary
        if isinstance(error_data["error"], dict) and error_data["error"].get("code") in ALCHEMY_INTERNAL_ERROR_CODES:
            log.error(f"Detected Alchemy internal error: {error_data}")
            return True
        # Handle case where error_data["error"] is a string
        elif isinstance(error_data["error"], str):
            log.error(f"Detected Alchemy error message (string): {error_data}")
            # Check if the error message contains any of the error codes we're looking for
            for code in ALCHEMY_INTERNAL_ERROR_CODES:
                if str(code) in error_data["error"]:
                    return True
    return False


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

                # First, filter transfers and collect unique addresses
                valid_transfers = []
                unique_addresses = set()

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
                        if transfer.get("asset") not in TOKENS_WHITELIST:
                            # Only ETH transfers are supported for now
                            # Uncomment the next line to log unknown assets
                            # log.warning(f"Transfer with unknown asset {transfer.get('asset')} found, but this asset is not in the whitelist. Skipping. Transaction hash: {transfer.get('hash')}")
                            continue

                        # Add to valid transfers and collect unique addresses
                        valid_transfers.append(transfer)
                        unique_addresses.add(transfer["from"])
                        if transfer.get("to"):
                            unique_addresses.add(transfer["to"])

                # Get address types for all unique addresses in one batch call
                if unique_addresses:
                    address_types = get_address_types(list(unique_addresses), network)

                    # Process valid transfers with the address types
                    for transfer in valid_transfers:
                        from_address_type = address_types.get(transfer["from"], AddressType.WALLET)  # Default to WALLET if not found
                        to_address_type = address_types.get(transfer["to"], AddressType.WALLET)  # Default to WALLET if not found

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

                # Check for more pages
                page_key = data["result"].get("pageKey")
                if not page_key:
                    break
            elif "error" in data:
                log.error(f"Error from Alchemy API: {data['error']}")
                if _is_alchemy_internal_error(data):
                    log.error(f"Skipping transfer because of Alchemy internal error: {data}")
                    continue
                else:
                    raise Exception(f"Error: {data['error']}")
            else:
                raise Exception(f"Response cannot be parsed: {data}")
        else:
            # Try to parse the response as JSON to check for error codes
            try:
                error_data = response.json()
                if _is_alchemy_internal_error(error_data):
                    log.error(f"Skipping transfer because of Alchemy internal error with status {response.status_code}: {error_data}")
                    continue
                else:
                    raise Exception(f"Error: {response.status_code}, {response.text}")
            except json.JSONDecodeError:
                raise Exception(f"Error: {response.status_code}, {response.text}")

    log.info(f"Found {len(interactions)} addresses for {direction} direction")
    return interactions
