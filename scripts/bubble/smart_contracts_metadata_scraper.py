import re
import time
import requests
import os

from dotenv import load_dotenv

from scripts.commons.logging_config import get_logger
from scripts.commons import checks
from smart_contracts_metadata import *

log = get_logger()

# Load environment variables from .env file
load_dotenv()

# Get the Alchemy API key from the environment variable
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

if not ALCHEMY_API_KEY:
    raise ValueError("ALCHEMY_API_KEY is not set in the environment variables.")


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
                log.info("Alchemy API request successful")
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
                contract_address: Address = Address(receipt_data["result"].get("contractAddress"))
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


def get_block_by_timestamp(timestamp: int, network: str = "eth-mainnet") -> str:
    """
    Gets block number by timestamp using Alchemy's utility endpoint.

    :param timestamp: Unix timestamp in seconds
    :param network: The Ethereum network
    :return: Block number in hex format (0x...)
    """
    # Utility API endpoint
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
            timestamp_from_block = block_info["timestamp"]
            log.info(f"Block number (dec): {number}, Timestamp: {timestamp_from_block}")
            log.info(f"Block number (hex): {hex(number)}")
            log.info(f"Block number for timestamp: {print_timestamp(timestamp)} on network: {network} was successfully fetched")
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
        network: str = "eth-mainnet"
) -> dict[InteractionDirection, set[Address]]:
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

    interacted_addresses: dict[InteractionDirection, set[Address]] = dict()

    # Get incoming transactions (to the address)
    incoming_addresses = get_interacting_addresses(
        target_address, network, from_block, to_block, InteractionDirection.INCOMING, ["external", "internal", "erc20", "erc721", "erc1155"]
    )
    interacted_addresses[InteractionDirection.INCOMING] = incoming_addresses

    # Get outgoing transactions (from the address)
    outgoing_addresses = get_interacting_addresses(
        target_address, network, from_block, to_block, InteractionDirection.OUTGOING, ["external", "internal", "erc20", "erc721", "erc1155"]
    )
    interacted_addresses[InteractionDirection.OUTGOING] = outgoing_addresses

    total_size = len(interacted_addresses[InteractionDirection.INCOMING]) + len(interacted_addresses[InteractionDirection.OUTGOING])
    log.info(f"Found {total_size} unique addresses interacting with address {target_address} between {from_timestamp} and {to_timestamp}")
    log.debug(interacted_addresses)
    return interacted_addresses


def get_interacting_addresses(
        target_address: Address,
        network: str,
        from_block: str,
        to_block: str,
        direction: InteractionDirection,
        categories: list[str]
) -> set[Address]:
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

    addresses: set[Address] = set()
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
                    if direction == InteractionDirection.OUTGOING and transfer.get("to"):
                        addresses.add(Address(transfer["to"].lower()))
                    elif direction == InteractionDirection.INCOMING and transfer.get("from"):
                        addresses.add(Address(transfer["from"].lower()))

                # Check for more pages
                page_key = data["result"].get("pageKey")
                if not page_key:
                    break
            elif "error" in data:
                log.error(f"Error from Alchemy API: {data['error']}")
                raise Exception(f"Error: {data['error']}")
            else:
                raise Exception(f"Response cannot be parsed: {data}")
        else:
            raise Exception(f"Error: {response.status_code}, {response.text}")

    log.info(f"Found {len(addresses)} addresses for {direction} direction")
    return addresses
