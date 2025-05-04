import re
import time
from typing import Dict, Set

import requests
import os
import logging

from dotenv import load_dotenv

from logging_config import get_logger

log = get_logger()

# Load environment variables from .env file
load_dotenv()

# Get the Alchemy API key from the environment variable
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

if not ALCHEMY_API_KEY:
    raise ValueError("ALCHEMY_API_KEY is not set in the environment variables.")


def get_smart_contracts(wallet_address: str, network: str = "eth-mainnet") -> set[str]:
    """
    Fetches a list of smart contracts created by a specific wallet

    :param wallet_address: The wallet address to query
    :param network: The Ethereum network (e.g., 'eth-mainnet', 'eth-goerli')
    :return: A list of smart contract addresses issued by the wallet_address
    """

    log.info(f"Fetching smart contracts for wallet: {wallet_address} on network: {network}")

    check_ethereum_address_validity(wallet_address)

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
                    "fromAddress": wallet_address,
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
    contract_addresses = set()
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
                contract_address = receipt_data["result"].get("contractAddress")
                if contract_address:
                    contract_addresses.add(contract_address)
        else:
            raise Exception(f"Failed to fetch receipt for transaction {tx_hash}")

    log.info(f"Found {len(contract_addresses)} smart contracts deployed by {wallet_address}")
    log.debug(f"Smart contracts: {contract_addresses}")
    return contract_addresses


def check_ethereum_address_validity(address: str) -> None:
    """
    Validates if the given string is a valid Ethereum address.

    :param address: The Ethereum address to validate
    :return: True if valid, False otherwise
    """
    if not bool(re.match(r"^0x[a-fA-F0-9]{40}$", address)):
        raise ValueError(f"Invalid Ethereum address: {address}")
    else:
        log.info(f"Valid Ethereum address: {address}")


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


def get_contract_interactions(
        contract_address: str,
        from_timestamp: int,
        to_timestamp: int,
        network: str = "eth-mainnet"
) -> dict[str, set[str]]:
    """
    Fetches unique addresses that interacted with a specific contract during a time period.

    :param contract_address: The contract address to query
    :param from_timestamp: Start timestamp in Unix seconds
    :param to_timestamp: End timestamp in Unix seconds
    :param network: The Ethereum network (e.g., 'eth-mainnet', 'eth-goerli')
    :return: A set of unique addresses that interacted with the contract
    """
    log.info(f"Fetching interactions for contract: {contract_address} from {print_timestamp(from_timestamp)} to {print_timestamp(to_timestamp)} on network: {network}")
    check_ethereum_address_validity(contract_address)

    # Convert timestamps to block numbers using the blocks-by-timestamp endpoint
    from_block = get_block_by_timestamp(from_timestamp, network)
    to_block = get_block_by_timestamp(to_timestamp, network)

    log.info(f"Converted timestamps to blocks: {from_timestamp} -> {from_block}, {to_timestamp} -> {to_block}")

    # Alchemy API endpoint
    url = f"https://{network}.g.alchemy.com/v2/{ALCHEMY_API_KEY}"

    interacted_addresses = dict()

    # Get incoming transactions (to the contract)
    incoming_addresses = get_interacting_addresses(
        url, contract_address, from_block, to_block, "to", ["external", "internal", "erc20", "erc721", "erc1155"]
    )
    interacted_addresses["to"] = incoming_addresses

    # Get outgoing transactions (from the contract)
    outgoing_addresses = get_interacting_addresses(
        url, contract_address, from_block, to_block, "from", ["external", "internal", "erc20", "erc721", "erc1155"]
    )
    interacted_addresses["from"] = outgoing_addresses

    # Remove the contract address itself from the results
    if contract_address.lower() in interacted_addresses["to"]:
        interacted_addresses["to"].remove(contract_address.lower())
    if contract_address.lower() in interacted_addresses["from"]:
        interacted_addresses["from"].remove(contract_address.lower())

    total_size = len(interacted_addresses["to"]) + len(interacted_addresses["from"])
    log.info(f"Found {total_size} unique addresses interacting with contract address {contract_address} between {from_timestamp} and {to_timestamp}")
    log.debug(interacted_addresses)
    return interacted_addresses


def get_interacting_addresses(
        url: str,
        contract_address: str,
        from_block: str,
        to_block: str,
        direction: str,
        categories: list[str]
) -> set[str]:
    """
    Get addresses interacting with a contract in a specific direction.

    :param url: Alchemy API URL
    :param contract_address: The contract address
    :param from_block: Start block in hex
    :param to_block: End block in hex
    :param direction: 'from' or 'to' indicating direction of interaction
    :param categories: List of transaction categories to query
    :return: A set of unique addresses
    """
    addresses = set()
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
        if direction == "from":
            params["fromAddress"] = contract_address
        else:
            params["toAddress"] = contract_address

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
                    if direction == "from" and transfer.get("to"):
                        addresses.add(transfer["to"].lower())
                    elif direction == "to" and transfer.get("from"):
                        addresses.add(transfer["from"].lower())

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
