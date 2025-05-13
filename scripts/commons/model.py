from dataclasses import dataclass
from enum import StrEnum, IntEnum

from scripts.commons.logging_config import get_logger

log = get_logger()


class InteractionDirection(StrEnum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"


class AddressType(IntEnum):
    WALLET = 0
    CONTRACT = 1


@dataclass(frozen=True)
class Address:
    address: str
    type: AddressType

    def __post_init__(self):
        self.check_ethereum_address_validity()

    def check_ethereum_address_validity(self):
        """
        Validates if the given string is a valid Ethereum address.

        :return: True if valid, False otherwise
        """
        if not self.address.startswith("0x") or len(self.address) != 42:
            raise ValueError(f"Invalid Ethereum address: {self.address}")
        else:
            log.debug(f"Valid Ethereum address: {self.address}")


@dataclass(frozen=True)
class SmartContract:
    address: Address

    def __post_init__(self):
        if not self.address.type == AddressType.CONTRACT:
            raise ValueError(f"Address {self.address} is not a contract address. Type: {self.address.type.name}")


@dataclass(frozen=True)
class Transaction:
    transaction_hash: str
    address_from: Address
    address_to: Address
    value: float
    timestamp: str
