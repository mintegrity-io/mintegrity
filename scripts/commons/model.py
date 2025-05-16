from dataclasses import dataclass
from enum import StrEnum, IntEnum
from typing import Any

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
        object.__setattr__(self, 'address', self.address.lower())
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address,
            "type": self.type.value
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Address':
        return cls(
            address=data["address"],
            type=AddressType(data["type"])
        )


@dataclass(frozen=True)
class SmartContract:
    address: Address

    def __post_init__(self):
        if not self.address.type == AddressType.CONTRACT:
            raise ValueError(f"Address {self.address} is not a contract address. Type: {self.address.type.name}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": self.address.to_dict()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'SmartContract':
        return cls(
            address=Address.from_dict(data["address"])
        )

@dataclass(frozen=True)
class Transaction:
    transaction_hash: str
    address_from: Address
    address_to: Address
    value: float
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_hash": self.transaction_hash,
            "address_from": self.address_from.to_dict(),
            "address_to": self.address_to.to_dict(),
            "value": self.value,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Transaction':
        return cls(
            transaction_hash=data["transaction_hash"],
            address_from=Address.from_dict(data["address_from"]),
            address_to=Address.from_dict(data["address_to"]),
            value=data["value"],
            timestamp=data["timestamp"]
        )
