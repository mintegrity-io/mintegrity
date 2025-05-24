from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from scripts.commons.logging_config import get_logger
from scripts.commons.metadata import get_token_price_usd

log = get_logger()


# IntEnum would be more efficient everywhere, but StrEnum for better readability - object is dumped to JSON

class InteractionDirection(StrEnum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"


class AddressType(StrEnum):
    WALLET = "wallet"
    CONTRACT = "contract"


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
    token_symbol: str
    value_usd: float = None

    def __post_init__(self):
        # Calculate value_usd after constructor is called
        if self.value_usd is None:
            # You'll need to implement the actual conversion logic here
            # This is just a placeholder
            object.__setattr__(self, 'value_usd', self.calculate_usd_value())

    def calculate_usd_value(self) -> float:
        """
        Calculate the USD value based on token value
        """
        return self.value * get_token_price_usd(self.token_symbol, self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_hash": self.transaction_hash,
            "address_from": self.address_from.to_dict(),
            "address_to": self.address_to.to_dict(),
            "value": self.value,
            "timestamp": self.timestamp,
            "token_symbol": self.token_symbol,
            "value_usd": self.value_usd
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Transaction':
        return cls(
            transaction_hash=data["transaction_hash"],
            address_from=Address.from_dict(data["address_from"]),
            address_to=Address.from_dict(data["address_to"]),
            value=data["value"],
            timestamp=data["timestamp"],
            token_symbol=data.get("token_symbol"),
            value_usd=data.get("value_usd")
        )