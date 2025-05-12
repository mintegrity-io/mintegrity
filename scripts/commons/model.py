from dataclasses import dataclass
from enum import StrEnum

from scripts.commons import checks


class InteractionDirection(StrEnum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"


@dataclass(frozen=True)
class Address:
    address: str

    def __post_init__(self):
        checks.check_ethereum_address_validity(self.address)


@dataclass(frozen=True)
class SmartContract:
    address: Address


@dataclass(frozen=True)
class Transaction:
    transaction_hash: str
    address_from: Address
    address_to: Address
    value: float
    timestamp: str
