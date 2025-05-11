from dataclasses import dataclass
from enum import StrEnum

from scripts.commons import checks

class InteractionDirection(StrEnum):
    INCOMING = "incoming"
    OUTGOING = "outgoing"

@dataclass(frozen=True)
class SmartContract:
    address: str

    def __post_init__(self):
        checks.check_ethereum_address_validity(self.address)


# TODO Refactor interaction model to use this style instead of "Direction + Address"
# class Interaction:
#     address_from: str
#     address_to: str
#     value: float
#     timestamp: int
#     direction: InteractionDirection
#
#     def __post_init__(self):
#         checks.check_ethereum_address_validity(self.address_from)
#         checks.check_ethereum_address_validity(self.address_to)



