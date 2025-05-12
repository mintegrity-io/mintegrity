from scripts.bubble.smart_contracts_metadata_scraper import *
from dateutil import parser

from scripts.commons.model import *

CONTRACT_ISSUER_ADDRESS = Address("0x8cfae48fb3e54e143e5454ca2784b7bf3a0dc0d4")
FROM_TIME = int(parser.parse("2025-04-01T00:00:00Z").timestamp())
TO_TIME = int(parser.parse("2025-05-01T00:00:00Z").timestamp())

contracts: set[SmartContract] = get_smart_contracts_by_issuer(CONTRACT_ISSUER_ADDRESS)

for contract in contracts:
    interactions_with_contract = get_address_interactions(contract.address, FROM_TIME, TO_TIME)
    print(interactions_with_contract)
