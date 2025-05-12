from dateutil import parser
from scripts.commons.model import get_contract_interactions

CONTRACT_ISSUER_ADDRESS = "0x8cfae48fb3e54e143e5454ca2784b7bf3a0dc0d4"
FROM_TIME = int(parser.parse("2025-05-01T00:00:00Z").timestamp())
TO_TIME = int(parser.parse("2025-05-04T17:00:00Z").timestamp())
interactions = get_contract_interactions(CONTRACT_ISSUER_ADDRESS, FROM_TIME, TO_TIME)

print(interactions)