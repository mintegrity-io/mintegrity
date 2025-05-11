import unittest

from scripts.bubble.smart_contracts_metadata_scraper import *
from scripts.bubble.smart_contracts_metadata import *

class TestGetSmartContracts(unittest.TestCase):
    def test_get_smart_contracts_contains_expected_address(self):
        # Verified contracts for test https://etherscan.io/contractsverified
        # Example:
        # Contract 0xdb9400478e42e2c226b85cc7e8d47a7c14b3dc9f
        # Created by 0x8cfae48fb3e54e143e5454ca2784b7bf3a0dc0d4
        contracts = get_smart_contracts_by_issuer("0x8cfae48fb3e54e143e5454ca2784b7bf3a0dc0d4")
        target_address = "0xdb9400478e42e2c226b85cc7e8d47a7c14b3dc9f"
        found_contracts = [c for c in contracts if c.address.lower() == target_address]
        self.assertTrue(len(found_contracts) > 0)

    def test_get_block_by_timestamp_1(self):
        # Test with a specific timestamp
        timestamp = 1633046400
        expected_block = "0xcb66aa"
        result = get_block_by_timestamp(timestamp)

        # Assert the result matches the expected block number
        self.assertEqual(expected_block, result)

    def test_get_block_by_timestamp_2(self):
        # Test with a specific timestamp
        timestamp = 1746377600
        expected_block = "0x155f9bb"
        result = get_block_by_timestamp(timestamp)

        # Assert the result matches the expected block number
        self.assertEqual(expected_block, result)