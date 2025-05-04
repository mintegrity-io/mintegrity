import unittest

from scripts.smart_contracts_metadata import get_smart_contracts, get_block_by_timestamp, get_contract_interactions


class TestGetSmartContracts(unittest.TestCase):
    def test_get_smart_contracts_contains_expected_address(self):
        # Verified contracts for test https://etherscan.io/contractsverified
        # Example:
        # Contract 0xdb9400478e42e2c226b85cc7e8d47a7c14b3dc9f
        # Created by 0x8cfae48fb3e54e143e5454ca2784b7bf3a0dc0d4
        contracts = get_smart_contracts("0x8cfae48fb3e54e143e5454ca2784b7bf3a0dc0d4")
        self.assertIn("0xdb9400478e42e2c226b85cc7e8d47a7c14b3dc9f", contracts)

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

    def test_get_smart_contracts_interactions_expected_address(self):
        interactions = get_contract_interactions("0x8cfae48fb3e54e143e5454ca2784b7bf3a0dc0d4", 1746300000, 1746377600)
        self.assertIsInstance(interactions, dict)
        print(interactions)