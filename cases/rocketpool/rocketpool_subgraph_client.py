"""
Rocket Pool Subgraph Client Module

Simple module to handle subgraph interactions for Rocket Pool data fetching.
"""

import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional


class SubgraphClient:
    """Client for interacting with Rocket Pool Subgraph"""
    
    def __init__(self, subgraph_url: str = "https://api.thegraph.com/subgraphs/name/rocket-pool/rocketpool"):
        self.subgraph_url = subgraph_url

    def fetch_user_data(self, address: str) -> Optional[Dict]:
        """Fetch data from Rocket Pool Subgraph - extracted from original fetch_subgraph_data method"""
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=365)
            start_timestamp = int(start_time.timestamp())
            
            query = """
            query GetWalletStats($address: String!, $startTimestamp: Int!) {
                user(id: $address) {
                    id
                    deposits(where: {timestamp_gte: $startTimestamp}, orderBy: timestamp, orderDirection: asc) {
                        id
                        amount
                        block
                        timestamp
                    }
                    withdrawals(where: {timestamp_gte: $startTimestamp}, orderBy: timestamp, orderDirection: asc) {
                        id
                        amount
                        block
                        timestamp
                    }
                }
            }
            """
            
            response = requests.post(
                self.subgraph_url,
                json={"query": query, "variables": {"address": address.lower(), "startTimestamp": start_timestamp}},
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if "errors" in data:
                return {"error": f"Subgraph error: {data['errors']}"}
            
            user_data = data.get("data", {}).get("user")
            if not user_data:
                return {"transactions": [], "fetch_timestamp": datetime.now(timezone.utc).isoformat()}
            
            # Combine deposits and withdrawals
            all_transactions = []
            all_transactions.extend(user_data.get("deposits", []))
            all_transactions.extend(user_data.get("withdrawals", []))
            
            return {
                "transactions": sorted(all_transactions, key=lambda x: int(x["timestamp"])),
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                "period_days": 365
            }
            
        except Exception as e:
            return {"error": f"Subgraph fetch error: {str(e)}"}
