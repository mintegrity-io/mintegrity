"""
Etherscan API Client

Universal client for fetching data from Etherscan API.
Supports transaction history, contract verification, and other Etherscan endpoints.
"""

import os
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
import logging
import time

log = logging.getLogger(__name__)


class EtherscanClient:
    """
    Universal Etherscan API client with rate limiting and error handling
    
    Features:
    - Automatic retry on failures with exponential backoff
    - Rate limiting (5 requests/second) to comply with Etherscan limits
    - Handles API errors gracefully (NOTOK, rate limits, result window size)
    - Supports pagination for large result sets
    
    Etherscan API Limits:
    - Free plan: 5 requests/second, 100,000 requests/day
    - Result window: PageNo x Offset <= 10,000
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize Etherscan API client
        
        Args:
            api_key: Etherscan API key. If None, tries to get from ETHERSCAN_API_KEY env var
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY")
        self.timeout = timeout
        self.base_url = "https://api.etherscan.io/api"
        
        if not self.api_key:
            log.warning("No Etherscan API key provided. Some operations will not work.")
        else:
            log.debug(f"Initialized EtherscanClient with API key: {self.api_key[:8]}...")
    
    def _make_request(self, params: Dict, max_retries: int = 3) -> Optional[Dict]:
        """
        Make API request to Etherscan with rate limiting and retries
        
        Args:
            params: API parameters
            max_retries: Maximum number of retries for failed requests
            
        Returns:
            API response data or None if error
        """
        if not self.api_key:
            return {"error": "API key not available"}
        
        # Add API key to parameters
        params["apikey"] = self.api_key
        
        for attempt in range(max_retries + 1):
            try:
                log.debug(f"Making Etherscan API request: {params.get('module')}.{params.get('action')} (attempt {attempt + 1})")
                
                # Rate limiting: 5 requests per second max
                time.sleep(0.2)  # 200ms delay between requests
                
                response = requests.get(self.base_url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                # Check API response status
                if data.get("status") != "1":
                    error_message = data.get("message", "Unknown error")
                    
                    # Handle specific error types
                    if "NOTOK" == error_message and attempt < max_retries:
                        log.debug(f"Got NOTOK error, retrying in {(attempt + 1) * 2} seconds...")
                        time.sleep((attempt + 1) * 2)  # Exponential backoff
                        continue
                    
                    if "Rate limit exceeded" in error_message and attempt < max_retries:
                        wait_time = (attempt + 1) * 5
                        log.warning(f"Rate limit exceeded, waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    
                    if "Result window is too large" in error_message:
                        # Don't retry for this error, let the caller handle it
                        log.warning(f"Etherscan API error: {error_message}")
                        return {"error": f"Etherscan API error: {error_message}"}
                    
                    log.warning(f"Etherscan API error: {error_message}")
                    return {"error": f"Etherscan API error: {error_message}"}
                
                return data
                
            except requests.exceptions.RequestException as e:
                error_msg = f"HTTP request failed: {str(e)}"
                if attempt < max_retries:
                    log.debug(f"{error_msg} - retrying in {(attempt + 1) * 2} seconds...")
                    time.sleep((attempt + 1) * 2)
                    continue
                else:
                    log.error(error_msg)
                    return {"error": f"Etherscan fetch error: {str(e)}"}
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                if attempt < max_retries:
                    log.debug(f"{error_msg} - retrying...")
                    time.sleep((attempt + 1) * 2)
                    continue
                else:
                    log.error(error_msg)
                    return {"error": error_msg}
        
        return {"error": "Max retries exceeded"}
    
    def fetch_transactions(self, address: str, days: int = 365, max_transactions: int = 10000) -> Optional[Dict]:
        """
        Fetch transaction history for an address with pagination support
        
        Args:
            address: Ethereum address
            days: Number of days to look back (default: 365)
            max_transactions: Maximum transactions to fetch (Etherscan limit: 10000)
            
        Returns:
            Dictionary with transactions or error information
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        start_timestamp = int(start_time.timestamp())
        
        # Etherscan limits: PageNo x Offset <= 10000
        # Use smaller offset to avoid the error
        max_offset = min(max_transactions, 10000)
        
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": max_offset,
            "sort": "desc",  # Get recent transactions first
        }
        
        response = self._make_request(params)
        
        if not response or "error" in response:
            # If error due to too many results, try with smaller offset
            if response and "Result window is too large" in str(response.get("error", "")):
                log.warning(f"Large result set for {address}, trying with reduced limit...")
                params["offset"] = 5000  # Try with smaller limit
                response = self._make_request(params)
                
                if not response or "error" in response:
                    # If still failing, try with even smaller limit
                    params["offset"] = 1000
                    response = self._make_request(params)
            
            # If still error, return the error
            if not response or "error" in response:
                return response
        
        all_transactions = response.get("result", [])
        
        # Filter transactions by time period
        filtered_transactions = [
            tx for tx in all_transactions 
            if int(tx["timeStamp"]) >= start_timestamp
        ]
        
        log.debug(f"Fetched {len(filtered_transactions)}/{len(all_transactions)} transactions for {address} ({days} days)")
        
        return {
            "transactions": filtered_transactions,
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "total_transactions": len(all_transactions),
            "filtered_transactions": len(filtered_transactions),
            "truncated": len(all_transactions) >= max_offset  # Indicate if results may be truncated
        }
    
    def fetch_internal_transactions(self, address: str, days: int = 365, max_transactions: int = 10000) -> Optional[Dict]:
        """
        Fetch internal transaction history for an address
        
        Args:
            address: Ethereum address
            days: Number of days to look back
            max_transactions: Maximum transactions to fetch (Etherscan limit: 10000)
            
        Returns:
            Dictionary with internal transactions or error information
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        start_timestamp = int(start_time.timestamp())
        
        # Use safe offset limit
        max_offset = min(max_transactions, 10000)
        
        params = {
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": max_offset,
            "sort": "desc",
        }
        
        response = self._make_request(params)
        
        if not response or "error" in response:
            # Try with smaller offset if needed
            if response and "Result window is too large" in str(response.get("error", "")):
                params["offset"] = 5000
                response = self._make_request(params)
            
            if not response or "error" in response:
                return response
        
        # Filter by time period
        all_transactions = response.get("result", [])
        
        filtered_transactions = [
            tx for tx in all_transactions 
            if int(tx["timeStamp"]) >= start_timestamp
        ]
        
        return {
            "internal_transactions": filtered_transactions,
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "truncated": len(all_transactions) >= max_offset
        }
    
    def fetch_token_transfers(self, address: str, days: int = 365, contract_address: Optional[str] = None, max_transfers: int = 10000) -> Optional[Dict]:
        """
        Fetch ERC-20 token transfer events for an address
        
        Args:
            address: Ethereum address
            days: Number of days to look back
            contract_address: Specific token contract address (optional)
            max_transfers: Maximum transfers to fetch (Etherscan limit: 10000)
            
        Returns:
            Dictionary with token transfers or error information
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        start_timestamp = int(start_time.timestamp())
        
        # Use safe offset limit
        max_offset = min(max_transfers, 10000)
        
        params = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "page": 1,
            "offset": max_offset,
            "sort": "desc",
        }
        
        if contract_address:
            params["contractaddress"] = contract_address
        
        response = self._make_request(params)
        
        if not response or "error" in response:
            # Try with smaller offset if needed
            if response and "Result window is too large" in str(response.get("error", "")):
                params["offset"] = 5000
                response = self._make_request(params)
            
            if not response or "error" in response:
                return response
        
        # Filter by time period
        all_transfers = response.get("result", [])
        
        filtered_transfers = [
            tx for tx in all_transfers 
            if int(tx["timeStamp"]) >= start_timestamp
        ]
        
        return {
            "token_transfers": filtered_transfers,
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "period_days": days,
            "truncated": len(all_transfers) >= max_offset
        }
    
    def get_contract_abi(self, contract_address: str) -> Optional[Dict]:
        """
        Get verified contract ABI
        
        Args:
            contract_address: Contract address
            
        Returns:
            Dictionary with ABI or error information
        """
        params = {
            "module": "contract",
            "action": "getabi",
            "address": contract_address,
        }
        
        response = self._make_request(params)
        
        if not response or "error" in response:
            return response
        
        return {
            "abi": response.get("result"),
            "fetch_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_eth_balance(self, address: str) -> Optional[Dict]:
        """
        Get ETH balance for an address
        
        Args:
            address: Ethereum address
            
        Returns:
            Dictionary with balance or error information
        """
        params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest"
        }
        
        response = self._make_request(params)
        
        if not response or "error" in response:
            return response
        
        balance_wei = int(response.get("result", "0"))
        balance_eth = balance_wei / 10**18
        
        return {
            "balance_wei": balance_wei,
            "balance_eth": balance_eth,
            "fetch_timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def is_available(self) -> bool:
        """
        Check if Etherscan API is available and working
        
        Returns:
            True if API is available, False otherwise
        """
        if not self.api_key:
            return False
        
        try:
            # Test with a simple balance check
            params = {
                "module": "account",
                "action": "balance",
                "address": "0x0000000000000000000000000000000000000000",
                "tag": "latest"
            }
            
            response = self._make_request(params)
            return response is not None and "error" not in response
            
        except Exception:
            return False


# Convenience functions for backward compatibility
def create_etherscan_client(api_key: Optional[str] = None) -> EtherscanClient:
    """Create Etherscan client with optional API key"""
    return EtherscanClient(api_key=api_key)


def fetch_address_transactions(address: str, days: int = 365, api_key: Optional[str] = None) -> Optional[Dict]:
    """Convenience function to fetch transactions for an address"""
    client = EtherscanClient(api_key=api_key)
    return client.fetch_transactions(address, days)