#!/usr/bin/env python3
"""
Rocket Pool Groups Analyzer
Анализирует группы адресов, предположительно принадлежащих одному пользователю,
с полной статистикой за 365 дней через API.

ЗАПУСК:
cd mintegrity
python scripts/stats_vis/rocket_pool_groups_analyzer.py

Функциональность:
1. Загружает существующий граф Rocket Pool
2. Определяет группы координированных адресов
3. Получает полную статистику каждого адреса за 365 дней:
   - Объемы транзакций в USD (с историческими ценами)
   - Gas fees и средние цены газа
   - Возраст кошельков и даты создания
   - Паттерны активности (дни, месяцы)
   - Взаимодействия с кошельками и контрактами
4. Агрегирует статистику групп и создает сравнения
5. Создает детальные визуализации и HTML отчеты
6. Сохраняет результаты в JSON и CSV форматах

ТРЕБОВАНИЯ:
• ETHERSCAN_API_KEY в .env файле
• Интернет-соединение для API запросов
• files/rocket_pool_full_graph_90_days.json
"""

import sys
import os
import json
import csv
import pandas as pd
import numpy as np
import time
import requests
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# Настройка matplotlib для headless серверов
import matplotlib
matplotlib.use('Agg')  # Использовать backend без GUI
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging

try:
    from tqdm import tqdm
except ImportError:
    # Создаем заглушку для tqdm если не установлен
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None):
            self.iterable = iterable
            self.total = total
            self.n = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n
        def __iter__(self):
            return iter(self.iterable)

# Добавляем путь к корневой директории проекта
current_file = Path(__file__).resolve()
scripts_dir = current_file.parent.parent  # scripts/
project_root = scripts_dir.parent  # mintegrity/
sys.path.insert(0, str(project_root))

# Настройка логирования (СНАЧАЛА!)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Загружаем переменные окружения из .env файла
def load_env_file():
    """Загружает переменные из .env файла"""
    env_file = project_root / ".env"
    
    if env_file.exists():
        try:
            # Пробуем использовать python-dotenv если доступен
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
                return True
            except ImportError:
                # Если python-dotenv не установлен, читаем файл вручную
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            os.environ[key] = value
                return True
        except Exception as e:
            log.warning(f"Failed to load .env file: {e}")
            return False
    
    return False

# Загружаем .env файл при старте
load_env_file()

warnings.filterwarnings('ignore')
plt.style.use('default')

# Импорты из существующих модулей проекта
try:
    from scripts.graph.util.transactions_graph_json import load_graph_from_json
    from scripts.graph.analysis.wallet_groups.wallet_grouping import (
        detect_wallet_coordination, 
        identify_wallet_groups
    )
    log.info("Successfully imported project modules")
except ImportError as e:
    log.error(f"Could not import project modules: {e}")
    log.error("Make sure you are running from the mintegrity directory and all modules exist")
    sys.exit(1)

# Попытка импорта дополнительных модулей для полного анализа
try:
    from scripts.commons import metadata
    from scripts.commons.tokens_metadata_scraper import fetch_current_token_prices
    from scripts.commons.known_token_list import ETH_TOKENS_WHITELIST
    FULL_ANALYSIS_AVAILABLE = True
    log.info("✅ Full API analysis modules available")
except ImportError as e:
    log.warning(f"⚠️  Some API analysis modules not available: {e}")
    log.warning("Will use basic graph-based analysis")
    FULL_ANALYSIS_AVAILABLE = False

@dataclass
class WalletStatistics:
    """Статистика адреса за 365 дней"""
    address: str
    address_type: str = None
    total_volume_usd_365d: Optional[float] = None
    total_transactions_365d: Optional[int] = None
    outgoing_transactions_365d: Optional[int] = None
    incoming_transactions_365d: Optional[int] = None
    wallet_age_days: Optional[int] = None
    active_days_365d: Optional[int] = None
    most_active_month_365d: Optional[str] = None
    total_gas_fees_usd_365d: Optional[float] = None
    unique_addresses_interacted_365d: Optional[int] = None
    average_volume_usd_365d: Optional[float] = None
    max_volume_usd_365d: Optional[float] = None
    median_volume_usd_365d: Optional[float] = None
    wallet_interactions_365d: Optional[int] = None
    contract_interactions_365d: Optional[int] = None
    avg_daily_volume_usd_365d: Optional[float] = None
    max_daily_volume_usd_365d: Optional[float] = None
    total_gas_used_365d: Optional[int] = None
    avg_gas_price_gwei_365d: Optional[float] = None
    creation_date: Optional[str] = None
    first_transaction_date: Optional[str] = None
    last_transaction_date: Optional[str] = None
    token_prices_used: Optional[Dict[str, float]] = None
    error: Optional[str] = None

@dataclass
class GroupStatistics:
    """Статистика группы адресов за 365 дней"""
    group_id: int
    group_size: int
    addresses: List[str]
    
    # Агрегированная статистика за 365 дней
    total_volume_usd_365d: float = 0.0
    total_transactions_365d: int = 0
    total_outgoing_transactions_365d: int = 0
    total_incoming_transactions_365d: int = 0
    
    # Средние значения
    avg_volume_per_address_365d: float = 0.0
    avg_transactions_per_address_365d: float = 0.0
    
    # Максимальные значения в группе
    max_volume_in_group_365d: float = 0.0
    max_transactions_in_group_365d: int = 0
    
    # Возраст и активность
    oldest_wallet_age_days: Optional[int] = None
    newest_wallet_age_days: Optional[int] = None
    avg_wallet_age_days: Optional[float] = None
    
    # Паттерны группы
    total_active_days_365d: int = 0
    unique_months_active: int = 0
    coordination_score_avg: float = 0.0
    
    # Распределение ролей в группе
    layer_wallets_count: int = 0
    storage_wallets_count: int = 0
    regular_wallets_count: int = 0
    contracts_count: int = 0
    
    # Gas и fees
    total_gas_fees_usd_365d: float = 0.0
    avg_gas_fees_per_address_365d: float = 0.0
    
    # Взаимодействия
    internal_transfers_count: int = 0  # Переводы внутри группы
    external_unique_addresses: int = 0  # Уникальные внешние адреса
    
    # Дополнительная информация
    distance_to_root: Optional[int] = None
    error_addresses: List[str] = None

# === Встроенный анализатор адресов ===
class BuiltInAddressAnalyzer:
    """Встроенный анализатор адресов с полной функциональностью API"""

    def __init__(self, max_workers: int = 5):
        global FULL_ANALYSIS_AVAILABLE
        self.max_workers = max_workers
        self.price_cache = {}  # Кеш для исторических цен
        self.current_token_prices = {}
        
        if FULL_ANALYSIS_AVAILABLE:
            try:
                # Инициализируем metadata
                metadata.init()
                self.current_token_prices = self._fetch_current_prices()
                log.info(f"Loaded fallback prices for {len(self.current_token_prices)} tokens")
            except Exception as e:
                log.warning(f"Failed to initialize pricing: {e}")
                FULL_ANALYSIS_AVAILABLE = False

    def _fetch_current_prices(self) -> Dict[str, float]:
        """Получает текущие цены токенов"""
        try:
            # Используем существующий модуль для получения цен
            token_prices_with_timestamps = fetch_current_token_prices(ETH_TOKENS_WHITELIST)
            
            current_prices = {}
            for token, (timestamp, price) in token_prices_with_timestamps.items():
                current_prices[token] = price
            
            return current_prices
            
        except Exception as e:
            log.warning(f"Failed to fetch current prices via API: {e}")
            
            # Fallback: используем цены из metadata
            fallback_prices = {}
            for token in ETH_TOKENS_WHITELIST:
                try:
                    price = metadata.get_token_price_usd(token, str(int(time.time())))
                    if price > 0:
                        fallback_prices[token] = price
                except:
                    pass
            
            return fallback_prices

    def get_historical_token_price(self, token_symbol: str, timestamp: int) -> float:
        """Получает историческую цену токена"""
        cache_key = f"{token_symbol.upper()}-{timestamp}"
        
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        # Сначала пробуем metadata
        try:
            price = metadata.get_token_price_usd(token_symbol, str(timestamp))
            if price > 0:
                self.price_cache[cache_key] = price
                return price
        except Exception:
            pass
        
        # Внешний API (Coinbase)
        try:
            token_to_pair = {
                'ETH': 'ETH-USD', 'BTC': 'BTC-USD', 'WETH': 'ETH-USD',
                'USDT': 'USDT-USD', 'USDC': 'USDC-USD', 'DAI': 'DAI-USD',
                'LINK': 'LINK-USD', 'UNI': 'UNI-USD', 'AAVE': 'AAVE-USD'
            }
            
            pair = token_to_pair.get(token_symbol.upper(), 'ETH-USD')
            
            start_time = timestamp - 3600
            end_time = timestamp + 3600
            
            url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
            params = {
                'start': datetime.fromtimestamp(start_time, timezone.utc).isoformat(),
                'end': datetime.fromtimestamp(end_time, timezone.utc).isoformat(),
                'granularity': 3600
            }
            
            import requests
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            candles = response.json()
            if candles:
                closest_candle = min(candles, key=lambda x: abs(x[0] - timestamp))
                price = float(closest_candle[4])  # close price
                self.price_cache[cache_key] = price
                return price
                
        except Exception:
            pass
        
        # Fallback: текущая цена
        return self.current_token_prices.get(token_symbol.upper(), 
               self.current_token_prices.get('ETH', 2500.0))

    def get_wallet_statistics_etherscan(self, address: str, address_type: str) -> WalletStatistics:
        """Получает статистику через Etherscan API"""
        import os
        import requests
        
        etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
        if not etherscan_api_key:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                error="ETHERSCAN_API_KEY not set"
            )
        
        try:
            # Получаем транзакции за 365 дней
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=365)
            
            url = "https://api.etherscan.io/api"
            params = {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": 0,
                "endblock": 99999999,
                "page": 1,
                "offset": 10000,
                "sort": "asc",
                "apikey": etherscan_api_key
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data["status"] != "1":
                error_msg = data.get('message', 'Unknown error')
                return WalletStatistics(
                    address=address,
                    address_type=address_type,
                    error=f"Etherscan API error: {error_msg}"
                )
            
            transactions = data["result"]
            if not transactions:
                return self._create_empty_stats(address, address_type)
            
            # Фильтруем за 365 дней
            start_timestamp = int(start_time.timestamp())
            filtered_transactions = [
                tx for tx in transactions 
                if int(tx["timeStamp"]) >= start_timestamp
            ]
            
            if not filtered_transactions:
                return self._create_empty_stats(address, address_type)
            
            return self._analyze_transactions(address, address_type, filtered_transactions, transactions)
            
        except Exception as e:
            return WalletStatistics(
                address=address,
                address_type=address_type,
                error=f"API error: {str(e)}"
            )

    def _analyze_transactions(self, address: str, address_type: str, 
                            transactions_365d: List[Dict], all_transactions: List[Dict]) -> WalletStatistics:
        """Анализирует транзакции"""
        
        volumes_usd = []
        outgoing_txs = []
        incoming_txs = []
        daily_volumes = {}
        monthly_volumes = {}
        gas_used_total = 0
        gas_fees_usd_total = 0.0
        unique_addresses = set()
        
        for tx in transactions_365d:
            value_wei = int(tx["value"])
            value_eth = value_wei / 10**18
            timestamp = int(tx["timeStamp"])
            from_addr = tx["from"].lower()
            to_addr = tx["to"].lower()
            
            is_outgoing = from_addr == address.lower()
            is_incoming = to_addr == address.lower()
            
            if is_outgoing:
                outgoing_txs.append(tx)
            if is_incoming:
                incoming_txs.append(tx)
            
            # Анализируем исходящие транзакции
            if is_outgoing and value_eth > 0:
                # Получаем цену ETH на момент транзакции
                eth_price = self.get_historical_token_price('ETH', timestamp)
                value_usd = value_eth * eth_price
                volumes_usd.append(value_usd)
                
                # Дневная и месячная активность
                tx_date = datetime.fromtimestamp(timestamp, timezone.utc).date()
                month_key = tx_date.strftime('%Y-%m')
                
                daily_volumes[tx_date] = daily_volumes.get(tx_date, 0) + value_usd
                monthly_volumes[month_key] = monthly_volumes.get(month_key, 0) + value_usd
            
            # Gas анализ
            if is_outgoing:
                gas_used = int(tx.get("gasUsed", 0))
                gas_price = int(tx.get("gasPrice", 0))
                gas_used_total += gas_used
                
                gas_fee_eth = (gas_used * gas_price) / 10**18
                eth_price = self.get_historical_token_price('ETH', timestamp)
                gas_fees_usd_total += gas_fee_eth * eth_price
            
            # Уникальные адреса
            other_address = to_addr if is_outgoing else from_addr
            unique_addresses.add(other_address)
        
        # Возраст кошелька
        all_timestamps = [int(tx["timeStamp"]) for tx in all_transactions]
        first_timestamp = min(all_timestamps) if all_timestamps else None
        
        first_date = None
        wallet_age_days = None
        if first_timestamp:
            first_date = datetime.fromtimestamp(first_timestamp, timezone.utc)
            wallet_age_days = (datetime.now(timezone.utc) - first_date).days
        
        # Статистики
        total_volume = sum(volumes_usd)
        avg_volume = total_volume / len(volumes_usd) if volumes_usd else 0
        max_volume = max(volumes_usd) if volumes_usd else 0
        median_volume = sorted(volumes_usd)[len(volumes_usd)//2] if volumes_usd else 0
        
        active_days = len(daily_volumes)
        avg_daily_volume = sum(daily_volumes.values()) / len(daily_volumes) if daily_volumes else 0
        max_daily_volume = max(daily_volumes.values()) if daily_volumes else 0
        most_active_month = max(monthly_volumes.items(), key=lambda x: x[1])[0] if monthly_volumes else None
        
        # Gas статистики
        avg_gas_price_gwei = 0.0
        if outgoing_txs:
            total_gas_price_wei = sum(int(tx.get("gasPrice", 0)) for tx in outgoing_txs)
            avg_gas_price_gwei = (total_gas_price_wei / len(outgoing_txs)) / 10**9
        
        return WalletStatistics(
            address=address,
            address_type=address_type,
            creation_date=first_date.isoformat() if first_date else None,
            first_transaction_date=first_date.isoformat() if first_date else None,
            last_transaction_date=datetime.fromtimestamp(max([int(tx["timeStamp"]) for tx in transactions_365d]), timezone.utc).isoformat(),
            wallet_age_days=wallet_age_days,
            total_volume_usd_365d=round(total_volume, 2),
            average_volume_usd_365d=round(avg_volume, 2),
            max_volume_usd_365d=round(max_volume, 2),
            median_volume_usd_365d=round(median_volume, 2),
            total_transactions_365d=len(transactions_365d),
            outgoing_transactions_365d=len(outgoing_txs),
            incoming_transactions_365d=len(incoming_txs),
            unique_addresses_interacted_365d=len(unique_addresses),
            active_days_365d=active_days,
            avg_daily_volume_usd_365d=round(avg_daily_volume, 2),
            max_daily_volume_usd_365d=round(max_daily_volume, 2),
            most_active_month_365d=most_active_month,
            total_gas_used_365d=gas_used_total,
            total_gas_fees_usd_365d=round(gas_fees_usd_total, 2),
            avg_gas_price_gwei_365d=round(avg_gas_price_gwei, 2),
            wallet_interactions_365d=0,  # Упрощено
            contract_interactions_365d=0,  # Упрощено
            token_prices_used={'ETH': self.current_token_prices.get('ETH', 0)}
        )

    def _create_empty_stats(self, address: str, address_type: str) -> WalletStatistics:
        """Создает пустую статистику"""
        return WalletStatistics(
            address=address,
            address_type=address_type,
            total_volume_usd_365d=0.0,
            total_transactions_365d=0,
            outgoing_transactions_365d=0,
            incoming_transactions_365d=0
        )

    def analyze_addresses_batch(self, addresses: List[str], graph) -> List[WalletStatistics]:
        """Анализирует пакет адресов"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_address = {}
            
            for address in addresses:
                # Определяем тип адреса из графа
                address_type = "wallet"
                if address in graph.nodes:
                    node = graph.nodes[address]
                    if hasattr(node, 'type') and str(node.type).upper() == "CONTRACT":
                        address_type = "contract"
                
                future = executor.submit(self.get_wallet_statistics_etherscan, address, address_type)
                future_to_address[future] = address
            
            with tqdm(total=len(addresses), desc="Analyzing addresses") as pbar:
                for future in as_completed(future_to_address):
                    pbar.update(1)
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        address = future_to_address[future]
                        log.warning(f"Failed to analyze {address}: {e}")
                        results.append(WalletStatistics(
                            address=address,
                            address_type="unknown",
                            error=str(e)
                        ))
        
        return results

class RocketPoolGroupsAnalyzer:
    """Анализатор групп адресов Rocket Pool"""
    
    def __init__(self, 
                 graph_file_path: str = "files/rocket_pool_full_graph_90_days.json",
                 addresses_file_path: Optional[str] = None,
                 output_dir: str = "files/rocket_pool_groups_analysis",
                 coordination_threshold: float = 5.0,
                 min_group_size: int = 2,
                 max_workers: int = 5):
        
        # Обрабатываем пути - если относительные, делаем их относительно корня проекта
        if not Path(graph_file_path).is_absolute():
            self.graph_file_path = project_root / graph_file_path
        else:
            self.graph_file_path = Path(graph_file_path)
            
        if addresses_file_path:
            if not Path(addresses_file_path).is_absolute():
                self.addresses_file_path = project_root / addresses_file_path
            else:
                self.addresses_file_path = Path(addresses_file_path)
        else:
            self.addresses_file_path = None
            
        if not Path(output_dir).is_absolute():
            self.output_dir = project_root / output_dir
        else:
            self.output_dir = Path(output_dir)
            
        self.coordination_threshold = coordination_threshold
        self.min_group_size = min_group_size
        self.max_workers = max_workers
        
        # Создаем директории
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        self.graph = None
        self.wallet_groups = []
        self.individual_stats = {}  # Статистика отдельных адресов
        self.group_stats = []  # Статистика групп
        
        log.info(f"Initialized Rocket Pool Groups Analyzer")
        log.info(f"Graph file: {self.graph_file_path}")
        log.info(f"Output directory: {self.output_dir}")
        log.info(f"Coordination threshold: {coordination_threshold}")
        log.info(f"Minimum group size: {min_group_size}")
        
        # Показываем доступные возможности анализа
        if FULL_ANALYSIS_AVAILABLE:
            log.info("🚀 Full API analysis available (365-day detailed statistics via Etherscan + historical prices)")
        else:
            log.info("📊 Basic analysis available (graph-based statistics only)")
        
        if self.addresses_file_path:
            log.info(f"📁 Will use existing addresses file: {self.addresses_file_path}")

    def load_graph(self):
        """Загружает граф из файла"""
        if not self.graph_file_path.exists():
            log.error(f"Graph file not found: {self.graph_file_path}")
            log.error("Please ensure the graph file exists or provide correct path with --graph-path")
            sys.exit(1)
            
        log.info(f"Loading graph from {self.graph_file_path}")
        self.graph = load_graph_from_json(str(self.graph_file_path))
        log.info(f"Successfully loaded graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

    def detect_wallet_groups(self):
        """Определяет группы координированных адресов"""
        log.info("Detecting coordinated wallet groups...")
        
        # Используем функциональность из wallet_grouping
        coordination_scores = detect_wallet_coordination(self.graph)
        
        # Импортируем wallet_metrics из модуля
        from scripts.graph.analysis.wallet_groups.wallet_grouping import wallet_metrics
        
        self.wallet_groups = identify_wallet_groups(
            coordination_scores, 
            wallet_metrics,
            threshold=self.coordination_threshold
        )
        
        # Фильтруем группы по минимальному размеру
        self.wallet_groups = [group for group in self.wallet_groups if len(group) >= self.min_group_size]
        
        log.info(f"Found {len(self.wallet_groups)} groups with {self.min_group_size}+ addresses")
        for i, group in enumerate(self.wallet_groups):
            log.info(f"Group {i+1}: {len(group)} addresses")

    def load_or_analyze_individual_addresses(self):
        """Загружает или анализирует статистику отдельных адресов"""
        
        # Проверяем, есть ли готовый файл со статистикой
        if self.addresses_file_path and self.addresses_file_path.exists():
            log.info(f"Loading existing addresses analysis from {self.addresses_file_path}")
            
            with open(self.addresses_file_path, 'r') as f:
                addresses_data = json.load(f)
            
            # Конвертируем в словарь для быстрого поиска
            for addr_data in addresses_data:
                if not addr_data.get('error'):
                    self.individual_stats[addr_data['address']] = WalletStatistics(**addr_data)
            
            log.info(f"Loaded statistics for {len(self.individual_stats)} addresses")
            return
        
        # Если файла нет, используем встроенный анализатор для полного анализа
        if FULL_ANALYSIS_AVAILABLE:
            log.info("No existing addresses file found. Performing full 365-day analysis via APIs...")
            self._analyze_addresses_with_full_stats()
        else:
            log.warning("Full analysis modules not available. Creating simplified statistics from graph data...")
            self._create_mock_statistics_from_graph()

    def _analyze_addresses_with_full_stats(self):
        """Анализирует адреса с полной статистикой через встроенный анализатор"""
        
        # Собираем все адреса из групп
        all_group_addresses = set()
        for group in self.wallet_groups:
            all_group_addresses.update(group)
        
        if not all_group_addresses:
            log.warning("No addresses found in groups")
            return
        
        log.info(f"Starting full 365-day analysis for {len(all_group_addresses)} addresses from groups...")
        log.info("This will fetch detailed statistics via Etherscan API with historical prices")
        log.info("This may take several minutes depending on the number of addresses...")
        
        # Создаем встроенный анализатор
        analyzer = BuiltInAddressAnalyzer(max_workers=self.max_workers)
        
        try:
            # Анализируем адреса пакетами
            addresses_list = list(all_group_addresses)
            batch_size = 50
            all_results = []
            
            for i in range(0, len(addresses_list), batch_size):
                batch = addresses_list[i:i + batch_size]
                log.info(f"Processing batch {i//batch_size + 1}/{(len(addresses_list) + batch_size - 1)//batch_size}")
                
                batch_results = analyzer.analyze_addresses_batch(batch, self.graph)
                all_results.extend(batch_results)
                
                # Пауза между пакетами
                if i + batch_size < len(addresses_list):
                    time.sleep(1)
            
            # Конвертируем результаты
            for result in all_results:
                if not result.error:
                    self.individual_stats[result.address] = result
                else:
                    log.warning(f"Failed to analyze {result.address}: {result.error}")
            
            # Сохраняем детальные результаты
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_file = self.output_dir / "data" / f"detailed_addresses_analysis_{timestamp}.json"
            
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json_data = [asdict(result) for result in all_results]
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            success_count = len([r for r in all_results if not r.error])
            log.info(f"Completed detailed analysis: {success_count}/{len(all_results)} addresses successfully analyzed")
            log.info(f"Detailed results saved to: {detailed_file}")
            
        except Exception as e:
            log.error(f"Failed to perform detailed analysis: {e}")
            log.warning("Falling back to simplified graph-based statistics...")
            self._create_mock_statistics_from_graph()

    def _create_mock_statistics_from_graph(self):
        """Создает упрощенную статистику из данных графа"""
        all_group_addresses = set()
        for group in self.wallet_groups:
            all_group_addresses.update(group)
        
        for address in all_group_addresses:
            # Простой подсчет из графа
            total_transactions = 0
            total_volume = 0.0
            
            # Считаем исходящие транзакции
            for (from_addr, to_addr), edge in self.graph.edges.items():
                if from_addr == address:
                    total_transactions += len(edge.transactions)
                    for tx in edge.transactions.values():
                        total_volume += tx.value_usd
            
            # Определяем тип адреса
            address_type = "wallet"
            if address in self.graph.nodes:
                node = self.graph.nodes[address]
                # Проверяем тип узла
                if hasattr(node, 'type'):
                    if hasattr(node.type, 'name'):
                        address_type = node.type.name.lower()
                    elif str(node.type).upper() == "CONTRACT":
                        address_type = "contract"
            
            self.individual_stats[address] = WalletStatistics(
                address=address,
                address_type=address_type,
                total_volume_usd_365d=total_volume,
                total_transactions_365d=total_transactions,
                outgoing_transactions_365d=total_transactions,
                incoming_transactions_365d=0,
                wallet_age_days=None,
                active_days_365d=None,
                most_active_month_365d=None,
                total_gas_fees_usd_365d=0.0,
                unique_addresses_interacted_365d=None
            )
        
        log.info(f"Created mock statistics for {len(self.individual_stats)} addresses")

    def calculate_group_statistics(self):
        """Рассчитывает агрегированную статистику для каждой группы"""
        log.info("Calculating group statistics...")
        
        self.group_stats = []
        
        for group_id, group_addresses in enumerate(self.wallet_groups, 1):
            log.info(f"Processing group {group_id} with {len(group_addresses)} addresses")
            
            # Фильтруем адреса, для которых есть статистика
            valid_addresses = []
            error_addresses = []
            
            for addr in group_addresses:
                if addr in self.individual_stats:
                    valid_addresses.append(addr)
                else:
                    error_addresses.append(addr)
            
            if not valid_addresses:
                log.warning(f"No valid statistics for group {group_id}")
                continue
            
            # Агрегируем статистику
            group_stat = self._aggregate_group_statistics(
                group_id, list(group_addresses), valid_addresses, error_addresses
            )
            
            self.group_stats.append(group_stat)
        
        log.info(f"Calculated statistics for {len(self.group_stats)} groups")

    def _aggregate_group_statistics(self, group_id: int, all_addresses: List[str], 
                                   valid_addresses: List[str], error_addresses: List[str]) -> GroupStatistics:
        """Агрегирует статистику для одной группы"""
        
        # Получаем статистику для валидных адресов
        stats_list = [self.individual_stats[addr] for addr in valid_addresses]
        
        # Базовая информация
        group_size = len(all_addresses)
        
        # Агрегированные объемы и транзакции
        total_volume = sum(s.total_volume_usd_365d or 0 for s in stats_list)
        total_transactions = sum(s.total_transactions_365d or 0 for s in stats_list)
        total_outgoing = sum(s.outgoing_transactions_365d or 0 for s in stats_list)
        total_incoming = sum(s.incoming_transactions_365d or 0 for s in stats_list)
        
        # Средние значения
        avg_volume_per_address = total_volume / len(valid_addresses) if valid_addresses else 0
        avg_transactions_per_address = total_transactions / len(valid_addresses) if valid_addresses else 0
        
        # Максимальные значения
        max_volume = max((s.total_volume_usd_365d or 0 for s in stats_list), default=0)
        max_transactions = max((s.total_transactions_365d or 0 for s in stats_list), default=0)
        
        # Возраст кошельков
        ages = [s.wallet_age_days for s in stats_list if s.wallet_age_days]
        oldest_age = max(ages) if ages else None
        newest_age = min(ages) if ages else None
        avg_age = sum(ages) / len(ages) if ages else None
        
        # Активность
        total_active_days = sum(s.active_days_365d or 0 for s in stats_list)
        
        # Подсчет уникальных месяцев активности
        unique_months = set()
        for s in stats_list:
            if s.most_active_month_365d:
                unique_months.add(s.most_active_month_365d)
        
        # Распределение типов кошельков
        contracts_count = sum(1 for s in stats_list if s.address_type == "contract")
        regular_count = len(stats_list) - contracts_count
        
        # Gas fees
        total_gas_fees = sum(s.total_gas_fees_usd_365d or 0 for s in stats_list)
        avg_gas_fees = total_gas_fees / len(valid_addresses) if valid_addresses else 0
        
        # Внутренние переводы
        internal_transfers = self._count_internal_transfers(valid_addresses)
        
        # Внешние взаимодействия
        external_addresses = set()
        for s in stats_list:
            if s.unique_addresses_interacted_365d:
                external_addresses.add(s.address)
        
        return GroupStatistics(
            group_id=group_id,
            group_size=group_size,
            addresses=all_addresses,
            total_volume_usd_365d=total_volume,
            total_transactions_365d=total_transactions,
            total_outgoing_transactions_365d=total_outgoing,
            total_incoming_transactions_365d=total_incoming,
            avg_volume_per_address_365d=avg_volume_per_address,
            avg_transactions_per_address_365d=avg_transactions_per_address,
            max_volume_in_group_365d=max_volume,
            max_transactions_in_group_365d=max_transactions,
            oldest_wallet_age_days=oldest_age,
            newest_wallet_age_days=newest_age,
            avg_wallet_age_days=avg_age,
            total_active_days_365d=total_active_days,
            unique_months_active=len(unique_months),
            regular_wallets_count=regular_count,
            contracts_count=contracts_count,
            total_gas_fees_usd_365d=total_gas_fees,
            avg_gas_fees_per_address_365d=avg_gas_fees,
            internal_transfers_count=internal_transfers,
            external_unique_addresses=len(external_addresses),
            error_addresses=error_addresses
        )

    def _count_internal_transfers(self, addresses: List[str]) -> int:
        """Подсчитывает количество переводов внутри группы"""
        internal_count = 0
        address_set = set(addresses)
        
        # Анализируем рёбра графа
        for (from_addr, to_addr), edge in self.graph.edges.items():
            if from_addr in address_set and to_addr in address_set:
                internal_count += len(edge.transactions)
        
        return internal_count

    def create_group_volume_distribution(self):
        """Создает распределение групп по объемам"""
        log.info("Creating group volume distribution...")
        
        if not self.group_stats:
            log.warning("No group statistics available")
            return
        
        volumes = [group.total_volume_usd_365d for group in self.group_stats]
        
        # Определяем бины для групп
        bins = [
            (0, 10_000, "$0-$10K"),
            (10_000, 100_000, "$10K-$100K"),
            (100_000, 1_000_000, "$100K-$1M"),
            (1_000_000, 10_000_000, "$1M-$10M"),
            (10_000_000, float('inf'), "$10M+")
        ]
        
        bin_counts, bin_labels = self._calculate_bins(volumes, bins)
        
        # Создаем график
        plt.figure(figsize=(12, 8))
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        bars = plt.bar(bin_labels, bin_counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, bin_counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bin_counts) * 0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.title('Distribution of Groups by Total Volume (365 days)', fontsize=16, fontweight='bold')
        plt.xlabel('Group Volume Range (USD)', fontsize=12)
        plt.ylabel('Number of Groups', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        total = len(self.group_stats)
        plt.figtext(0.02, 0.98, f'Total Groups: {total}', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "group_volume_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_group_size_analysis(self):
        """Создает анализ размеров групп"""
        log.info("Creating group size analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Group Size Analysis', fontsize=16, fontweight='bold')
        
        # 1. Распределение по размерам групп
        group_sizes = [group.group_size for group in self.group_stats]
        unique_sizes = sorted(set(group_sizes))
        size_counts = [group_sizes.count(size) for size in unique_sizes]
        
        axes[0, 0].bar(unique_sizes, size_counts, color='#3498db', alpha=0.8, edgecolor='black')
        axes[0, 0].set_title('Distribution by Group Size')
        axes[0, 0].set_xlabel('Group Size (addresses)')
        axes[0, 0].set_ylabel('Number of Groups')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Объем vs размер группы
        volumes = [group.total_volume_usd_365d for group in self.group_stats]
        axes[0, 1].scatter(group_sizes, volumes, color='#e74c3c', alpha=0.7, s=50)
        axes[0, 1].set_title('Group Volume vs Size')
        axes[0, 1].set_xlabel('Group Size (addresses)')
        axes[0, 1].set_ylabel('Total Volume (USD)')
        if any(v > 0 for v in volumes):
            axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Средний объем на адрес в группе
        avg_volumes = [group.avg_volume_per_address_365d for group in self.group_stats]
        axes[1, 0].scatter(group_sizes, avg_volumes, color='#2ecc71', alpha=0.7, s=50)
        axes[1, 0].set_title('Average Volume per Address vs Group Size')
        axes[1, 0].set_xlabel('Group Size (addresses)')
        axes[1, 0].set_ylabel('Average Volume per Address (USD)')
        if any(v > 0 for v in avg_volumes):
            axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Эффективность групп
        efficiency = np.array(avg_volumes)
        if any(efficiency > 0):
            axes[1, 1].hist(efficiency[efficiency > 0], bins=15, color='#f39c12', alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Distribution of Group Efficiency')
            axes[1, 1].set_xlabel('Average Volume per Address (USD)')
            axes[1, 1].set_ylabel('Number of Groups')
            axes[1, 1].set_xscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'No volume data available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "group_size_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_top_groups_analysis(self):
        """Создает анализ топ групп"""
        log.info("Creating top groups analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Top Groups Analysis', fontsize=16, fontweight='bold')
        
        n_top = min(10, len(self.group_stats))
        
        # 1. Топ по объему
        top_volume = sorted(self.group_stats, key=lambda x: x.total_volume_usd_365d, reverse=True)[:n_top]
        group_ids = [f"Group {g.group_id}\n({g.group_size} addr)" for g in top_volume]
        volumes = [g.total_volume_usd_365d for g in top_volume]
        
        y_pos = np.arange(len(top_volume))
        axes[0, 0].barh(y_pos, volumes, color='#e74c3c', alpha=0.8)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(group_ids, fontsize=9)
        axes[0, 0].set_xlabel('Total Volume (USD)')
        axes[0, 0].set_title(f'Top {n_top} Groups by Volume')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Топ по количеству транзакций
        top_tx = sorted(self.group_stats, key=lambda x: x.total_transactions_365d, reverse=True)[:n_top]
        group_ids_tx = [f"Group {g.group_id}\n({g.group_size} addr)" for g in top_tx]
        transactions = [g.total_transactions_365d for g in top_tx]
        
        y_pos = np.arange(len(top_tx))
        axes[0, 1].barh(y_pos, transactions, color='#3498db', alpha=0.8)
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(group_ids_tx, fontsize=9)
        axes[0, 1].set_xlabel('Total Transactions')
        axes[0, 1].set_title(f'Top {n_top} Groups by Transactions')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Эффективность групп
        top_efficiency = sorted(self.group_stats, key=lambda x: x.avg_volume_per_address_365d, reverse=True)[:n_top]
        group_ids_eff = [f"Group {g.group_id}\n({g.group_size} addr)" for g in top_efficiency]
        avg_volumes = [g.avg_volume_per_address_365d for g in top_efficiency]
        
        y_pos = np.arange(len(top_efficiency))
        axes[1, 0].barh(y_pos, avg_volumes, color='#2ecc71', alpha=0.8)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(group_ids_eff, fontsize=9)
        axes[1, 0].set_xlabel('Average Volume per Address (USD)')
        axes[1, 0].set_title(f'Top {n_top} Most Efficient Groups')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Внутренняя координация групп
        coordination_data = [(g.group_id, g.group_size, g.internal_transfers_count) for g in self.group_stats]
        coordination_data.sort(key=lambda x: x[2], reverse=True)
        
        if coordination_data:
            top_coord = coordination_data[:n_top]
            group_ids_coord = [f"Group {g[0]}\n({g[1]} addr)" for g in top_coord]
            internal_transfers = [g[2] for g in top_coord]
            
            y_pos = np.arange(len(top_coord))
            axes[1, 1].barh(y_pos, internal_transfers, color='#9b59b6', alpha=0.8)
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(group_ids_coord, fontsize=9)
            axes[1, 1].set_xlabel('Internal Transfers Count')
            axes[1, 1].set_title(f'Top {n_top} Groups by Internal Coordination')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "top_groups_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_groups_vs_individuals_comparison(self):
        """Создает сравнение групп vs индивидуальных адресов"""
        log.info("Creating groups vs individuals comparison...")
        
        # Получаем адреса, которые НЕ входят в группы
        all_group_addresses = set()
        for group in self.wallet_groups:
            all_group_addresses.update(group)
        
        individual_addresses = []
        for addr, stats in self.individual_stats.items():
            if addr not in all_group_addresses:
                individual_addresses.append(stats)
        
        if not individual_addresses:
            log.warning("No individual addresses found for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Groups vs Individual Addresses Comparison', fontsize=16, fontweight='bold')
        
        # 1. Распределение объемов
        group_volumes = [g.total_volume_usd_365d for g in self.group_stats if g.total_volume_usd_365d > 0]
        individual_volumes = [s.total_volume_usd_365d for s in individual_addresses if s.total_volume_usd_365d and s.total_volume_usd_365d > 0]
        
        axes[0, 0].hist([group_volumes, individual_volumes], bins=20, label=['Groups', 'Individuals'], 
                       alpha=0.7, color=['red', 'blue'], edgecolor='black')
        axes[0, 0].set_xlabel('Total Volume (USD)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Volume Distribution: Groups vs Individuals')
        if group_volumes or individual_volumes:
            axes[0, 0].set_xscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot сравнение объемов
        data_to_plot = []
        labels = []
        if group_volumes:
            data_to_plot.append(group_volumes)
            labels.append('Groups')
        if individual_volumes:
            data_to_plot.append(individual_volumes)
            labels.append('Individuals')
        
        if data_to_plot:
            axes[0, 1].boxplot(data_to_plot, labels=labels)
            axes[0, 1].set_ylabel('Total Volume (USD)')
            axes[0, 1].set_title('Volume Distribution Comparison')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Средние значения по группам vs индивидуальным
        group_avg_volumes = [g.avg_volume_per_address_365d for g in self.group_stats if g.avg_volume_per_address_365d > 0]
        
        if group_avg_volumes and individual_volumes:
            axes[1, 0].hist([group_avg_volumes, individual_volumes], bins=15, 
                           label=['Groups (avg per address)', 'Individuals'], 
                           alpha=0.7, color=['orange', 'blue'], edgecolor='black')
            axes[1, 0].set_xlabel('Volume per Address (USD)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Volume per Address: Groups vs Individuals')
            axes[1, 0].set_xscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Статистика
        group_total_volume = sum(group_volumes) if group_volumes else 0
        individual_total_volume = sum(individual_volumes) if individual_volumes else 0
        total_addresses_in_groups = sum(g.group_size for g in self.group_stats)
        
        stats_text = f"""
        Groups Statistics:
        • Count: {len(self.group_stats)}
        • Total Volume: ${group_total_volume:,.0f}
        • Avg Volume per Group: ${group_total_volume/len(self.group_stats) if self.group_stats else 0:,.0f}
        
        Individual Statistics:
        • Count: {len(individual_addresses)}
        • Total Volume: ${individual_total_volume:,.0f}
        • Avg Volume per Individual: ${individual_total_volume/len(individual_addresses) if individual_addresses else 0:,.0f}
        
        Efficiency:
        • Groups control {group_total_volume/(group_total_volume+individual_total_volume)*100 if (group_total_volume+individual_total_volume) > 0 else 0:.1f}% of volume
        • With {total_addresses_in_groups}/{total_addresses_in_groups+len(individual_addresses)*100 if (total_addresses_in_groups+len(individual_addresses)) > 0 else 0:.1f}% of addresses
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "groups_vs_individuals_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_transaction_volume_bins_for_all_groups(self):
        """Создает bin plot по полному объёму транзакций для ВСЕХ групп"""
        log.info("Creating transaction volume bins for all groups...")
        
        if not self.group_stats:
            log.warning("No group statistics available")
            return
        
        # Получаем данные по объёму транзакций для всех групп
        transaction_volumes = [group.total_transactions_365d for group in self.group_stats]
        
        # Определяем бины по количеству транзакций
        bins = [
            (0, 100, "0-100 tx"),
            (100, 500, "100-500 tx"),
            (500, 1000, "500-1K tx"),
            (1000, 5000, "1K-5K tx"),
            (5000, 10000, "5K-10K tx"),
            (10000, float('inf'), "10K+ tx")
        ]
        
        bin_counts, bin_labels = self._calculate_bins(transaction_volumes, bins)
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Transaction Volume Distribution for All Groups (365 days)', fontsize=16, fontweight='bold')
        
        # 1. Bin chart
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6']
        bars = ax1.bar(bin_labels, bin_counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, bin_counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bin_counts) * 0.01,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        ax1.set_title('Groups Distribution by Transaction Count')
        ax1.set_xlabel('Transaction Count Range')
        ax1.set_ylabel('Number of Groups')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Cumulative distribution
        sorted_volumes = sorted(transaction_volumes, reverse=True)
        cumulative_percent = [(i+1)/len(sorted_volumes)*100 for i in range(len(sorted_volumes))]
        
        ax2.plot(sorted_volumes, cumulative_percent, 'o-', color='#e74c3c', linewidth=2, markersize=4)
        ax2.set_title('Cumulative Transaction Distribution')
        ax2.set_xlabel('Total Transactions (365d)')
        ax2.set_ylabel('Cumulative Percentage of Groups (%)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        # Добавляем статистику
        total_groups = len(self.group_stats)
        total_transactions = sum(transaction_volumes)
        avg_transactions = total_transactions / total_groups if total_groups > 0 else 0
        median_transactions = sorted(transaction_volumes)[len(transaction_volumes)//2] if transaction_volumes else 0
        
        stats_text = f"""
        Total Groups: {total_groups}
        Total Transactions: {total_transactions:,}
        Average per Group: {avg_transactions:,.0f}
        Median per Group: {median_transactions:,}
        """
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "transaction_volume_bins_all_groups.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Печатаем статистику в лог
        self._print_bins_stats("Transaction Volume", bin_labels, bin_counts, np.array(transaction_volumes))

    def _print_bins_stats(self, prefix, bin_labels, bin_counts, data):
        """Выводит статистику по бинам"""
        total = sum(bin_counts)
        log.info(f"📊 {prefix} distribution:")
        for label, count in zip(bin_labels, bin_counts):
            percentage = (count / total * 100) if total > 0 else 0
            log.info(f"  {label}: {count} groups ({percentage:.1f}%)")
        
        log.info(f"📈 {prefix} statistics:")
        log.info(f"  Total: {data.sum():,.0f}")
        log.info(f"  Average: {data.mean():,.0f}")
        log.info(f"  Median: {np.median(data):,.0f}")
        log.info(f"  Max: {data.max():,.0f}")
        log.info(f"  Min: {data.min():,.0f}")

    def generate_groups_report(self):
        """Генерирует базовую статистику по группам (без HTML отчета)"""
        log.info("Generating groups statistics summary...")
        
        if not self.group_stats:
            log.warning("No group statistics for report")
            return {}
        
        df_groups = pd.DataFrame([asdict(group) for group in self.group_stats])
        
        # Базовая статистика
        total_groups = len(self.group_stats)
        total_addresses_in_groups = df_groups['group_size'].sum()
        total_volume = df_groups['total_volume_usd_365d'].sum()
        total_transactions = df_groups['total_transactions_365d'].sum()
        avg_group_size = df_groups['group_size'].mean()
        avg_volume_per_group = df_groups['total_volume_usd_365d'].mean()
        
        # Эффективность групп
        largest_group = df_groups['group_size'].max()
        most_active_group_id = df_groups.loc[df_groups['total_transactions_365d'].idxmax(), 'group_id']
        highest_volume_group_id = df_groups.loc[df_groups['total_volume_usd_365d'].idxmax(), 'group_id']
        
        stats = {
            'total_groups': total_groups,
            'total_addresses_in_groups': total_addresses_in_groups,
            'avg_group_size': avg_group_size,
            'largest_group_size': largest_group,
            'total_volume': total_volume,
            'avg_volume_per_group': avg_volume_per_group,
            'total_transactions': total_transactions,
            'most_active_group_id': most_active_group_id,
            'highest_volume_group_id': highest_volume_group_id
        }
        
        # Выводим статистику в лог вместо HTML
        log.info("=" * 50)
        log.info("📊 GROUPS ANALYSIS SUMMARY")
        log.info("=" * 50)
        log.info(f"Total Groups: {stats['total_groups']}")
        log.info(f"Total Addresses in Groups: {stats['total_addresses_in_groups']}")
        log.info(f"Average Group Size: {stats['avg_group_size']:.1f}")
        log.info(f"Largest Group Size: {stats['largest_group_size']}")
        log.info(f"Total Volume: ${stats['total_volume']:,.0f}")
        log.info(f"Average Volume per Group: ${stats['avg_volume_per_group']:,.0f}")
        log.info(f"Total Transactions: {stats['total_transactions']:,.0f}")
        log.info(f"Most Active Group: {stats['most_active_group_id']}")
        log.info(f"Highest Volume Group: {stats['highest_volume_group_id']}")
        log.info("=" * 50)
        
        # Топ 5 групп в логе
        log.info("🏆 TOP 5 GROUPS BY VOLUME:")
        top_5_groups = df_groups.nlargest(5, 'total_volume_usd_365d')
        for i, (_, group) in enumerate(top_5_groups.iterrows(), 1):
            log.info(f"{i}. Group {group['group_id']} - {group['group_size']} addresses")
            log.info(f"   Volume: ${group['total_volume_usd_365d']:,.0f} | Transactions: {group['total_transactions_365d']:,}")
        
        log.info("=" * 50)
        
        return stats

    def run_full_analysis(self):
        """Запускает полный анализ групп"""
        log.info("=" * 60)
        log.info("ROCKET POOL GROUPS ANALYSIS STARTED")
        log.info("=" * 60)
        
        # Показываем информацию о типе анализа
        if FULL_ANALYSIS_AVAILABLE and not (self.addresses_file_path and self.addresses_file_path.exists()):
            log.info("🚀 FULL ANALYSIS MODE:")
            log.info("   • Will fetch 365-day detailed statistics via APIs")
            log.info("   • Uses Etherscan API for transaction history") 
            log.info("   • Uses Coinbase API for historical token prices")
            log.info("   • Includes gas fees, wallet age, activity patterns")
            log.info("")
            
            # Проверяем API ключ
            etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
            if etherscan_api_key:
                masked_key = etherscan_api_key[:8] + "..." + etherscan_api_key[-4:] if len(etherscan_api_key) > 12 else "***"
                log.info(f"✅ ETHERSCAN_API_KEY found: {masked_key}")
            else:
                log.warning("⚠️  ETHERSCAN_API_KEY not set")
                log.warning("   Will use basic graph-based analysis instead")
                log.warning("   Add ETHERSCAN_API_KEY=your_key to .env file for full functionality")
            log.info("")
        
        try:
            # 1. Загружаем граф
            self.load_graph()
            
            # 2. Определяем группы адресов
            self.detect_wallet_groups()
            
            if not self.wallet_groups:
                log.error("No wallet groups found")
                return
            
            # 3. Загружаем или анализируем статистику отдельных адресов
            self.load_or_analyze_individual_addresses()
            
            # 4. Рассчитываем статистику групп
            self.calculate_group_statistics()
            
            if not self.group_stats:
                log.error("No group statistics calculated")
                return
            
            python# 5. Создаем только PNG графики (БЕЗ HTML)
            log.info("Creating static PNG visualizations...")
            self.create_transaction_volume_bins_for_all_groups()  # НОВЫЙ: bin plot по транзакциям
            self.create_group_volume_distribution()
            self.create_group_size_analysis()
            self.create_top_groups_analysis()
            self.create_groups_vs_individuals_comparison()

            # 6. Сохраняем данные в JSON/CSV
            json_file, csv_file = self.save_groups_data()

            # 7. Выводим статистику в лог (БЕЗ HTML файла)
            log.info("=" * 60)
            log.info("GROUPS ANALYSIS COMPLETED SUCCESSFULLY")
            log.info("=" * 60)
            log.info(f"📊 {stats['total_groups']} groups analyzed")
            log.info(f"👥 {stats['total_addresses_in_groups']} addresses in groups")
            log.info(f"💰 ${stats['total_volume']:,.0f} total volume")
            log.info(f"🔄 {stats['total_transactions']:,.0f} total transactions")
            log.info(f"📁 Charts saved to: {self.output_dir}/plots/")
            log.info(f"📁 Data saved to: {json_file} and {csv_file}")
            log.info("📈 NEW: Transaction volume bins chart created")
            log.info("📈 Generated 5 PNG charts (no HTML reports)")
            log.info("=" * 60)
            
        except Exception as e:
            log.error(f"Groups analysis failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            raise

    def save_groups_data(self):
        """Сохраняет данные групп в JSON и CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Определяем префикс в зависимости от типа анализа
        if FULL_ANALYSIS_AVAILABLE and not (self.addresses_file_path and self.addresses_file_path.exists()):
            prefix = "groups_full_analysis_365d"
        else:
            prefix = "groups_analysis"
        
        # Сохранение в JSON
        json_file = self.output_dir / "data" / f"{prefix}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json_data = [asdict(group) for group in self.group_stats]
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Сохранение в CSV
        csv_file = self.output_dir / "data" / f"{prefix}_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.group_stats:
                writer = csv.DictWriter(f, fieldnames=asdict(self.group_stats[0]).keys())
                writer.writeheader()
                for group in self.group_stats:
                    writer.writerow(asdict(group))
        
        log.info(f"Groups data saved to: {json_file} and {csv_file}")
        return json_file, csv_file

    def _calculate_bins(self, data, bins):
        """Подсчитывает элементы в бинах"""
        bin_counts = []
        bin_labels = []
        
        for min_val, max_val, label in bins:
            if max_val == float('inf'):
                count = len([x for x in data if x >= min_val])
            else:
                count = len([x for x in data if min_val <= x < max_val])
            
            bin_counts.append(count)
            bin_labels.append(label)
        
        return bin_counts, bin_labels

    def run_full_analysis(self):
        """Запускает полный анализ групп"""
        log.info("=" * 60)
        log.info("ROCKET POOL GROUPS ANALYSIS STARTED")
        log.info("=" * 60)
        
        # Показываем информацию о типе анализа
        if FULL_ANALYSIS_AVAILABLE and not (self.addresses_file_path and self.addresses_file_path.exists()):
            log.info("🚀 FULL ANALYSIS MODE:")
            log.info("   • Will fetch 365-day detailed statistics via APIs")
            log.info("   • Uses Etherscan API for transaction history") 
            log.info("   • Uses Coinbase API for historical token prices")
            log.info("   • Includes gas fees, wallet age, activity patterns")
            log.info("")
            
            # Проверяем API ключ
            etherscan_api_key = os.getenv("ETHERSCAN_API_KEY")
            if etherscan_api_key:
                masked_key = etherscan_api_key[:8] + "..." + etherscan_api_key[-4:] if len(etherscan_api_key) > 12 else "***"
                log.info(f"✅ ETHERSCAN_API_KEY found: {masked_key}")
            else:
                log.warning("⚠️  ETHERSCAN_API_KEY not set")
                log.warning("   Will use basic graph-based analysis instead")
                log.warning("   Add ETHERSCAN_API_KEY=your_key to .env file for full functionality")
            log.info("")
        
        try:
            # 1. Загружаем граф
            self.load_graph()
            
            # 2. Определяем группы адресов
            self.detect_wallet_groups()
            
            if not self.wallet_groups:
                log.error("No wallet groups found")
                return
            
            # 3. Загружаем или анализируем статистику отдельных адресов
            self.load_or_analyze_individual_addresses()
            
            # 4. Рассчитываем статистику групп
            self.calculate_group_statistics()
            
            if not self.group_stats:
                log.error("No group statistics calculated")
                return
            
            # 5. Создаем визуализации
            self.create_group_volume_distribution()
            self.create_group_size_analysis()
            self.create_top_groups_analysis()
            self.create_groups_vs_individuals_comparison()
            
            # 6. Сохраняем данные
            json_file, csv_file = self.save_groups_data()
            
            # 7. Генерируем отчет
            stats = self.generate_groups_report()
            
            log.info("=" * 60)
            log.info("GROUPS ANALYSIS COMPLETED SUCCESSFULLY")
            log.info("=" * 60)
            log.info(f"📊 {stats['total_groups']} groups analyzed")
            log.info(f"👥 {stats['total_addresses_in_groups']} addresses in groups")
            log.info(f"💰 ${stats['total_volume']:,.0f} total volume")
            log.info(f"📁 Results saved to: {self.output_dir}")
            log.info("=" * 60)
            
        except Exception as e:
            log.error(f"Groups analysis failed: {e}")
            import traceback
            log.error(traceback.format_exc())
            raise

def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rocket Pool Groups Analyzer - analyzes coordinated groups of addresses with FULL 365-day API statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 FULL ANALYSIS FEATURES:
• Detects coordinated wallet groups using advanced algorithms
• Fetches 365-day detailed statistics via Etherscan API  
• Uses historical token prices from Coinbase API
• Includes gas fees, wallet age, activity patterns
• Creates comprehensive visualizations and reports

📋 REQUIREMENTS & SETUP:
• ETHERSCAN_API_KEY for full functionality
• Internet connection for API calls
• Run from mintegrity project root directory

🔑 API KEY SETUP (choose one method):
Method 1 - .env file (recommended):
  echo "ETHERSCAN_API_KEY=your_api_key_here" >> .env
  
Method 2 - Environment variable:
  export ETHERSCAN_API_KEY="your_api_key_here"
  
Method 3 - Terminal session:
  ETHERSCAN_API_KEY="your_api_key" python scripts/stats_vis/rocket_pool_groups_analyzer.py

Get free API key: https://etherscan.io/apis

🚀 EXAMPLES:
From mintegrity root (recommended):
  cd mintegrity && python scripts/stats_vis/rocket_pool_groups_analyzer.py
  cd mintegrity && python scripts/stats_vis/rocket_pool_groups_analyzer.py --threshold 6.0

From scripts/stats_vis/ directory:
  python rocket_pool_groups_analyzer.py --graph-path ../../files/rocket_pool_full_graph_90_days.json
  python rocket_pool_groups_analyzer.py --graph-path ../../files/custom_graph.json --threshold 3.0

With custom settings:
  python scripts/stats_vis/rocket_pool_groups_analyzer.py --min-group-size 3 --max-workers 10
  python scripts/stats_vis/rocket_pool_groups_analyzer.py --addresses-file files/existing_analysis.json

Note: 
- Script automatically detects correct paths
- Without API key: basic analysis using graph data only
- With API key: full 365-day analysis with USD values, gas fees, etc.
- Script automatically loads .env file from project root
        """
    )
    
    parser.add_argument(
        "--graph-path",
        default="files/rocket_pool_full_graph_90_days.json",
        help="Path to graph file (default: files/rocket_pool_full_graph_90_days.json)"
    )
    
    parser.add_argument(
        "--addresses-file",
        help="Path to existing addresses analysis JSON file (optional)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="files/rocket_pool_groups_analysis",
        help="Output directory (default: files/rocket_pool_groups_analysis)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Coordination threshold for grouping (default: 5.0)"
    )
    
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum group size (default: 2)"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum concurrent API requests (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Показываем где мы работаем
    log.info(f"Working directory: {Path.cwd()}")
    
    try:
        analyzer = RocketPoolGroupsAnalyzer(
            graph_file_path=args.graph_path,
            addresses_file_path=args.addresses_file,
            output_dir=args.output_dir,
            coordination_threshold=args.threshold,
            min_group_size=args.min_group_size,
            max_workers=args.max_workers
        )
        
        # Проверяем, что граф существует после инициализации
        if not analyzer.graph_file_path.exists():
            log.error(f"Graph file not found: {analyzer.graph_file_path}")
            log.error("Solutions:")
            log.error("1. Run from mintegrity root: cd /path/to/mintegrity")
            log.error("2. Use correct relative path: --graph-path ../../files/graph.json")
            log.error("3. Use absolute path: --graph-path /full/path/to/graph.json")
            return 1
        
        analyzer.run_full_analysis()
        
    except KeyboardInterrupt:
        log.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        log.error(f"Analysis failed: {e}")
        log.error("If you're getting import errors, make sure you're running from the mintegrity project root")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
