#!/usr/bin/env python3
"""
Enhanced Rocket Pool Statistics Analyzer
Extended Rocket Pool address analyzer with additional charts
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
import sys
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')


class EnhancedRocketPoolAnalyzer:
    """Enhanced Rocket Pool address analyzer"""

    def __init__(self, results_file: str, output_dir: str = "../../files/rocket_pool_addresses_vis/"):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir)

        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)

        self.df = None

        print(f"🚀 Enhanced Rocket Pool Analyzer")
        print(f"📄 Input file: {self.results_file}")
        print(f"📁 Output: {self.output_dir}")

    def load_data(self):
        """Loads data from JSON or CSV"""
        print("📖 Loading data...")

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        # Determine file format and load
        if self.results_file.suffix.lower() == '.json':
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.df = pd.DataFrame(data)
        elif self.results_file.suffix.lower() == '.csv':
            self.df = pd.read_csv(self.results_file)
        else:
            raise ValueError(f"Unsupported file format: {self.results_file.suffix}")

        print(f"✅ Loaded {len(self.df)} addresses")
        print(f"📋 Columns: {list(self.df.columns)}")

        # Basic data cleaning
        self.df['address_type'] = self.df['address_type'].fillna('unknown')

        # Fill NaN with zeros for numeric columns
        numeric_cols = [col for col in self.df.columns if 'volume' in col or 'transactions' in col or 'gas' in col or 'days' in col]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def create_volume_bins(self):
        """Creates bin chart for volume distribution"""
        print("📊 Creating volume distribution bins...")

        volume_col = 'total_volume_usd_365d'
        if volume_col not in self.df.columns:
            print(f"⚠️ Column {volume_col} not found")
            return

        volumes = self.df[volume_col].fillna(0)

        # Define volume bins
        bins = [
            (0, 1_000, "$0-$1K"),
            (1_000, 10_000, "$1K-$10K"),
            (10_000, 100_000, "$10K-$100K"),
            (100_000, 1_000_000, "$100K-$1M"),
            (1_000_000, float('inf'), "$1M+")
        ]

        bin_counts, bin_labels = self._calculate_bins(volumes, bins)

        # Create chart
        plt.figure(figsize=(12, 8))
        colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        bars = plt.bar(bin_labels, bin_counts, color=colors, alpha=0.8, edgecolor='black')

        # Add values on bars
        for bar, count in zip(bars, bin_counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bin_counts) * 0.01,
                         str(count), ha='center', va='bottom', fontweight='bold')

        plt.title('Distribution of Addresses by Volume (365 days)', fontsize=16, fontweight='bold')
        plt.xlabel('Volume Range (USD)', fontsize=12)
        plt.ylabel('Number of Addresses', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        total = sum(bin_counts)
        plt.figtext(0.02, 0.98, f'Total Addresses: {total:,}', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "volume_distribution_bins.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Statistics
        self._print_bins_stats("Volume", bin_labels, bin_counts, volumes)

    def create_transactions_bins(self):
        """Creates bin chart for transaction distribution"""
        print("📊 Creating transactions distribution bins...")

        tx_col = 'total_transactions_365d'
        if tx_col not in self.df.columns:
            print(f"⚠️ Column {tx_col} not found")
            return

        transactions = self.df[tx_col].fillna(0)

        # Define transaction bins
        bins = [
            (0, 10, "0-10 tx"),
            (10, 50, "10-50 tx"),
            (50, 200, "50-200 tx"),
            (200, 1000, "200-1K tx"),
            (1000, float('inf'), "1K+ tx")
        ]

        bin_counts, bin_labels = self._calculate_bins(transactions, bins)

        # Create chart
        plt.figure(figsize=(12, 8))
        colors = ['#9b59b6', '#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        bars = plt.bar(bin_labels, bin_counts, color=colors, alpha=0.8, edgecolor='black')

        # Add values on bars
        for bar, count in zip(bars, bin_counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(bin_counts) * 0.01,
                         str(count), ha='center', va='bottom', fontweight='bold')

        plt.title('Distribution of Addresses by Transaction Count (365 days)', fontsize=16, fontweight='bold')
        plt.xlabel('Transaction Count Range', fontsize=12)
        plt.ylabel('Number of Addresses', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        total = sum(bin_counts)
        plt.figtext(0.02, 0.98, f'Total Addresses: {total:,}', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "transactions_distribution_bins.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Statistics
        self._print_bins_stats("Transactions", bin_labels, bin_counts, transactions)

    def create_address_type_analysis(self):
        """Creates simplified address type analysis (only 2 charts)"""
        print("👛 Creating simplified address type analysis...")

        # Count by types
        type_counts = self.df['address_type'].value_counts()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Address Type Analysis', fontsize=16, fontweight='bold')

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        # 1. Distribution by types
        axes[0].bar(type_counts.index, type_counts.values, color=colors[:len(type_counts)])
        axes[0].set_title('Address Count by Type', fontweight='bold')
        axes[0].set_ylabel('Count')
        for i, v in enumerate(type_counts.values):
            axes[0].text(i, v + max(type_counts.values) * 0.01, str(v), ha='center', fontweight='bold')

        # 2. Transactions by types
        tx_col = 'total_transactions_365d'
        if tx_col in self.df.columns:
            tx_by_type = self.df.groupby('address_type')[tx_col].sum()
            axes[1].bar(tx_by_type.index, tx_by_type.values, color=colors[:len(tx_by_type)])
            axes[1].set_title('Total Transactions by Address Type', fontweight='bold')
            axes[1].set_ylabel('Total Transactions')
            axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
            for i, v in enumerate(tx_by_type.values):
                axes[1].text(i, v + max(tx_by_type.values) * 0.01, f'{v:,.0f}', ha='center', fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'No transaction data', transform=axes[1].transAxes, ha='center', va='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "address_type_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_activity_analysis(self):
        """Creates address activity analysis"""
        print("📈 Creating activity analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Address Activity Analysis', fontsize=16, fontweight='bold')

        # 1. Active days distribution
        if 'active_days_365d' in self.df.columns:
            active_days = self.df['active_days_365d'].fillna(0)
            active_days = active_days[active_days > 0]

            axes[0, 0].hist(active_days, bins=30, alpha=0.7, color='#2ecc71', edgecolor='black')
            axes[0, 0].set_xlabel('Active Days (365d)')
            axes[0, 0].set_ylabel('Number of Addresses')
            axes[0, 0].set_title('Distribution of Active Days')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axvline(active_days.mean(), color='red', linestyle='--',
                               label=f'Mean: {active_days.mean():.1f}')
            axes[0, 0].legend()
        else:
            axes[0, 0].text(0.5, 0.5, 'No active days data', transform=axes[0, 0].transAxes, ha='center', va='center')

        # 2. Average transaction volume
        if 'average_volume_usd_365d' in self.df.columns:
            avg_volume = self.df['average_volume_usd_365d'].fillna(0)
            avg_volume = avg_volume[avg_volume > 0]

            # Logarithmic scale for better display
            axes[0, 1].hist(np.log10(avg_volume + 1), bins=30, alpha=0.7, color='#f39c12', edgecolor='black')
            axes[0, 1].set_xlabel('Log10(Average Volume per Transaction + 1)')
            axes[0, 1].set_ylabel('Number of Addresses')
            axes[0, 1].set_title('Distribution of Average Transaction Volume')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No average volume data', transform=axes[0, 1].transAxes, ha='center', va='center')

        # 3. Volume vs Active Days correlation
        volume_col = 'total_volume_usd_365d'
        if volume_col in self.df.columns and 'active_days_365d' in self.df.columns:
            scatter_data = self.df[(self.df[volume_col] > 0) & (self.df['active_days_365d'] > 0)]

            if len(scatter_data) > 0:
                colors = ['red' if x == 'contract' else 'blue' for x in scatter_data['address_type']]
                axes[1, 0].scatter(scatter_data['active_days_365d'], scatter_data[volume_col],
                                   c=colors, alpha=0.6, s=30)
                axes[1, 0].set_xlabel('Active Days')
                axes[1, 0].set_ylabel('Total Volume (USD)')
                axes[1, 0].set_title('Volume vs Active Days')
                axes[1, 0].set_yscale('log')
                axes[1, 0].grid(True, alpha=0.3)

                # Add legend
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Wallet'),
                                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Contract')]
                axes[1, 0].legend(handles=legend_elements)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data for correlation', transform=axes[1, 0].transAxes, ha='center', va='center')

        # 4. Frequency analysis (transactions per day)
        tx_col = 'total_transactions_365d'
        if tx_col in self.df.columns and 'active_days_365d' in self.df.columns:
            freq_data = self.df[(self.df[tx_col] > 0) & (self.df['active_days_365d'] > 0)]
            if len(freq_data) > 0:
                freq_data['tx_per_day'] = freq_data[tx_col] / freq_data['active_days_365d']

                axes[1, 1].hist(freq_data['tx_per_day'], bins=30, alpha=0.7, color='#9b59b6', edgecolor='black')
                axes[1, 1].set_xlabel('Transactions per Active Day')
                axes[1, 1].set_ylabel('Number of Addresses')
                axes[1, 1].set_title('Transaction Frequency Distribution')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].axvline(freq_data['tx_per_day'].median(), color='red', linestyle='--',
                                   label=f'Median: {freq_data["tx_per_day"].median():.2f}')
                axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data for frequency analysis', transform=axes[1, 1].transAxes, ha='center', va='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "activity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_whale_analysis(self):
        """Creates whale vs regular users analysis"""
        print("🐋 Creating whale analysis...")

        volume_col = 'total_volume_usd_365d'
        if volume_col not in self.df.columns:
            print("⚠️ No volume data for whale analysis")
            return

        # Define threshold for "whales" (top 5% by volume)
        volume_threshold = self.df[volume_col].quantile(0.95)

        self.df['user_category'] = self.df[volume_col].apply(
            lambda x: 'Whale' if x >= volume_threshold else 'Regular'
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Whale Analysis (Threshold: ${volume_threshold:,.0f})', fontsize=16, fontweight='bold')

        # 1. Whales vs regular users distribution
        user_counts = self.df['user_category'].value_counts()
        colors = ['#e74c3c', '#3498db']

        wedges, texts, autotexts = axes[0, 0].pie(user_counts.values, labels=user_counts.index,
                                                  colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Distribution: Whales vs Regular Users')

        # 2. Volume by category
        volume_by_category = self.df.groupby('user_category')[volume_col].sum()
        axes[0, 1].bar(volume_by_category.index, volume_by_category.values, color=colors)
        axes[0, 1].set_title('Total Volume by User Category')
        axes[0, 1].set_ylabel('Total Volume (USD)')
        axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        for i, v in enumerate(volume_by_category.values):
            axes[0, 1].text(i, v + max(volume_by_category.values) * 0.01,
                            f'${v:,.0f}', ha='center', fontweight='bold')

        # 3. Transactions by category
        tx_col = 'total_transactions_365d'
        if tx_col in self.df.columns:
            tx_by_category = self.df.groupby('user_category')[tx_col].sum()
            axes[1, 0].bar(tx_by_category.index, tx_by_category.values, color=colors)
            axes[1, 0].set_title('Total Transactions by User Category')
            axes[1, 0].set_ylabel('Total Transactions')

            for i, v in enumerate(tx_by_category.values):
                axes[1, 0].text(i, v + max(tx_by_category.values) * 0.01,
                                f'{v:,}', ha='center', fontweight='bold')

        # 4. Average activity
        if 'active_days_365d' in self.df.columns:
            activity_by_category = self.df.groupby('user_category')['active_days_365d'].mean()
            axes[1, 1].bar(activity_by_category.index, activity_by_category.values, color=colors)
            axes[1, 1].set_title('Average Active Days by User Category')
            axes[1, 1].set_ylabel('Average Active Days')

            for i, v in enumerate(activity_by_category.values):
                axes[1, 1].text(i, v + max(activity_by_category.values) * 0.01,
                                f'{v:.1f}', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "whale_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_gas_analysis(self):
        """Creates gas costs analysis"""
        print("⛽ Creating gas analysis...")

        gas_cols = [col for col in self.df.columns if 'gas' in col.lower()]

        if not gas_cols:
            print("⚠️ No gas data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gas Usage Analysis', fontsize=16, fontweight='bold')

        # Use first found gas column
        gas_col = gas_cols[0]
        gas_data = self.df[gas_col].fillna(0)
        gas_data = gas_data[gas_data > 0]

        if len(gas_data) == 0:
            for ax in axes.flat:
                ax.text(0.5, 0.5, 'No gas data available', transform=ax.transAxes, ha='center', va='center')
        else:
            # 1. Gas costs distribution
            axes[0, 0].hist(gas_data, bins=30, alpha=0.7, color='#e67e22', edgecolor='black')
            axes[0, 0].set_xlabel('Gas Used')
            axes[0, 0].set_ylabel('Number of Addresses')
            axes[0, 0].set_title(f'Distribution of {gas_col}')
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Gas by address types
            gas_by_type = self.df.groupby('address_type')[gas_col].mean()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            axes[0, 1].bar(gas_by_type.index, gas_by_type.values, color=colors[:len(gas_by_type)])
            axes[0, 1].set_title('Average Gas Usage by Address Type')
            axes[0, 1].set_ylabel('Average Gas')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # 3. Top gas consumers
            top_gas = self.df.nlargest(15, gas_col)
            y_pos = np.arange(len(top_gas))
            colors = ['red' if x == 'contract' else 'blue' for x in top_gas['address_type']]

            axes[1, 0].barh(y_pos, top_gas[gas_col], color=colors)
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels([f"{addr[:8]}...({t})" for addr, t in
                                        zip(top_gas['address'], top_gas['address_type'])], fontsize=8)
            axes[1, 0].set_xlabel('Gas Used')
            axes[1, 0].set_title('Top 15 Gas Consumers')
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Gas vs volume correlation
            volume_col = 'total_volume_usd_365d'
            if volume_col in self.df.columns:
                corr_data = self.df[(self.df[gas_col] > 0) & (self.df[volume_col] > 0)]
                if len(corr_data) > 0:
                    colors = ['red' if x == 'contract' else 'blue' for x in corr_data['address_type']]
                    axes[1, 1].scatter(corr_data[gas_col], corr_data[volume_col], c=colors, alpha=0.6, s=30)
                    axes[1, 1].set_xlabel('Gas Used')
                    axes[1, 1].set_ylabel('Total Volume (USD)')
                    axes[1, 1].set_title('Gas Usage vs Volume')
                    axes[1, 1].set_yscale('log')
                    axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No volume data for correlation', transform=axes[1, 1].transAxes, ha='center', va='center')

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "gas_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_top_performers(self):
        """Creates top performers analysis"""
        print("🏆 Creating top performers analysis...")

        volume_col = 'total_volume_usd_365d'
        tx_col = 'total_transactions_365d'

        if volume_col not in self.df.columns:
            print("⚠️ No volume data for top performers")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Top Performers Analysis', fontsize=16, fontweight='bold')

        n_top = min(15, len(self.df))

        # 1. Top by volume
        top_volume = self.df.nlargest(n_top, volume_col)
        y_pos = np.arange(len(top_volume))
        colors = ['red' if x == 'contract' else 'blue' for x in top_volume['address_type']]

        axes[0, 0].barh(y_pos, top_volume[volume_col], color=colors)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels([f"{addr[:8]}...({t})" for addr, t in
                                    zip(top_volume['address'], top_volume['address_type'])], fontsize=8)
        axes[0, 0].set_xlabel('Total Volume (USD)')
        axes[0, 0].set_title(f'Top {n_top} by Volume')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Top by transactions
        if tx_col in self.df.columns:
            top_tx = self.df.nlargest(n_top, tx_col)
            y_pos = np.arange(len(top_tx))
            colors = ['red' if x == 'contract' else 'blue' for x in top_tx['address_type']]

            axes[0, 1].barh(y_pos, top_tx[tx_col], color=colors)
            axes[0, 1].set_yticks(y_pos)
            axes[0, 1].set_yticklabels([f"{addr[:8]}...({t})" for addr, t in
                                        zip(top_tx['address'], top_tx['address_type'])], fontsize=8)
            axes[0, 1].set_xlabel('Total Transactions')
            axes[0, 1].set_title(f'Top {n_top} by Transactions')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Scatter plot Volume vs Transactions
        if tx_col in self.df.columns:
            scatter_data = self.df[self.df[volume_col] > 0]
            colors = ['red' if x == 'contract' else 'blue' for x in scatter_data['address_type']]

            axes[1, 0].scatter(scatter_data[tx_col], scatter_data[volume_col], c=colors, alpha=0.6, s=30)
            axes[1, 0].set_xlabel('Total Transactions')
            axes[1, 0].set_ylabel('Total Volume (USD)')
            axes[1, 0].set_title('Volume vs Transactions')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)

            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Wallet'),
                               Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Contract')]
            axes[1, 0].legend(handles=legend_elements)

        # 4. Efficiency (volume per transaction)
        if tx_col in self.df.columns:
            efficiency_data = self.df[(self.df[volume_col] > 0) & (self.df[tx_col] > 0)]
            if len(efficiency_data) > 0:
                efficiency_data['volume_per_tx'] = efficiency_data[volume_col] / efficiency_data[tx_col]
                top_efficiency = efficiency_data.nlargest(n_top, 'volume_per_tx')

                y_pos = np.arange(len(top_efficiency))
                colors = ['red' if x == 'contract' else 'blue' for x in top_efficiency['address_type']]

                axes[1, 1].barh(y_pos, top_efficiency['volume_per_tx'], color=colors)
                axes[1, 1].set_yticks(y_pos)
                axes[1, 1].set_yticklabels([f"{addr[:8]}...({t})" for addr, t in
                                            zip(top_efficiency['address'], top_efficiency['address_type'])], fontsize=8)
                axes[1, 1].set_xlabel('Volume per Transaction (USD)')
                axes[1, 1].set_title(f'Top {n_top} by Efficiency')
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "top_performers.png", dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_dashboard(self):
        """Creates summary dashboard"""
        print("📋 Creating summary dashboard...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Rocket Pool Analytics Summary Dashboard', fontsize=18, fontweight='bold')

        # 1. General statistics
        total_addresses = len(self.df)
        wallets = len(self.df[self.df['address_type'] == 'wallet'])
        contracts = len(self.df[self.df['address_type'] == 'contract'])

        volume_col = 'total_volume_usd_365d'
        tx_col = 'total_transactions_365d'

        total_volume = 0
        total_transactions = 0
        avg_volume_per_address = 0
        avg_tx_per_address = 0

        if volume_col in self.df.columns:
            total_volume = self.df[volume_col].fillna(0).sum()
            avg_volume_per_address = self.df[volume_col].fillna(0).mean()

        if tx_col in self.df.columns:
            total_transactions = self.df[tx_col].fillna(0).sum()
            avg_tx_per_address = self.df[tx_col].fillna(0).mean()

        # Text summary
        summary_text = f"""
ROCKET POOL ANALYTICS SUMMARY
{'=' * 40}

📊 Total Addresses: {total_addresses:,}
👛 Wallets: {wallets:,} ({wallets / total_addresses * 100:.1f}%)
📄 Contracts: {contracts:,} ({contracts / total_addresses * 100:.1f}%)

💰 Total Volume: ${total_volume:,.0f}
🔄 Total Transactions: {total_transactions:,}

📈 Avg Volume/Address: ${avg_volume_per_address:,.0f}
📈 Avg Transactions/Address: {avg_tx_per_address:.1f}
        """

        axes[0, 0].text(0.05, 0.95, summary_text, transform=axes[0, 0].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Key Statistics')

        # 2. Top volumes (mini version)
        if volume_col in self.df.columns:
            top_5_volume = self.df.nlargest(5, volume_col)
            colors = ['red' if x == 'contract' else 'blue' for x in top_5_volume['address_type']]

            y_pos = np.arange(len(top_5_volume))
            axes[0, 1].barh(y_pos, top_5_volume[volume_col], color=colors)
            axes[0, 1].set_yticks(y_pos)
            axes[0, 1].set_yticklabels([f"{addr[:6]}..." for addr in top_5_volume['address']], fontsize=8)
            axes[0, 1].set_title('Top 5 by Volume')
            axes[0, 1].set_xlabel('Volume (USD)')
            axes[0, 1].ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

        # 3. Address types distribution
        type_counts = self.df['address_type'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        wedges, texts, autotexts = axes[0, 2].pie(type_counts.values, labels=type_counts.index,
                                                  colors=colors[:len(type_counts)], autopct='%1.1f%%')
        axes[0, 2].set_title('Address Types Distribution')

        # 4. Volume bins (simplified version)
        if volume_col in self.df.columns:
            volumes = self.df[volume_col].fillna(0)
            bins = [(0, 1_000), (1_000, 10_000), (10_000, 100_000), (100_000, 1_000_000), (1_000_000, float('inf'))]
            labels = ["$0-$1K", "$1K-$10K", "$10K-$100K", "$100K-$1M", "$1M+"]

            bin_counts = []
            for min_val, max_val in bins:
                if max_val == float('inf'):
                    count = len(volumes[volumes >= min_val])
                else:
                    count = len(volumes[(volumes >= min_val) & (volumes < max_val)])
                bin_counts.append(count)

            colors = ['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
            axes[1, 0].bar(labels, bin_counts, color=colors)
            axes[1, 0].set_title('Volume Distribution')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].tick_params(axis='x', rotation=45)

        # 5. Activity (if data available)
        if 'active_days_365d' in self.df.columns:
            active_days = self.df['active_days_365d'].fillna(0)
            active_days = active_days[active_days > 0]

            axes[1, 1].hist(active_days, bins=20, alpha=0.7, color='#2ecc71', edgecolor='black')
            axes[1, 1].set_title('Active Days Distribution')
            axes[1, 1].set_xlabel('Active Days')
            axes[1, 1].set_ylabel('Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'No activity data', transform=axes[1, 1].transAxes, ha='center', va='center')

        # 6. Transactions vs Volume (correlation)
        if tx_col in self.df.columns and volume_col in self.df.columns:
            scatter_data = self.df[(self.df[volume_col] > 0) & (self.df[tx_col] > 0)]
            if len(scatter_data) > 100:  # Limit for readability
                scatter_data = scatter_data.sample(100)

            colors = ['red' if x == 'contract' else 'blue' for x in scatter_data['address_type']]
            axes[1, 2].scatter(scatter_data[tx_col], scatter_data[volume_col], c=colors, alpha=0.6, s=20)
            axes[1, 2].set_xlabel('Transactions')
            axes[1, 2].set_ylabel('Volume (USD)')
            axes[1, 2].set_title('Volume vs Transactions')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "summary_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_bins(self, data, bins):
        """Calculates elements in bins"""
        bin_counts = []
        bin_labels = []

        for min_val, max_val, label in bins:
            if max_val == float('inf'):
                count = len(data[data >= min_val])
            else:
                count = len(data[(data >= min_val) & (data < max_val)])

            bin_counts.append(count)
            bin_labels.append(label)

        return bin_counts, bin_labels

    def _print_bins_stats(self, prefix, bin_labels, bin_counts, data):
        """Prints bin statistics"""
        total = sum(bin_counts)
        print(f"📊 {prefix} distribution:")
        for label, count in zip(bin_labels, bin_counts):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {label}: {count:,} addresses ({percentage:.1f}%)")

        print(f"💰 {prefix} statistics:")
        print(f"  Total: {data.sum():,.2f}")
        print(f"  Average: {data.mean():,.2f}")
        print(f"  Median: {data.median():,.2f}")
        print(f"  Max: {data.max():,.2f}")

    def run_full_analysis(self):
        """Runs full enhanced analysis"""
        print("🚀 Starting Enhanced Rocket Pool Analytics...")
        print("=" * 60)

        try:
            # 1. Load data
            self.load_data()

            # 2. Main bin charts
            self.create_volume_bins()
            self.create_transactions_bins()

            # 3. Simplified address type analysis
            self.create_address_type_analysis()

            # 4. New enhanced analyses
            self.create_activity_analysis()
            self.create_whale_analysis()
            self.create_gas_analysis()

            # 5. Top performers analysis
            self.create_top_performers()

            # 6. Summary dashboard
            self.create_summary_dashboard()

            # 7. Output final statistics
            volume_col = 'total_volume_usd_365d'
            tx_col = 'total_transactions_365d'

            total_addresses = len(self.df)
            wallets = len(self.df[self.df['address_type'] == 'wallet'])
            contracts = len(self.df[self.df['address_type'] == 'contract'])

            total_volume = 0
            total_transactions = 0

            if volume_col in self.df.columns:
                total_volume = self.df[volume_col].fillna(0).sum()

            if tx_col in self.df.columns:
                total_transactions = self.df[tx_col].fillna(0).sum()

            print("=" * 60)
            print("✅ ENHANCED ANALYSIS COMPLETED!")
            print("=" * 60)
            print(f"📊 {total_addresses} addresses analyzed")
            print(f"👛 {wallets} wallets, 📄 {contracts} contracts")
            print(f"💰 ${total_volume:,.0f} total volume")
            print(f"🔄 {total_transactions:,.0f} total transactions")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"📈 Plots saved in: {self.output_dir / 'plots'}")
            print("\n🎨 Generated visualizations:")
            print("  • volume_distribution_bins.png")
            print("  • transactions_distribution_bins.png")
            print("  • address_type_analysis.png (simplified)")
            print("  • activity_analysis.png (NEW)")
            print("  • whale_analysis.png (NEW)")
            print("  • gas_analysis.png (NEW)")
            print("  • top_performers.png")
            print("  • summary_dashboard.png (NEW)")
            print("=" * 60)

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            raise


def main():
    """Main function with CLI interface"""

    parser = argparse.ArgumentParser(
        description="🚀 Enhanced Rocket Pool Statistics Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Examples:
  python enhanced_rocket_pool_analyzer.py results.json
  python enhanced_rocket_pool_analyzer.py data.csv --output-dir ./my_analysis

📊 Output:
  - 8 comprehensive PNG visualizations in output_dir/plots/

🧩 New Features:
  - Activity Analysis (active days, frequency, correlations)
  - Whale Analysis (top 5% vs regular users)
  - Gas Usage Analysis
  - Summary Dashboard
  - Enhanced Top Performers with efficiency metrics
        """
    )

    parser.add_argument(
        "results_file",
        help="📄 Path to JSON or CSV results file from rocket_pool_analyzer.py"
    )

    parser.add_argument(
        "--output-dir",
        default="../../files/rocket_pool_addresses_vis/",
        help="📁 Output directory for visualizations and reports"
    )

    args = parser.parse_args()

    try:
        # Create and run enhanced analyzer
        analyzer = EnhancedRocketPoolAnalyzer(
            results_file=args.results_file,
            output_dir=args.output_dir
        )

        analyzer.run_full_analysis()

        return 0

    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
