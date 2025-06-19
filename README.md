# Mintegrity - blockchain transaction analysis toolkit

Most Web3 users juggle multiple wallets across different protocols and services. 
Our analytics connect these fragmented identities by analyzing on-chain behavior patterns, giving you an insight into your true audience.

Are whales driving most of your volume? Which other protocols are popular among your users? What does your typical user's portfolio look like?

Stop guessing what your community wants. With Mintegrity's user analysis, you'll understand who you're serving - and create products that resonate with their actual needs.

![Graph Group Highlights](docs/images/graph_group_highlights.png)


## Features
Mintegrity is a comprehensive toolkit for analyzing ETH blockchain transaction data, including direct wallet interactions, contract interactions, and interactions among popular tokens.
It enables users to build, visualize, and analyze transaction graphs to gain deep insights into blockchain networks and identify patterns of behavior.

### Core Analytics
- **Transaction Graph Building**: Create transaction graphs from blockchain data, using both direct wallet interactions and contract interactions
- **Graph Visualization**: Generate interactive visualizations of transaction networks
- **Node Categorization**: Categorize all addresses in the graph based on heuristics
- **Wallet Grouping**: Group related wallets based on transaction patterns

### Advanced Statistics & Visualization
- **Address Statistics Analysis**: Comprehensive 365-day analysis of wallet behavior including:
  - Volume analysis with historical USD prices
  - Transaction patterns and frequency
  - Gas usage and fees analysis
  - Wallet age and creation patterns
  - Activity correlation analysis
- **Group Behavior Analysis**: Detect and analyze coordinated wallet groups with:
  - Multi-wallet coordination detection
  - Aggregated group statistics
  - Internal vs external transaction patterns
  - Comparative analysis of group vs individual behavior
- **Rich Visualizations**: Generate publication-ready charts including:
  - Volume and transaction distribution bins
  - Whale vs regular user analysis
  - Top performers analysis
  - Activity heatmaps and correlation plots
  - Group efficiency and coordination metrics


## Installation
Compatible with Python 3.11 and newer.

1. Clone this repository
2. Create a virtual environment using venv
3. Install dependencies using requirements.txt
4. Create a `.env` file with required API keys:
   ```
   ALCHEMY_API_KEY=your_alchemy_key_here
   ETHERSCAN_API_KEY=your_etherscan_key_here  # For enhanced analytics
   ```

## Usage Examples

### Basic Transaction Graph Analysis
The `cases\rocketpool` directory contains complete examples demonstrating how to use the framework based on the [Rocket Pool](https://rocketpool.net/) smart contracts for basic graph building and analysis.

### Enhanced Statistics and Visualization
The `scripts\stats_vis` directory provides advanced analytical tools:

#### 1. Address Statistics Analysis
```bash
# Analyze addresses that interact with Rocket Pool contracts
python scripts/stats_vis/rocket_pool_analyzer.py

# Analyze all addresses in the graph
python scripts/stats_vis/rocket_pool_analyzer.py --all-addresses

# Custom graph and output directory
python scripts/stats_vis/rocket_pool_analyzer.py --graph-path path/to/graph.json --output-dir path/to/results
```

#### 2. Groups Analysis
```bash
# Detect and analyze coordinated wallet groups
python scripts/stats_vis/rocket_pool_groups_analyzer.py

# Custom coordination threshold and minimum group size
python scripts/stats_vis/rocket_pool_groups_analyzer.py --threshold 6.0 --min-group-size 3

# Use existing address analysis
python scripts/stats_vis/rocket_pool_groups_analyzer.py --addresses-file path/to/analysis.json
```

#### 3. Enhanced Visualizations
```bash
# Create comprehensive visualizations from analysis results
python scripts/stats_vis/rocket_pool_statistics_visualizer_enh.py results.json

# Custom output directory
python scripts/stats_vis/rocket_pool_statistics_visualizer_enh.py results.csv --output-dir ./my_analysis
```

**Generated Visualizations Include:**
- Volume distribution bins (by USD ranges)
- Transaction distribution bins
- Address type analysis
- Activity patterns (daily/monthly)
- Whale analysis (top 5% vs regular users)
- Gas usage analysis
- Top performers comparison

## Project Structure
- **scripts\\**: Core functionality
  - **commons\\**: Common utilities and models
  - **graph\\**: Graph-related functionality
    - **analysis\\**: Analytical tools including wallet grouping
    - **building\\**: Graph building
    - **categorization\\**: Node categorization
    - **model\\**: Data models
    - **optimization\\**: Graph optimization
    - **visualization\\**: Graph visualization
  - **stats_vis\\**: Advanced statistics and visualization tools
  - **cases\\**: Example use cases
  - **rocketpool\\**: Rocket Pool analysis examples

## API Requirements

### Core Features
- **ALCHEMY_API_KEY**: Required for basic graph building and blockchain data access

### Enhanced Analytics (Optional)
- **ETHERSCAN_API_KEY**: Enables 365-day detailed statistics including:
  - Historical transaction analysis
  - Wallet age and creation dates
  - Gas usage and fees calculation
  - Activity pattern analysis

Without ETHERSCAN_API_KEY, the enhanced analytics will fall back to basic graph-based statistics.

## Costs
> **Note:** Data collection for 1 popular contract for 365 days consumes only 2-3% of the Alchemy free quota, which corresponds to approximately $1 in value. Enhanced analytics with Etherscan API (free tier: 100,000 requests/day) can analyze hundreds of addresses daily at no additional cost. Usage in your use case depends on the number of contract interactions and the time period you analyze.

## Output Formats
The toolkit generates multiple output formats:
- **JSON/CSV**: Raw data for further processing
- **PNG Charts**: Publication-ready visualizations
- **Summary Reports**: Aggregated statistics and insights
- **Graph Files**: Transaction network data for visualization tools

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the license included in the repository.
