# Saucedo Quantitative Trading Engine

A streamlined backtesting and strategy analysis framework optimized for rapid strategy development and testing.

## 🚀 Quick Start

### Instant Strategy Testing
```bash
# Test a strategy in seconds
python tests/quick_test.py --strategy momentum --symbols BTC_USD,ETH_USD --period 2024

# Test composite pair trading strategy  
python tests/quick_test.py --strategy composite --period ytd

# List available strategies
python tests/quick_test.py --list-strategies
```

### Comprehensive Single Strategy Analysis
```bash
# Full statistical analysis with optimized bootstrap
python tests/single_strategy_comprehensive_test.py momentum --symbols BTC_USD --start-date 2023-01-01

# Quick validation mode (development)
python tests/single_strategy_comprehensive_test.py momentum --symbols BTC_USD,ETH_USD --quick
```

### Interactive Strategy Runner
```bash
# Simple interactive testing
python tests/run_strategy.py

# Run specific strategy with plots
python tests/run_strategy.py momentum --plot

# Quick parameter optimization
python tests/run_strategy.py --optimize composite --params lookback_window=20,30,40
```

### Batch Testing & Comparison
```bash
# Test multiple strategies
python tests/batch_test.py --strategies momentum,mean_reversion,composite --symbols BTC_USD,ETH_USD

# Use configuration file
python tests/batch_test.py --config config/batch_configs/quick_test.yaml

# Parameter optimization
python tests/batch_test.py --optimize composite --param-range lookback_window=20,30,40 z_entry_threshold=1.5,2.0,2.5
```

## 📁 Optimized Structure

```
📦 saucedo-quantitative-trading-engine/
├── 🚀 quick_test.py              # Instant strategy testing (5 sec)
├── 🔄 batch_test.py              # Multi-strategy comparison (1 min)  
├── 🎮 run_strategy.py            # Interactive development
├── 🔍 validate_strategies.py     # Quality control & overfitting detection
├── 🧪 test_integration.py        # System integration tests
├── 📖 WORKFLOW_GUIDE.md          # Complete development workflow
│
├── 📁 src/strategies/            # Core strategy implementations
│   ├── momentum.py                 # ⚠️ Needs fixing (40,000% returns)
│   ├── mean_reversion.py           # ⚠️ Needs fixing (10,000% returns)
│   ├── composite_pair_trading_strategy.py  # ✅ Working (realistic returns)
│   └── strategy_suite.py           # Strategy testing framework
│
├── 📁 config/batch_configs/      # Ready-to-use test configurations
│   ├── quick_test.yaml             # Basic 3-strategy comparison
│   └── comprehensive.yaml          # Multi-parameter testing
│
├── 📁 data/                      # Preserved market data
├── 📁 results/exports/           # Auto-saved test results (JSON)
└── 📁 scripts/backtesting/       # Advanced statistical analysis
```

## 🎯 Optimized Development Workflow

### ⚡ 1. Instant Validation (5 seconds)
```bash
# Quick test any strategy + validation check
python quick_test.py --strategy momentum --quick && python validate_strategies.py
```

### 🔧 2. Parameter Optimization (30 seconds)
```bash
# Grid search with immediate results
python batch_test.py --optimize composite --param-range lookback_window=20,30,40
```

### 📊 3. Strategy Comparison (1 minute)
```bash
# Comprehensive multi-strategy analysis
python batch_test.py --config config/batch_configs/comprehensive.yaml
```

### 🔍 4. Quality Control (Always)
```bash
# Automated validation catches overfitting
python validate_strategies.py  # Flags suspicious results automatically
```

### 📈 5. Production Analysis (Optional)
```bash
# Full statistical validation with bootstrap
python scripts/backtesting/comprehensive_stat_arb_backtest.py --strategy composite
```

> **💡 Pro Tip**: Always run `validate_strategies.py` - it automatically detects overfitted strategies with >1000% returns

## 💡 Key Features

### ⚡ Performance Optimized
- **10x Faster Bootstrap**: Optimized configurations for development vs production
- **Smart Caching**: Intelligent data loading and result caching
- **Memory Efficient**: Reduced memory footprint for large backtests
- **Configurable Modes**: Development (fast) vs Production (rigorous) configurations

### 🧪 Development-First Approach
- **Sub-5 Second Tests**: Quick strategy validation without overhead  
- **One-Command Testing**: Simple CLI for immediate results
- **Interactive Mode**: Development-friendly interface
- **Parameter Optimization**: Automated parameter sweeps

### 🔄 Batch Processing  
- **Multi-Strategy Comparison**: Test multiple strategies simultaneously
- **Configuration-Driven**: YAML-based batch test setup
- **Export Results**: JSON exports for further analysis
- **Progress Tracking**: Real-time optimization progress

### 📊 Comprehensive Analysis
- **Bootstrap Validation**: Statistical significance testing
- **Risk Metrics**: VaR, CVaR, Maximum Drawdown, Sharpe ratios
- **Visual Analysis**: Automated plotting and reporting
- **Performance Attribution**: Detailed trade-level analysis

### 🎛️ Strategy Framework
- **Base Strategy Class**: Easy strategy development
- **Built-in Strategies**: Momentum, mean reversion, pair trading
- **Hot Reloading**: Modify strategies without restart
- **Type Safety**: Full type hints and validation

## 📈 Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **Momentum** | Trend-following based on price momentum | Trending markets |
| **Mean Reversion** | Buy low, sell high based on statistical levels | Range-bound markets |  
| **Composite Pair Trading** | BTC-ETH pair trading with entropy confirmation | Market-neutral strategies |

## ⚙️ Configuration Examples

### Quick Test Configuration (`config/batch_configs/quick_test.yaml`)
```yaml
strategies:
  - momentum
  - mean_reversion
  - composite

symbols:
  - BTC_USD
  - ETH_USD

period: "2024"

strategy_params:
  momentum:
    lookback_period: 20
  composite:
    lookback_window: 30
    z_entry_threshold: 2.0
```

### Parameter Optimization
```yaml
# Test multiple parameter combinations
strategy_params:
  composite:
    - lookback_window: 20
      z_entry_threshold: 1.5
    - lookback_window: 30  
      z_entry_threshold: 2.0
    - lookback_window: 40
      z_entry_threshold: 2.5
```

## 📊 Example Output

```
🚀 Quick Testing: MOMENTUM
📊 Symbols: BTC_USD
📅 Period: 2024

📥 Loading BTC_USD...
⚡ Running backtest...

📊 QUICK RESULTS:
  • Total Return: 67.84%
  • Annual Return: 67.84%
  • Sharpe Ratio: 1.245
  • Max Drawdown: -12.45%
  • Win Rate: 54.32%
  • Total Trades: 156

✅ Test completed successfully!
```

## 🔧 Advanced Usage

### Custom Strategy Development
```python
from src.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, param1=10, param2=0.02):
        self.param1 = param1
        self.param2 = param2
    
    def generate_signals(self, data):
        # Implement your strategy logic
        return signals
```

### Batch Optimization
```bash
# Optimize multiple parameters simultaneously  
python batch_test.py --optimize composite \
  --param-range lookback_window=20,30,40 \
               z_entry_threshold=1.5,2.0,2.5 \
               risk_budget=0.01,0.02,0.03
```

### Integration with External Tools
```python
# Use results in Jupyter notebooks
from quick_test import QuickTester

tester = QuickTester()
results = tester.quick_test('momentum', ['BTC_USD'], '2024')
returns = results['returns']

# Further analysis with your preferred tools
```
- **Real-Time Monitoring**: Live dashboard for running backtests
- **Interactive Visualizations**: Comprehensive plotting suite
- **Export Capabilities**: CSV, Excel, JSON, and HTML reports
- **Comparison Tools**: Multi-strategy performance comparison

## Quick Start

### Basic Strategy Backtesting
```python
from src.strategies import MomentumStrategy, StrategyTestSuite
from src.utils import load_sample_data

# Load data and create strategy
data = load_sample_data()
strategy = MomentumStrategy(lookback_period=20)

# Run backtest
results = strategy.backtest(data)
print(f"Annual Return: {results['annual_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
```

### Multi-Strategy Testing
```python
from src.strategies import StrategyTestSuite

# Initialize test suite
suite = StrategyTestSuite()

# Register multiple strategies
suite.register_strategy('momentum_10', momentum_strategy, {'lookback': 10})
suite.register_strategy('momentum_20', momentum_strategy, {'lookback': 20})

# Run comprehensive analysis
results = suite.run_single_strategy('momentum_10', data)
```

### Bootstrap Analysis
```python
from src.bootstrapping import AdvancedBootstrapping, BootstrapConfig

# Configure bootstrap analysis
config = BootstrapConfig(n_sims=5000, method='stationary')
bootstrap = AdvancedBootstrapping(ret_series=returns, config=config)

# Run analysis
results = bootstrap.run_bootstrap_simulation()
print(f"95% Confidence Interval: {results['confidence_intervals']}")
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-strategy-analyzer.git
cd trading-strategy-analyzer

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Requirements

- Python 3.9+
- NumPy, Pandas, SciPy
- Matplotlib, Plotly, Seaborn
- Numba (optional, for performance)
- Scikit-learn, Statsmodels
- PyPortfolioOpt (optional)

## Usage Examples

### Basic Bootstrap Analysis
```python
from src.bootstrapping import AdvancedBootstrapping

# Simple analysis
bootstrap = AdvancedBootstrapping(ret_series=returns, timeframe='1d')
results = bootstrap.mc_with_replacement()
summary = bootstrap.results()
```

### Strategy Comparison
```python
from src.strategies import StrategyTestSuite

suite = StrategyTestSuite()
suite.register_strategy('momentum', momentum_strategy)
suite.register_strategy('mean_reversion', mean_reversion_strategy)
comparison = suite.run_comprehensive_analysis()
```

### Advanced Statistical Testing
```python
# Regime analysis
regimes = bootstrap.detect_regime_changes(method='hmm')

# Tail risk analysis
tail_metrics = bootstrap.tail_risk_analysis()

# Robustness testing
robustness = bootstrap.robustness_tests()
```

## Project Structure

```
saucedo-quantitative-trading-engine/
├── 📁 src/                           # Core framework code
│   ├── 📁 bootstrapping/             # Statistical analysis & bootstrap methods
│   ├── 📁 strategies/                # Strategy implementations
│   ├── 📁 utils/                     # Utility functions & data handling
│   └── 📁 analysis/                  # Performance & risk analysis
├── 📁 tests/                         # All testing & validation tools
│   ├── 🚀 quick_test.py             # Instant strategy testing
│   ├── 📊 single_strategy_comprehensive_test.py  # Full analysis
│   ├── 🔄 batch_test.py             # Multi-strategy comparison
│   └── ✅ validate_strategies.py     # Quality control
├── 📁 config/                        # Configuration files
│   ├── 📁 batch_configs/             # Batch testing configurations
│   ├── 📁 strategy_configs/          # Strategy parameters
│   └── 🔧 bootstrap_configs.yaml    # Performance profiles
├── 📁 scripts/                       # Advanced analysis scripts
│   ├── 📁 backtesting/               # Production backtesting
│   ├── 📁 analysis/                  # Research analysis
│   └── 📁 data_collection/           # Data acquisition
├── 📁 research/                      # Research & prototypes
│   ├── 📁 strategy_prototypes/       # Experimental strategies
│   └── 📁 results_analysis/          # Research notebooks
├── 📁 results/                       # Output directory
│   ├── 📁 backtests/                 # Backtest results
│   ├── 📁 plots/                     # Generated visualizations
│   └── 📁 exports/                   # Export data
├── 📁 docs/                          # Documentation
│   ├── 📖 PERFORMANCE_OPTIMIZATION.md # Performance guide
│   └── 📁 tutorials/                 # Getting started guides
└── 📁 examples/                      # Usage examples
├── examples/                  # Usage examples
├── data/                      # Sample data and benchmarks
├── results/                   # Output directory
├── docs/                      # Documentation
└── config/                    # Configuration files
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_bootstrapping.py -v
```

## Key Metrics

The framework calculates over 20 performance and risk metrics:

- **Return Metrics**: CAGR, Total Return, Annualized Return
- **Risk Metrics**: Volatility, VaR, CVaR, Maximum Drawdown
- **Risk-Adjusted**: Sharpe, Sortino, Calmar, Omega Ratio
- **Tail Risk**: Skewness, Kurtosis, Tail Ratio
- **Advanced**: Information Ratio, Treynor Ratio, Jensen's Alpha

## Statistical Methods

### Bootstrap Variants
- **IID Bootstrap**: Simple resampling with replacement
- **Stationary Bootstrap**: Preserves autocorrelation structure
- **Block Bootstrap**: Maintains temporal dependencies
- **Wild Bootstrap**: Handles heteroscedasticity

### Significance Testing
- Empirical p-values with multiple comparison corrections
- Kolmogorov-Smirnov tests
- Anderson-Darling tests
- Mann-Whitney U tests

## Advanced Features

### Edge Detection
- Regime change detection using HMM
- Autocorrelation structure analysis
- Market condition performance attribution
- Strategy consistency metrics

### Robustness Analysis
- Parameter sensitivity testing
- Walk-forward validation
- Out-of-sample performance
- Stress testing scenarios

## Documentation

- [API Reference](docs/api_reference.md)
- [Tutorial: Getting Started](docs/tutorials/getting_started.md)
- [Advanced Usage](docs/tutorials/advanced_usage.md)
- [Strategy Development Guide](docs/tutorials/strategy_development.md)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by modern quantitative finance research.
- Intended for the trading community, for academic and professional use.

## Support

- For bugs or feature requests, please create an issue.
- Consult the documentation for common questions.
- Contributions are welcome.

---

**Disclaimer**: This software is for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough testing before deploying any trading strategy.
