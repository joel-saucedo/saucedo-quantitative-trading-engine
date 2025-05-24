# Saucedo Quantitative Trading Engine 📈

A comprehensive, production-ready backtesting and strategy analysis framework designed for rapid iteration and testing of quantitative trading strategies. Built with emphasis on performance, scalability, and statistical rigor.

## 🌟 Features

### High-Performance Backtesting Engine
- **Vectorized Operations**: Optimized for speed with numpy/pandas vectorizations
- **Parallel Strategy Execution**: Test multiple strategies simultaneously
- **Memory Efficient**: Smart data loading and caching for large datasets
- **Event-Driven Architecture**: Realistic order execution with slippage and costs

### Rapid Strategy Development & Testing
- **Strategy Factory Pattern**: Easy registration and management of trading strategies
- **Parameter Sweep Framework**: Automated parameter optimization across strategy variants
- **Hot Reloading**: Modify strategies without restarting backtesting process
- **Configuration Management**: YAML/JSON-based strategy configurations

### Advanced Statistical Analysis
- **Multiple Bootstrap Methods**: IID, Stationary, Block, Circular, and Wild Bootstrap
- **Comprehensive Risk Metrics**: VaR, CVaR, Maximum Drawdown, Ulcer Index, and more
- **Edge Detection**: Regime analysis, autocorrelation preservation, tail risk assessment
- **Significance Testing**: Multiple hypothesis testing with proper corrections

### Multi-Asset & Cross-Sectional Capabilities
- **Universe Management**: Handle large asset universes efficiently
- **Cross-Asset Strategies**: Enable strategies across different asset classes
- **Multi-Timeframe Support**: Simultaneous analysis across multiple timeframes
- **Portfolio Optimization**: Mean-variance and risk-parity optimization

### Professional Analytics & Reporting
- **Real-Time Monitoring**: Live dashboard for running backtests
- **Interactive Visualizations**: Comprehensive plotting suite
- **Export Capabilities**: CSV, Excel, JSON, and HTML reports
- **Comparison Tools**: Multi-strategy performance comparison

## 🚀 Quick Start

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

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-strategy-analyzer.git
cd trading-strategy-analyzer

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## 🔧 Requirements

- Python 3.9+
- NumPy, Pandas, SciPy
- Matplotlib, Plotly, Seaborn
- Numba (optional, for performance)
- Scikit-learn, Statsmodels
- PyPortfolioOpt (optional)

## 📊 Usage Examples

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

## 📁 Project Structure

```
trading-strategy-analyzer/
├── src/
│   ├── bootstrapping/          # Core bootstrapping methods
│   ├── strategies/             # Strategy implementations
│   ├── utils/                  # Utility functions
│   └── analysis/              # Analysis modules
├── tests/                     # Test suite
├── examples/                  # Usage examples
├── data/                      # Sample data and benchmarks
├── results/                   # Output directory
├── docs/                      # Documentation
└── config/                    # Configuration files
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_bootstrapping.py -v
```

## 📈 Key Metrics

The framework calculates 20+ performance and risk metrics:

- **Return Metrics**: CAGR, Total Return, Annualized Return
- **Risk Metrics**: Volatility, VaR, CVaR, Maximum Drawdown
- **Risk-Adjusted**: Sharpe, Sortino, Calmar, Omega Ratio
- **Tail Risk**: Skewness, Kurtosis, Tail Ratio
- **Advanced**: Information Ratio, Treynor Ratio, Jensen's Alpha

## 🔬 Statistical Methods

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

## 🎯 Advanced Features

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

## 📖 Documentation

- [API Reference](docs/api_reference.md)
- [Tutorial: Getting Started](docs/tutorials/getting_started.md)
- [Advanced Usage](docs/tutorials/advanced_usage.md)
- [Strategy Development Guide](docs/tutorials/strategy_development.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by modern quantitative finance research
- Built for the trading community
- Designed for academic and professional use

## 📞 Support

- Create an issue for bugs or feature requests
- Check the documentation for common questions
- Contribute to make it better!

---

**⚠️ Disclaimer**: This software is for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough testing before deploying any trading strategy.
