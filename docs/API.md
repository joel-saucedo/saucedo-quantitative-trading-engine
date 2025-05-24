# API Documentation

## Trading Strategy Analyzer Framework

### Overview

This framework provides comprehensive tools for analyzing trading strategies using advanced statistical methods, Monte Carlo simulations, and robust backtesting techniques.

### Core Modules

#### 1. Bootstrapping Module (`src.bootstrapping`)

The bootstrapping module provides advanced statistical resampling techniques for robust strategy analysis.

**Classes:**
- `Bootstrap`: Core bootstrapping functionality
- `BootstrapTests`: Statistical significance testing
- `BootstrapRiskMetrics`: Risk metric estimation with confidence intervals
- `BootstrapPlotter`: Visualization tools for bootstrap results

**Example Usage:**
```python
from src.bootstrapping import AdvancedBootstrapping

# Create bootstrap instance
bootstrap = AdvancedBootstrapping(
    n_bootstrap=1000,
    confidence_level=0.95,
    method='circular'
)

# Bootstrap returns
bootstrap_results = bootstrap.bootstrap_returns(returns_data)

# Test statistical significance
significance_test = bootstrap.test_significance(
    strategy_returns, 
    benchmark_returns
)
```

#### 2. Strategies Module (`src.strategies`)

The strategies module provides base strategy framework and implementations.

**Classes:**
- `BaseStrategy`: Abstract base class for all strategies
- `MomentumStrategy`: Momentum-based trading strategy
- `MeanReversionStrategy`: Mean reversion trading strategy
- `TrendFollowingStrategy`: Trend following strategy
- `StrategyTestSuite`: Comprehensive strategy testing framework

**Example Usage:**
```python
from src.strategies import MomentumStrategy

# Create strategy
strategy = MomentumStrategy(
    lookback_period=20,
    signal_threshold=0.02
)

# Fit strategy to data
strategy.fit(price_data)

# Generate signals
signals = strategy.generate_signals(price_data)

# Calculate returns
returns = strategy.calculate_returns(price_data, signals)
```

#### 3. Analysis Module (`src.analysis`)

The analysis module provides comprehensive performance and risk analysis tools.

**Classes:**
- `PerformanceAnalyzer`: Performance metrics and attribution analysis
- `RiskAnalyzer`: Risk metrics and stress testing
- `PortfolioAnalyzer`: Portfolio optimization and analysis
- `ScenarioAnalyzer`: Scenario analysis and Monte Carlo simulations

**Example Usage:**
```python
from src.analysis import PerformanceAnalyzer, RiskAnalyzer

# Performance analysis
perf_analyzer = PerformanceAnalyzer()
metrics = perf_analyzer.calculate_metrics(returns)
attribution = perf_analyzer.performance_attribution(
    returns, 
    benchmark_returns, 
    factor_returns
)

# Risk analysis
risk_analyzer = RiskAnalyzer()
var_results = risk_analyzer.calculate_var(returns, confidence_level=0.05)
stress_results = risk_analyzer.stress_test(returns, scenarios)
```

#### 4. Utils Module (`src.utils`)

The utils module provides utility functions for data handling and calculations.

**Classes:**
- `DataLoader`: Data loading and preprocessing
- `PerformanceMetrics`: Performance metric calculations
- `RiskMetrics`: Risk metric calculations
- `PortfolioUtils`: Portfolio management utilities
- `ParameterOptimizer`: Strategy parameter optimization

**Functions:**
- `load_sample_data()`: Load sample market data
- `generate_synthetic_data()`: Generate synthetic data for testing
- `validate_data()`: Data validation and quality checks
- `calculate_metrics()`: Quick metric calculations

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd backtesting_engine

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Quick Start

```python
# Import the framework
from src import AdvancedBootstrapping, StrategyTestSuite
from src.utils import load_sample_data

# Load sample data
data = load_sample_data()

# Create and test a strategy
strategy_suite = StrategyTestSuite()
results = strategy_suite.run_comprehensive_test(data)

# Bootstrap analysis
bootstrap = AdvancedBootstrapping()
bootstrap_results = bootstrap.bootstrap_returns(results['returns'])

# Display results
print("Strategy Performance:")
print(f"Annual Return: {results['annual_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
print(f"Bootstrap CI: {bootstrap_results['confidence_interval']}")
```

### Configuration

The framework supports configuration through YAML files in the `config/` directory:

- `strategy_configs/`: Strategy-specific configurations
- Default parameters can be overridden programmatically

### Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_bootstrapping.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Examples

See the `examples/` directory for comprehensive examples:
- `bootstrapping_example.py`: Bootstrap analysis walkthrough
- `basic_strategy_example.py`: Basic strategy implementation
- `portfolio_optimization_example.py`: Portfolio optimization example

### Documentation

Additional documentation available in `docs/`:
- `tutorials/`: Step-by-step tutorials
- `examples/`: Advanced usage examples

### Support

For issues and questions:
1. Check the documentation in `docs/`
2. Review examples in `examples/`
3. Run tests to verify installation
4. Create an issue on GitHub

### License

This project is licensed under the MIT License. See `LICENSE` file for details.
