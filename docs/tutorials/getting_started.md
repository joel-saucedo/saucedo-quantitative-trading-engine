# Getting Started Tutorial

## Introduction

This tutorial will guide you through the basics of using the Trading Strategy Analyzer framework. You'll learn how to:

1. Load and prepare data
2. Implement and test trading strategies
3. Perform bootstrap analysis
4. Analyze performance and risk
5. Optimize portfolio allocation

## Prerequisites

- Python 3.8+
- Basic understanding of trading strategies
- Familiarity with pandas and numpy

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd backtesting_engine

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Tutorial 1: Basic Strategy Implementation

### Step 1: Load Sample Data

```python
import sys
import os
sys.path.append('.')

from src.utils import load_sample_data
import pandas as pd
import numpy as np

# Load sample market data
data = load_sample_data(
    start_date='2020-01-01',
    end_date='2023-12-31',
    frequency='daily'
)

print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(data.head())
```

### Step 2: Implement a Simple Strategy

```python
from src.strategies import MomentumStrategy

# Create momentum strategy
strategy = MomentumStrategy(
    lookback_period=20,      # 20-day momentum
    signal_threshold=0.02,   # 2% threshold
    position_size=0.1        # 10% position size
)

# Fit strategy to data (if needed)
strategy.fit(data)

# Generate trading signals
signals = strategy.generate_signals(data)

# Calculate strategy returns
returns = strategy.calculate_returns(data, signals)

print(f"Strategy generated {signals.sum()} buy signals")
print(f"Total return: {(1 + returns).prod() - 1:.2%}")
```

### Step 3: Basic Performance Analysis

```python
from src.analysis import PerformanceAnalyzer

# Create performance analyzer
analyzer = PerformanceAnalyzer()

# Calculate basic metrics
metrics = analyzer.calculate_metrics(returns)

print("Performance Metrics:")
for metric, value in metrics.items():
    if isinstance(value, float):
        if 'ratio' in metric.lower() or 'return' in metric.lower():
            print(f"{metric}: {value:.3f}")
        else:
            print(f"{metric}: {value:.2%}")
    else:
        print(f"{metric}: {value}")
```

## Tutorial 2: Bootstrap Analysis

### Step 1: Basic Bootstrap Resampling

```python
from src.bootstrapping import AdvancedBootstrapping

# Create bootstrap instance
bootstrap = AdvancedBootstrapping(
    n_bootstrap=1000,
    confidence_level=0.95,
    method='circular'  # Preserves autocorrelation
)

# Bootstrap the returns
bootstrap_results = bootstrap.bootstrap_returns(returns)

print("Bootstrap Results:")
print(f"Original Sharpe: {bootstrap_results['original_sharpe']:.3f}")
print(f"Bootstrap Mean Sharpe: {bootstrap_results['bootstrap_sharpe_mean']:.3f}")
print(f"95% CI: [{bootstrap_results['sharpe_ci'][0]:.3f}, {bootstrap_results['sharpe_ci'][1]:.3f}]")
```

### Step 2: Statistical Significance Testing

```python
from src.bootstrapping import BootstrapTests

# Create test instance
tester = BootstrapTests(n_bootstrap=1000)

# Test against benchmark (assuming zero returns as benchmark)
benchmark_returns = pd.Series(np.zeros(len(returns)), index=returns.index)

# Test for statistical significance
significance_result = tester.test_significance(
    returns, 
    benchmark_returns,
    test_type='sharpe_ratio'
)

print(f"P-value: {significance_result['p_value']:.4f}")
print(f"Statistically significant: {significance_result['is_significant']}")
print(f"Effect size: {significance_result['effect_size']:.3f}")
```

## Tutorial 3: Risk Analysis

### Step 1: Value at Risk (VaR) Analysis

```python
from src.analysis import RiskAnalyzer

# Create risk analyzer
risk_analyzer = RiskAnalyzer()

# Calculate VaR at different confidence levels
var_results = risk_analyzer.calculate_var(
    returns, 
    confidence_levels=[0.01, 0.05, 0.10],
    method='historical'
)

print("Value at Risk Results:")
for level, var_value in var_results.items():
    print(f"VaR ({level:.0%}): {var_value:.2%}")

# Expected Shortfall (Conditional VaR)
es_results = risk_analyzer.calculate_expected_shortfall(
    returns,
    confidence_levels=[0.01, 0.05, 0.10]
)

print("\nExpected Shortfall Results:")
for level, es_value in es_results.items():
    print(f"ES ({level:.0%}): {es_value:.2%}")
```

### Step 2: Stress Testing

```python
# Define stress scenarios
stress_scenarios = {
    'market_crash': {'shock_size': -0.20, 'duration': 5},
    'volatility_spike': {'vol_multiplier': 3.0, 'duration': 10},
    'correlation_breakdown': {'correlation_shock': 0.9, 'duration': 20}
}

# Run stress tests
stress_results = risk_analyzer.stress_test(
    returns, 
    scenarios=stress_scenarios
)

print("\nStress Test Results:")
for scenario, result in stress_results.items():
    print(f"{scenario}: {result['total_loss']:.2%} loss")
```

## Tutorial 4: Portfolio Optimization

### Step 1: Multi-Asset Portfolio

```python
from src.analysis import PortfolioAnalyzer
from src.utils import generate_synthetic_data

# Generate multi-asset data
portfolio_data = generate_synthetic_data(
    n_assets=5,
    n_periods=252*3,  # 3 years daily
    start_date='2021-01-01'
)

# Create portfolio analyzer
portfolio_analyzer = PortfolioAnalyzer()

# Mean-variance optimization
mv_result = portfolio_analyzer.mean_variance_optimization(
    portfolio_data,
    target_return=0.10,  # 10% annual target
    risk_aversion=5.0
)

print("Mean-Variance Optimization:")
print(f"Optimal weights: {mv_result['weights']}")
print(f"Expected return: {mv_result['expected_return']:.2%}")
print(f"Expected volatility: {mv_result['expected_volatility']:.2%}")
print(f"Sharpe ratio: {mv_result['sharpe_ratio']:.3f}")
```

### Step 2: Risk Parity Portfolio

```python
# Risk parity optimization
rp_result = portfolio_analyzer.risk_parity_optimization(portfolio_data)

print("\nRisk Parity Optimization:")
print(f"Risk parity weights: {rp_result['weights']}")
print(f"Risk contributions: {rp_result['risk_contributions']}")
print(f"Portfolio volatility: {rp_result['portfolio_volatility']:.2%}")
```

## Tutorial 5: Advanced Analysis

### Step 1: Monte Carlo Simulation

```python
from src.analysis import ScenarioAnalyzer

# Create scenario analyzer
scenario_analyzer = ScenarioAnalyzer()

# Monte Carlo simulation
mc_results = scenario_analyzer.monte_carlo_simulation(
    returns,
    n_simulations=10000,
    time_horizon=252,  # 1 year
    initial_value=100000  # $100,000 initial portfolio
)

print("Monte Carlo Results:")
print(f"Mean final value: ${mc_results['mean_final_value']:,.0f}")
print(f"5th percentile: ${mc_results['percentiles'][5]:,.0f}")
print(f"95th percentile: ${mc_results['percentiles'][95]:,.0f}")
print(f"Probability of loss: {mc_results['prob_loss']:.1%}")
```

### Step 2: Comprehensive Strategy Suite

```python
from src.strategies import StrategyTestSuite

# Create strategy test suite
suite = StrategyTestSuite()

# Run comprehensive tests on multiple strategies
comprehensive_results = suite.run_comprehensive_test(
    data,
    strategies=['momentum', 'mean_reversion', 'trend_following'],
    bootstrap_analysis=True,
    n_bootstrap=1000
)

print("\nComprehensive Strategy Results:")
for strategy_name, results in comprehensive_results.items():
    print(f"\n{strategy_name.upper()}:")
    print(f"  Annual Return: {results['annual_return']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Bootstrap CI (Sharpe): [{results['bootstrap']['sharpe_ci'][0]:.3f}, {results['bootstrap']['sharpe_ci'][1]:.3f}]")
```

## Best Practices

### 1. Data Quality
- Always validate input data using `validate_data()`
- Check for missing values and outliers
- Ensure proper date indexing

### 2. Strategy Development
- Start with simple strategies before adding complexity
- Use proper position sizing and risk management
- Backtest on out-of-sample data

### 3. Statistical Analysis
- Use bootstrap methods for robust confidence intervals
- Test for statistical significance
- Consider multiple testing corrections

### 4. Risk Management
- Monitor VaR and Expected Shortfall
- Perform regular stress testing
- Use portfolio diversification

### 5. Performance Evaluation
- Compare against appropriate benchmarks
- Use risk-adjusted metrics
- Consider transaction costs and market impact

## Next Steps

1. **Explore Examples**: Check out the `examples/` directory for more detailed use cases
2. **Read API Documentation**: Review `docs/API.md` for complete API reference
3. **Run Tests**: Execute the test suite to verify your installation
4. **Customize**: Implement your own strategies and analysis methods
5. **Contribute**: Submit improvements and new features

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you've installed all dependencies and the package is in your Python path
2. **Data Issues**: Verify data format and date indexing
3. **Memory Issues**: Reduce bootstrap iterations or data size for large datasets
4. **Convergence Issues**: Check optimization parameters and constraints

### Getting Help

- Check the documentation in `docs/`
- Review examples in `examples/`
- Run tests to identify issues
- Create GitHub issues for bugs

## Conclusion

This tutorial covered the basics of using the Trading Strategy Analyzer framework. The framework provides powerful tools for strategy development, statistical analysis, and risk management. Continue exploring the examples and documentation to unlock its full potential.
