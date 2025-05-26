# Trading Strategies

This directory contains the implemented trading strategies for the quantitative trading engine.

## Available Strategies

### Base Strategy
- `base_strategy.py` - Abstract base class for all trading strategies

### Core Strategies
- `momentum.py` - Momentum-based trading strategy
- `mean_reversion.py` - Mean reversion trading strategy

### Advanced Strategies
- `composite_pair_trading_strategy.py` - Composite pair trading strategy with pseudo-cointegration and transfer entropy

## Usage

All strategies inherit from `BaseStrategy` and implement the required methods:
- `generate_signals()` - Generate trading signals
- `backtest()` - Backtest the strategy
- `calculate_returns()` - Calculate strategy returns

Example:
```python
from src.strategies import MomentumStrategy

strategy = MomentumStrategy(lookback_period=10)
results = strategy.backtest(data)
```

## Strategy Development

When developing new strategies:
1. Inherit from `BaseStrategy`
2. Implement required abstract methods
3. Add comprehensive docstrings
4. Include parameter validation
5. Add unit tests in the `tests/` directory
