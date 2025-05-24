"""
Test Configuration

This file contains pytest configuration and fixtures for testing
the backtesting engine framework.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
    n_days = len(dates)
    
    # Generate realistic price data using geometric Brownian motion
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.015, n_days)  # Daily returns
    
    prices = [100.0]  # Starting price
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Add some intraday noise
        noise = np.random.normal(0, 0.002, 4)
        open_price = price * (1 + noise[0])
        high_price = max(open_price, price) * (1 + abs(noise[1]))
        low_price = min(open_price, price) * (1 - abs(noise[2]))
        close_price = price
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'open': open_price,
            'high': high_price, 
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_returns():
    """Generate sample returns series for testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='D')
    returns = np.random.normal(0.0005, 0.015, len(dates))
    return pd.Series(returns, index=dates)


@pytest.fixture
def sample_trades():
    """Generate sample trades for testing."""
    from src.strategies.base_strategy import Trade, Signal
    
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    trades = []
    
    for i in range(0, len(dates), 10):  # Create trade every 10 days
        if i + 5 < len(dates):
            entry_date = dates[i]
            exit_date = dates[i + 5]
            entry_price = 100 + np.random.normal(0, 5)
            exit_price = entry_price * (1 + np.random.normal(0.02, 0.05))
            size = 100
            
            pnl = (exit_price - entry_price) * size
            return_pct = (exit_price - entry_price) / entry_price
            
            trade = Trade(
                symbol='TEST',
                entry_time=entry_date,
                exit_time=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                pnl=pnl,
                return_pct=return_pct,
                duration=exit_date - entry_date,
                signal=Signal.BUY
            )
            trades.append(trade)
    
    return trades


@pytest.fixture
def bootstrap_data():
    """Generate data for bootstrap testing."""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    returns = np.random.normal(0.001, 0.02, len(dates))
    return pd.Series(returns, index=dates, name='returns')


# Test configuration
pytest_plugins = []
