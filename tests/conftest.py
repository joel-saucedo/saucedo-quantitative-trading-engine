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
    """Generate sample price data (OHLCV DataFrame) for testing."""
    dates = pd.date_range('2020-01-01', '2022-12-31', freq='B') # Use business days
    n_days = len(dates)
    
    np.random.seed(42)
    # Simulate daily returns first
    daily_returns = np.random.normal(0.0005, 0.015, n_days)
    
    # Create price series from returns
    prices = [100.0]  # Starting price
    for ret in daily_returns[:-1]: # Ensure prices align with n_days for OHLCV
        prices.append(prices[-1] * (1 + ret))
    prices = np.array(prices)

    if len(prices) != n_days:
        # Adjust if lengths don't match, simple truncation or padding if necessary
        # This can happen if daily_returns was used directly with n_days for cumprod
        prices = np.resize(prices, n_days)
        prices[0] = 100.0 # Ensure starting price
        for i in range(1, n_days):
            prices[i] = prices[i-1] * (1 + daily_returns[i-1]) # Recalculate if resized

    data = pd.DataFrame(index=dates)
    data['open'] = prices * (1 + np.random.normal(0, 0.002, n_days))
    data['close'] = prices # Close is the main simulated price
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
    data['volume'] = np.random.randint(100000, 1000000, n_days)
    
    return data.dropna()


@pytest.fixture
def bootstrap_returns_data(sample_price_data):
    """Generate sample returns data (pd.Series) for bootstrapping tests."""
    if sample_price_data.empty or 'close' not in sample_price_data.columns:
        return pd.Series(dtype=float) # Return empty series if data is bad
    return sample_price_data['close'].pct_change().dropna()


@pytest.fixture
def sample_returns_data(bootstrap_returns_data):
    """Alias for bootstrap_returns_data for broader use."""
    return bootstrap_returns_data


@pytest.fixture
def sample_price_series_data(sample_price_data):
    """Generate a sample price series (pd.Series) for drawdown tests."""
    if sample_price_data.empty or 'close' not in sample_price_data.columns:
        return pd.Series(dtype=float)
    return sample_price_data['close']


@pytest.fixture
def sample_multi_asset_returns_data():
    """Generate sample multi-asset returns data (pd.DataFrame)."""
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='B')
    n_days = len(dates)
    assets = ['AssetA', 'AssetB', 'AssetC']
    df = pd.DataFrame(index=dates)
    np.random.seed(42)
    for asset in assets:
        returns = np.random.normal(0.0005, np.random.uniform(0.01, 0.03), n_days)
        prices = 100 * (1 + returns).cumprod()
        df[asset] = pd.Series(prices, index=dates).pct_change()
    return df.dropna()


@pytest.fixture
def mock_strategy_returns(sample_returns_data):
    """Generate mock strategy returns (pd.Series)."""
    # Simulate some positive and negative periods
    np.random.seed(123)
    returns = sample_returns_data.copy() * np.random.choice([-0.5, 0.5, 1, 1.5], size=len(returns))
    return returns


@pytest.fixture
def mock_benchmark_returns(sample_returns_data):
    """Generate mock benchmark returns (pd.Series)."""
    np.random.seed(456)
    return sample_returns_data.copy() * np.random.normal(1, 0.2, size=len(sample_returns_data))

# Test configuration
pytest_plugins = []
