"""
Data Loading and Generation Utilities

This module provides functions for loading historical data, generating synthetic
datasets, and handling various data sources for backtesting.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Comprehensive data loader for various financial data sources.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        
    def load_yahoo_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Load data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'SPY')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            data = data.reset_index()
            
            # Ensure required columns exist
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if 'date' not in data.columns and data.index.name == 'Date':
                data = data.reset_index()
                data.columns = [col.lower() if col != 'Date' else 'date' for col in data.columns]
            
            # Set date as index
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            return data[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            raise ValueError(f"Failed to load data for {symbol}: {str(e)}")
    
    def load_csv_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = pd.read_csv(filepath)
            
            # Try to infer date column
            date_cols = ['date', 'timestamp', 'time', 'datetime']
            date_col = None
            for col in date_cols:
                if col.lower() in data.columns.str.lower():
                    date_col = col
                    break
            
            if date_col:
                data[date_col] = pd.to_datetime(data[date_col])
                data.set_index(date_col, inplace=True)
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV data: {str(e)}")
    
    def load_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        column: str = 'close'
    ) -> pd.DataFrame:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            column: Which column to extract
            
        Returns:
            DataFrame with symbols as columns
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data = self.load_yahoo_data(symbol, start_date, end_date)
                data_dict[symbol] = data[column]
            except Exception as e:
                print(f"Warning: Failed to load {symbol}: {e}")
                continue
        
        if not data_dict:
            raise ValueError("No data could be loaded for any symbol")
        
        combined_data = pd.DataFrame(data_dict)
        return combined_data.dropna()


def generate_synthetic_data(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_price: float = 100.0,
    volatility: float = 0.2,
    drift: float = 0.05,
    regime_changes: bool = True,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic financial data using geometric Brownian motion.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
        initial_price: Starting price
        volatility: Annual volatility
        drift: Annual drift (expected return)
        regime_changes: Whether to include regime changes
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Time step (daily)
    dt = 1/252  # Daily time step in years
    
    # Generate returns with potential regime changes
    if regime_changes:
        # Create regime periods
        regime_length = n_days // 4  # 4 regimes
        regimes = []
        
        for i in range(4):
            if i % 2 == 0:  # Bull market
                regime_vol = volatility * 0.8
                regime_drift = drift * 1.5
            else:  # Bear market
                regime_vol = volatility * 1.3
                regime_drift = drift * 0.3
            
            regime_returns = np.random.normal(
                regime_drift * dt,
                regime_vol * np.sqrt(dt),
                regime_length
            )
            regimes.extend(regime_returns)
        
        # Adjust length to match date range
        returns = np.array(regimes[:n_days])
    else:
        # Standard geometric Brownian motion
        returns = np.random.normal(
            drift * dt,
            volatility * np.sqrt(dt),
            n_days
        )
    
    # Generate price series
    if n_days == 0:
        prices = []
    else:
        prices_arr = np.empty(n_days)
        prices_arr[0] = initial_price
        for i in range(1, n_days):
            # returns[i-1] is the return from period i-1 to i, affecting price[i]
            prices_arr[i] = prices_arr[i-1] * np.exp(returns[i-1])
        prices = prices_arr.tolist()
    
    # Create OHLC data
    data = []
    for i, price in enumerate(prices):
        # Generate intraday noise
        daily_vol = volatility * np.sqrt(dt) * 0.5  # Reduced intraday volatility
        noise = np.random.normal(0, daily_vol, 4)
        
        open_price = price * np.exp(noise[0])
        high_price = max(open_price, price) * (1 + abs(noise[1]))
        low_price = min(open_price, price) * (1 - abs(noise[2]))
        close_price = price
        
        # Generate volume (correlated with price volatility)
        base_volume = 1000000
        # Use the corresponding return for volume calculation, or default for first day
        ret_for_volume = returns[i] if i < len(returns) else 0.001
        vol_multiplier = 1 + abs(ret_for_volume) * 10  # Higher volume on big moves
        volume = int(base_volume * vol_multiplier * (0.5 + np.random.random()))
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=date_range)
    return df


def load_sample_data(dataset: str = "spy_sample") -> pd.DataFrame:
    """
    Load sample datasets for testing and examples.
    
    Args:
        dataset: Name of sample dataset
        
    Returns:
        Sample data DataFrame
    """
    if dataset == "spy_sample":
        # Generate SPY-like data
        return generate_synthetic_data(
            start_date="2020-01-01",
            end_date="2023-12-31",
            initial_price=300.0,
            volatility=0.18,
            drift=0.08,
            regime_changes=True,
            seed=42
        )
    
    elif dataset == "volatile_stock":
        # High volatility stock
        return generate_synthetic_data(
            start_date="2020-01-01",
            end_date="2023-12-31",
            initial_price=50.0,
            volatility=0.45,
            drift=0.12,
            regime_changes=True,
            seed=123
        )
    
    elif dataset == "stable_stock":
        # Low volatility stock
        return generate_synthetic_data(
            start_date="2020-01-01",
            end_date="2023-12-31",
            initial_price=80.0,
            volatility=0.12,
            drift=0.06,
            regime_changes=False,
            seed=456
        )
    
    elif dataset == "crypto_like":
        # Crypto-like high volatility
        return generate_synthetic_data(
            start_date="2020-01-01",
            end_date="2023-12-31",
            initial_price=20000.0,
            volatility=0.8,
            drift=0.15,
            regime_changes=True,
            seed=789
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_benchmark_data() -> Dict[str, pd.DataFrame]:
    """
    Create standard benchmark datasets for strategy comparison.
    
    Returns:
        Dictionary of benchmark datasets
    """
    benchmarks = {}
    
    # S&P 500 proxy
    benchmarks['sp500'] = load_sample_data("spy_sample")
    
    # High volatility benchmark
    benchmarks['high_vol'] = load_sample_data("volatile_stock")
    
    # Low volatility benchmark  
    benchmarks['low_vol'] = load_sample_data("stable_stock")
    
    # Trending market
    trending_data = generate_synthetic_data(
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_price=100.0,
        volatility=0.15,
        drift=0.12,  # Strong upward trend
        regime_changes=False,
        seed=999
    )
    benchmarks['trending'] = trending_data
    
    # Mean-reverting market
    mean_reverting_data = generate_synthetic_data(
        start_date="2020-01-01", 
        end_date="2023-12-31",
        initial_price=100.0,
        volatility=0.25,
        drift=0.02,  # Minimal drift
        regime_changes=False,
        seed=111
    )
    # Add mean reversion
    for i in range(1, len(mean_reverting_data)):
        if i % 30 == 0:  # Every 30 days, pull back to mean
            mean_price = mean_reverting_data['close'].iloc[:i].mean()
            current_price = mean_reverting_data['close'].iloc[i]
            reversion_factor = 0.1
            adjustment = (mean_price - current_price) * reversion_factor
            mean_reverting_data.iloc[i:i+10] *= (1 + adjustment/current_price)
    
    benchmarks['mean_reverting'] = mean_reverting_data
    
    return benchmarks


def prepare_data_for_strategy(
    data: pd.DataFrame,
    lookback_days: int = 100
) -> pd.DataFrame:
    """
    Prepare data for strategy backtesting by adding common indicators.
    
    Args:
        data: Raw OHLCV data
        lookback_days: Days of history to ensure availability
        
    Returns:
        Enhanced data with indicators
    """
    df = data.copy()
    
    # Add basic technical indicators
    # Simple moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential moving averages
    df['ema_10'] = df['close'].ewm(span=10).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # Volume indicators
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    
    # High-low indicators
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    
    # Ensure we have enough history
    df = df.iloc[lookback_days:].copy()
    
    return df


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        data: Full dataset
        train_ratio: Fraction for training
        validation_ratio: Fraction for validation
        
    Returns:
        Tuple of (train, validation, test) DataFrames
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    return train_data, val_data, test_data


def generate_multi_asset_data(
    n_assets: int = 3,
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    correlation_matrix: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic multi-asset return data.
    
    Args:
        n_assets: Number of assets
        start_date: Start date
        end_date: End date
        correlation_matrix: Correlation matrix for assets
        seed: Random seed
        
    Returns:
        DataFrame with returns for multiple assets
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_periods = len(date_range)
    
    # Default correlation matrix if not provided
    if correlation_matrix is None:
        correlation_matrix = np.eye(n_assets)
        # Add some correlation
        for i in range(n_assets):
            for j in range(n_assets):
                if i != j:
                    correlation_matrix[i, j] = 0.3
    
    # Generate correlated returns
    mean_returns = np.random.uniform(0.05, 0.15, n_assets) / 252  # Daily
    volatilities = np.random.uniform(0.15, 0.40, n_assets) / np.sqrt(252)  # Daily
    
    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(correlation_matrix)
    
    # Generate uncorrelated random returns
    uncorr_returns = np.random.normal(0, 1, (n_periods, n_assets))
    
    # Apply correlation
    corr_returns = uncorr_returns @ L.T
    
    # Scale by volatility and add drift
    final_returns = corr_returns * volatilities + mean_returns
    
    # Create asset names
    asset_names = [f'Asset_{i+1}' for i in range(n_assets)]
    
    return pd.DataFrame(final_returns, index=date_range, columns=asset_names)
