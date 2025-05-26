"""
Simple Pairs Trading Strategy

A simplified pairs trading strategy that works with the base strategy framework.
This strategy trades the ratio between two correlated assets (like BTC/ETH).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy, Signal


class SimplePairsStrategy(BaseStrategy):
    """
    Simple pairs trading strategy based on z-score of price ratio.
    """
    
    def __init__(
        self,
        lookback_period: int = 30,
        z_entry_threshold: float = 2.0,
        z_exit_threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize pairs strategy.
        
        Args:
            lookback_period: Number of periods to calculate rolling statistics
            z_entry_threshold: Z-score threshold to enter position
            z_exit_threshold: Z-score threshold to exit position
        """
        super().__init__(
            name="SimplePairs",
            parameters={
                'lookback_period': lookback_period,
                'z_entry_threshold': z_entry_threshold,
                'z_exit_threshold': z_exit_threshold
            },
            **kwargs
        )
        self.lookback_period = lookback_period
        self.z_entry_threshold = z_entry_threshold
        self.z_exit_threshold = z_exit_threshold
        
        # For pairs trading, we'll use the price itself as a proxy for the ratio
        # In a real implementation, you'd need two assets
        self.price_ratios = []
        self.z_scores = []
        self.in_position = False
        
    def generate_signals(self, historical_data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate pairs trading signals based on z-score."""
        if current_idx < self.lookback_period:
            return Signal.HOLD
        
        # For simplicity, we'll use the moving average of price as our "ratio"
        # In a real pairs strategy, this would be price_a / price_b
        current_price = historical_data.iloc[current_idx]['close']
        
        # Calculate moving average and standard deviation
        prices = historical_data.iloc[max(0, current_idx - self.lookback_period):current_idx + 1]['close']
        rolling_mean = prices.mean()
        rolling_std = prices.std()
        
        if rolling_std == 0:
            return Signal.HOLD
            
        # Calculate z-score
        z_score = (current_price - rolling_mean) / rolling_std
        self.z_scores.append(z_score)
        
        # Generate signals based on z-score
        if not self.in_position:
            if z_score > self.z_entry_threshold:
                self.in_position = True
                return Signal.SELL  # Price is too high, expect reversion
            elif z_score < -self.z_entry_threshold:
                self.in_position = True
                return Signal.BUY   # Price is too low, expect reversion
        else:
            # Exit condition: z-score returns to normal range
            if abs(z_score) < self.z_exit_threshold:
                self.in_position = False
                return Signal.HOLD  # This will close the position
                
        return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'lookback_period': (10, 60),
            'z_entry_threshold': (1.0, 3.0),
            'z_exit_threshold': (0.1, 1.0)
        }


class TrendFollowingStrategy(BaseStrategy):
    """
    Simple trend following strategy using moving average crossovers.
    """
    
    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        **kwargs
    ):
        """
        Initialize trend following strategy.
        
        Args:
            short_window: Short moving average period
            long_window: Long moving average period
        """
        super().__init__(
            name="TrendFollowing",
            parameters={
                'short_window': short_window,
                'long_window': long_window
            },
            **kwargs
        )
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, historical_data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate trend following signals."""
        if current_idx < self.long_window:
            return Signal.HOLD
        
        # Calculate moving averages
        prices = historical_data.iloc[max(0, current_idx - self.long_window):current_idx + 1]['close']
        
        short_ma = prices.iloc[-self.short_window:].mean()
        long_ma = prices.mean()
        
        # Generate signal based on moving average crossover
        if short_ma > long_ma:
            return Signal.BUY   # Uptrend
        elif short_ma < long_ma:
            return Signal.SELL  # Downtrend
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'short_window': (5, 20),
            'long_window': (20, 50)
        }


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    Volatility breakout strategy that trades on price movements beyond normal volatility.
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        breakout_threshold: float = 2.0,
        **kwargs
    ):
        """
        Initialize volatility breakout strategy.
        
        Args:
            volatility_window: Window for calculating volatility
            breakout_threshold: Number of standard deviations for breakout signal
        """
        super().__init__(
            name="VolatilityBreakout",
            parameters={
                'volatility_window': volatility_window,
                'breakout_threshold': breakout_threshold
            },
            **kwargs
        )
        self.volatility_window = volatility_window
        self.breakout_threshold = breakout_threshold
        
    def generate_signals(self, historical_data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate volatility breakout signals."""
        if current_idx < self.volatility_window:
            return Signal.HOLD
        
        # Calculate returns and volatility
        prices = historical_data.iloc[max(0, current_idx - self.volatility_window):current_idx + 1]['close']
        returns = prices.pct_change().dropna()
        
        if len(returns) < 2:
            return Signal.HOLD
            
        # Current return
        current_return = returns.iloc[-1]
        
        # Historical volatility
        volatility = returns.std()
        
        if volatility == 0:
            return Signal.HOLD
            
        # Check for breakout
        breakout_level = self.breakout_threshold * volatility
        
        if current_return > breakout_level:
            return Signal.BUY   # Positive breakout
        elif current_return < -breakout_level:
            return Signal.SELL  # Negative breakout
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'volatility_window': (10, 40),
            'breakout_threshold': (1.5, 3.0)
        }
