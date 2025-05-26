"""
Momentum Trading Strategy

This module implements various momentum-based trading strategies including
price momentum, volume momentum, and risk-adjusted momentum.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    """
    Basic momentum strategy based on price momentum.
    Goes long when momentum is positive, short when negative.
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        momentum_threshold: float = 0.02,
        **kwargs
    ):
        """
        Initialize momentum strategy.
        
        Args:
            lookback_period: Number of periods to calculate momentum
            momentum_threshold: Minimum momentum to trigger signal
        """
        super().__init__(
            name="Momentum",
            parameters={
                'lookback_period': lookback_period,
                'momentum_threshold': momentum_threshold
            },
            **kwargs
        )
        self.lookback_period = lookback_period
        self.momentum_threshold = momentum_threshold
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate momentum-based trading signals."""
        # Ensure we have enough historical data
        if len(data) <= self.lookback_period or current_idx < self.lookback_period:
            return Signal.HOLD
        
        # Calculate momentum using only historical data (no look-ahead bias)
        current_price = data.iloc[-1]['close']  # Current (most recent) price
        past_price = data.iloc[-(self.lookback_period + 1)]['close']  # Price lookback_period ago
        momentum = (current_price - past_price) / past_price
        
        if momentum > self.momentum_threshold:
            return Signal.BUY
        elif momentum < -self.momentum_threshold:
            return Signal.SELL
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'lookback_period': (5, 50),
            'momentum_threshold': (0.005, 0.1)
        }


class VolumeMomentumStrategy(BaseStrategy):
    """
    Volume-weighted momentum strategy.
    Considers both price momentum and volume confirmation.
    """
    
    def __init__(
        self,
        price_lookback: int = 20,
        volume_lookback: int = 10,
        momentum_threshold: float = 0.02,
        volume_threshold: float = 1.2,
        **kwargs
    ):
        """
        Initialize volume momentum strategy.
        
        Args:
            price_lookback: Periods for price momentum calculation
            volume_lookback: Periods for volume momentum calculation
            momentum_threshold: Minimum price momentum threshold
            volume_threshold: Minimum volume ratio threshold
        """
        super().__init__(
            name="VolumeMomentum",
            parameters={
                'price_lookback': price_lookback,
                'volume_lookback': volume_lookback,
                'momentum_threshold': momentum_threshold,
                'volume_threshold': volume_threshold
            },
            **kwargs
        )
        self.price_lookback = price_lookback
        self.volume_lookback = volume_lookback
        self.momentum_threshold = momentum_threshold
        self.volume_threshold = volume_threshold
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate volume-confirmed momentum signals."""
        if current_idx < max(self.price_lookback, self.volume_lookback):
            return Signal.HOLD
        
        # Calculate price momentum
        current_price = data.iloc[current_idx]['close']
        past_price = data.iloc[current_idx - self.price_lookback]['close']
        price_momentum = (current_price - past_price) / past_price
        
        # Calculate volume momentum
        recent_volume = data.iloc[current_idx - self.volume_lookback:current_idx]['volume'].mean()
        historical_volume = data.iloc[current_idx - 2*self.volume_lookback:current_idx - self.volume_lookback]['volume'].mean()
        volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1
        
        # Generate signal only if both price and volume confirm
        if (price_momentum > self.momentum_threshold and 
            volume_ratio > self.volume_threshold):
            return Signal.BUY
        elif (price_momentum < -self.momentum_threshold and 
              volume_ratio > self.volume_threshold):
            return Signal.SELL
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'price_lookback': (10, 50),
            'volume_lookback': (5, 20),
            'momentum_threshold': (0.01, 0.08),
            'volume_threshold': (1.1, 2.0)
        }


class RiskAdjustedMomentumStrategy(BaseStrategy):
    """
    Risk-adjusted momentum strategy using volatility scaling.
    Adjusts position sizes based on recent volatility.
    """
    
    def __init__(
        self,
        momentum_lookback: int = 20,
        volatility_lookback: int = 30,
        momentum_threshold: float = 0.02,
        volatility_target: float = 0.15,
        **kwargs
    ):
        """
        Initialize risk-adjusted momentum strategy.
        
        Args:
            momentum_lookback: Periods for momentum calculation
            volatility_lookback: Periods for volatility calculation
            momentum_threshold: Minimum momentum threshold
            volatility_target: Target annualized volatility
        """
        super().__init__(
            name="RiskAdjustedMomentum",
            parameters={
                'momentum_lookback': momentum_lookback,
                'volatility_lookback': volatility_lookback,
                'momentum_threshold': momentum_threshold,
                'volatility_target': volatility_target
            },
            **kwargs
        )
        self.momentum_lookback = momentum_lookback
        self.volatility_lookback = volatility_lookback
        self.momentum_threshold = momentum_threshold
        self.volatility_target = volatility_target
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate risk-adjusted momentum signals."""
        if current_idx < max(self.momentum_lookback, self.volatility_lookback):
            return Signal.HOLD
        
        # Calculate momentum
        current_price = data.iloc[current_idx]['close']
        past_price = data.iloc[current_idx - self.momentum_lookback]['close']
        momentum = (current_price - past_price) / past_price
        
        if abs(momentum) > self.momentum_threshold:
            return Signal.BUY if momentum > 0 else Signal.SELL
        else:
            return Signal.HOLD
    
    def calculate_position_size(
        self, 
        signal: Signal, 
        price: float, 
        data: pd.DataFrame, 
        current_idx: int
    ) -> float:
        """Calculate volatility-adjusted position size."""
        if signal == Signal.HOLD or current_idx < self.volatility_lookback:
            return 0.0
        
        # Calculate recent volatility
        returns = data.iloc[current_idx - self.volatility_lookback:current_idx]['close'].pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate volatility scaling factor
        vol_scalar = self.volatility_target / realized_vol if realized_vol > 0 else 1.0
        vol_scalar = min(vol_scalar, 3.0)  # Cap at 3x leverage
        
        # Base position size
        base_size = super().calculate_position_size(signal, price, data, current_idx)
        
        return base_size * vol_scalar
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'momentum_lookback': (10, 50),
            'volatility_lookback': (15, 60),
            'momentum_threshold': (0.01, 0.08),
            'volatility_target': (0.05, 0.3)
        }


class DualMomentumStrategy(BaseStrategy):
    """
    Dual momentum strategy combining absolute and relative momentum.
    """
    
    def __init__(
        self,
        absolute_lookback: int = 12,
        relative_lookback: int = 6,
        absolute_threshold: float = 0.0,
        **kwargs
    ):
        """
        Initialize dual momentum strategy.
        
        Args:
            absolute_lookback: Periods for absolute momentum
            relative_lookback: Periods for relative momentum
            absolute_threshold: Threshold for absolute momentum
        """
        super().__init__(
            name="DualMomentum",
            parameters={
                'absolute_lookback': absolute_lookback,
                'relative_lookback': relative_lookback,
                'absolute_threshold': absolute_threshold
            },
            **kwargs
        )
        self.absolute_lookback = absolute_lookback
        self.relative_lookback = relative_lookback
        self.absolute_threshold = absolute_threshold
        
        # For relative momentum, we need benchmark data
        self.benchmark_returns = None
    
    def set_benchmark(self, benchmark_data: pd.DataFrame):
        """Set benchmark data for relative momentum calculation."""
        self.benchmark_returns = benchmark_data['close'].pct_change().dropna()
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate dual momentum signals."""
        if current_idx < max(self.absolute_lookback, self.relative_lookback):
            return Signal.HOLD
        
        # Absolute momentum
        current_price = data.iloc[current_idx]['close']
        past_price_abs = data.iloc[current_idx - self.absolute_lookback]['close']
        absolute_momentum = (current_price - past_price_abs) / past_price_abs
        
        # Only proceed if absolute momentum is positive
        if absolute_momentum <= self.absolute_threshold:
            return Signal.HOLD
        
        # Relative momentum (if benchmark is available)
        if self.benchmark_returns is not None and len(self.benchmark_returns) > current_idx:
            asset_returns = data['close'].pct_change().dropna()
            
            if len(asset_returns) > current_idx and current_idx >= self.relative_lookback:
                asset_rel_return = asset_returns.iloc[current_idx - self.relative_lookback:current_idx].mean()
                benchmark_rel_return = self.benchmark_returns.iloc[current_idx - self.relative_lookback:current_idx].mean()
                
                if asset_rel_return > benchmark_rel_return:
                    return Signal.BUY
                else:
                    return Signal.HOLD
        
        # If no benchmark, use absolute momentum only
        return Signal.BUY if absolute_momentum > self.absolute_threshold else Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'absolute_lookback': (6, 24),
            'relative_lookback': (3, 12),
            'absolute_threshold': (-0.05, 0.05)
        }


class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    Cross-sectional momentum strategy for multiple assets.
    Ranks assets by momentum and goes long top performers, short bottom performers.
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        num_long: int = 1,
        num_short: int = 1,
        rebalance_frequency: int = 5,
        **kwargs
    ):
        """
        Initialize cross-sectional momentum strategy.
        
        Args:
            lookback_period: Periods for momentum calculation
            num_long: Number of assets to buy
            num_short: Number of assets to short
            rebalance_frequency: Rebalance every N periods
        """
        super().__init__(
            name="CrossSectionalMomentum",
            parameters={
                'lookback_period': lookback_period,
                'num_long': num_long,
                'num_short': num_short,
                'rebalance_frequency': rebalance_frequency
            },
            **kwargs
        )
        self.lookback_period = lookback_period
        self.num_long = num_long
        self.num_short = num_short
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance = 0
        
        # Multi-asset data
        self.asset_data = {}
    
    def set_universe(self, asset_data: Dict[str, pd.DataFrame]):
        """Set universe of assets for cross-sectional analysis."""
        self.asset_data = asset_data
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Generate cross-sectional momentum signals.
        Note: This is simplified for single asset. Full implementation
        would require modification of the backtesting framework.
        """
        if (current_idx < self.lookback_period or 
            current_idx - self.last_rebalance < self.rebalance_frequency):
            return Signal.HOLD
        
        # Calculate momentum for current asset
        current_price = data.iloc[current_idx]['close']
        past_price = data.iloc[current_idx - self.lookback_period]['close']
        momentum = (current_price - past_price) / past_price
        
        self.last_rebalance = current_idx
        
        # Simplified: assume this asset is in top momentum quartile
        # In practice, would rank against universe
        if momentum > 0.05:  # Top quartile threshold
            return Signal.BUY
        elif momentum < -0.05:  # Bottom quartile threshold
            return Signal.SELL
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'lookback_period': (10, 60),
            'num_long': (1, 5),
            'num_short': (1, 5),
            'rebalance_frequency': (1, 20)
        }
