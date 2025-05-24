"""
Trend Following Trading Strategies

This module implements various trend following strategies including
moving average crossovers, breakout strategies, and channel breakouts.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from .base_strategy import BaseStrategy, Signal


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Classic moving average crossover strategy.
    Generates buy signal when fast MA crosses above slow MA, sell when opposite.
    """
    
    def __init__(
        self,
        fast_window: int = 10,
        slow_window: int = 30,
        signal_smoothing: int = 3,
        **kwargs
    ):
        """
        Initialize moving average crossover strategy.
        
        Args:
            fast_window: Fast moving average window
            slow_window: Slow moving average window
            signal_smoothing: Smoothing window for signal generation
        """
        super().__init__(
            name="MovingAverageCrossover",
            parameters={
                'fast_window': fast_window,
                'slow_window': slow_window,
                'signal_smoothing': signal_smoothing
            },
            **kwargs
        )
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.signal_smoothing = signal_smoothing
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate moving average crossover signals."""
        if current_idx < max(self.slow_window, self.signal_smoothing):
            return Signal.HOLD
        
        # Calculate moving averages
        prices = data.iloc[max(0, current_idx - self.slow_window):current_idx + 1]['close']
        
        if len(prices) < self.slow_window:
            return Signal.HOLD
        
        fast_ma = prices.rolling(window=self.fast_window).mean()
        slow_ma = prices.rolling(window=self.slow_window).mean()
        
        # Get recent crossover signals
        signal_diff = fast_ma - slow_ma
        recent_diffs = signal_diff.tail(self.signal_smoothing)
        
        # Check for crossover
        if len(recent_diffs) >= 2:
            current_diff = recent_diffs.iloc[-1]
            previous_diff = recent_diffs.iloc[-2]
            
            # Golden cross (fast MA crosses above slow MA)
            if previous_diff <= 0 and current_diff > 0:
                return Signal.BUY
            # Death cross (fast MA crosses below slow MA)
            elif previous_diff >= 0 and current_diff < 0:
                return Signal.SELL
        
        return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'fast_window': (5, 20),
            'slow_window': (20, 50),
            'signal_smoothing': (1, 5)
        }


class TripleMovingAverageStrategy(BaseStrategy):
    """
    Triple moving average strategy using short, medium, and long-term MAs.
    """
    
    def __init__(
        self,
        short_window: int = 5,
        medium_window: int = 20,
        long_window: int = 50,
        **kwargs
    ):
        """
        Initialize triple moving average strategy.
        
        Args:
            short_window: Short-term MA window
            medium_window: Medium-term MA window
            long_window: Long-term MA window
        """
        super().__init__(
            name="TripleMovingAverage",
            parameters={
                'short_window': short_window,
                'medium_window': medium_window,
                'long_window': long_window
            },
            **kwargs
        )
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate triple MA signals."""
        if current_idx < self.long_window:
            return Signal.HOLD
        
        # Calculate all moving averages
        prices = data.iloc[max(0, current_idx - self.long_window):current_idx + 1]['close']
        
        short_ma = prices.rolling(window=self.short_window).mean().iloc[-1]
        medium_ma = prices.rolling(window=self.medium_window).mean().iloc[-1]
        long_ma = prices.rolling(window=self.long_window).mean().iloc[-1]
        
        # Strong bullish: short > medium > long
        if short_ma > medium_ma > long_ma:
            return Signal.BUY
        # Strong bearish: short < medium < long
        elif short_ma < medium_ma < long_ma:
            return Signal.SELL
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'short_window': (3, 10),
            'medium_window': (15, 30),
            'long_window': (40, 80)
        }


class BreakoutStrategy(BaseStrategy):
    """
    Price breakout strategy based on support/resistance levels.
    """
    
    def __init__(
        self,
        lookback_window: int = 20,
        breakout_threshold: float = 0.02,
        volume_confirmation: bool = True,
        volume_threshold: float = 1.5,
        **kwargs
    ):
        """
        Initialize breakout strategy.
        
        Args:
            lookback_window: Window to identify support/resistance
            breakout_threshold: Minimum breakout percentage
            volume_confirmation: Whether to require volume confirmation
            volume_threshold: Volume multiple for confirmation
        """
        super().__init__(
            name="Breakout",
            parameters={
                'lookback_window': lookback_window,
                'breakout_threshold': breakout_threshold,
                'volume_confirmation': volume_confirmation,
                'volume_threshold': volume_threshold
            },
            **kwargs
        )
        self.lookback_window = lookback_window
        self.breakout_threshold = breakout_threshold
        self.volume_confirmation = volume_confirmation
        self.volume_threshold = volume_threshold
    
    def identify_support_resistance(self, data: pd.DataFrame, current_idx: int) -> tuple:
        """Identify support and resistance levels."""
        if current_idx < self.lookback_window:
            return None, None
        
        window_data = data.iloc[current_idx - self.lookback_window:current_idx]
        
        resistance = window_data['high'].max()
        support = window_data['low'].min()
        
        return support, resistance
    
    def check_volume_confirmation(self, data: pd.DataFrame, current_idx: int) -> bool:
        """Check if current volume confirms the breakout."""
        if not self.volume_confirmation or current_idx < self.lookback_window:
            return True
        
        current_volume = data.iloc[current_idx]['volume']
        avg_volume = data.iloc[current_idx - self.lookback_window:current_idx]['volume'].mean()
        
        return current_volume >= (avg_volume * self.volume_threshold)
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate breakout signals."""
        if current_idx < self.lookback_window:
            return Signal.HOLD
        
        support, resistance = self.identify_support_resistance(data, current_idx)
        if support is None or resistance is None:
            return Signal.HOLD
        
        current_price = data.iloc[current_idx]['close']
        current_high = data.iloc[current_idx]['high']
        current_low = data.iloc[current_idx]['low']
        
        # Calculate breakout thresholds
        resistance_breakout = resistance * (1 + self.breakout_threshold)
        support_breakout = support * (1 - self.breakout_threshold)
        
        # Check for breakouts
        volume_confirmed = self.check_volume_confirmation(data, current_idx)
        
        if current_high > resistance_breakout and volume_confirmed:
            return Signal.BUY  # Upward breakout
        elif current_low < support_breakout and volume_confirmed:
            return Signal.SELL  # Downward breakout
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'lookback_window': (10, 40),
            'breakout_threshold': (0.01, 0.05),
            'volume_threshold': (1.2, 2.5)
        }


class ChannelBreakoutStrategy(BaseStrategy):
    """
    Donchian Channel breakout strategy.
    """
    
    def __init__(
        self,
        entry_window: int = 20,
        exit_window: int = 10,
        atr_window: int = 14,
        risk_multiple: float = 2.0,
        **kwargs
    ):
        """
        Initialize channel breakout strategy.
        
        Args:
            entry_window: Window for entry channel
            exit_window: Window for exit channel
            atr_window: Window for ATR calculation
            risk_multiple: Risk multiple for position sizing
        """
        super().__init__(
            name="ChannelBreakout",
            parameters={
                'entry_window': entry_window,
                'exit_window': exit_window,
                'atr_window': atr_window,
                'risk_multiple': risk_multiple
            },
            **kwargs
        )
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.atr_window = atr_window
        self.risk_multiple = risk_multiple
    
    def calculate_atr(self, data: pd.DataFrame, current_idx: int) -> float:
        """Calculate Average True Range."""
        if current_idx < self.atr_window:
            return data.iloc[current_idx]['high'] - data.iloc[current_idx]['low']
        
        window_data = data.iloc[current_idx - self.atr_window:current_idx + 1]
        
        true_ranges = []
        for i in range(1, len(window_data)):
            high = window_data.iloc[i]['high']
            low = window_data.iloc[i]['low']
            prev_close = window_data.iloc[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        return np.mean(true_ranges) if true_ranges else 0
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate channel breakout signals."""
        if current_idx < max(self.entry_window, self.exit_window):
            return Signal.HOLD
        
        current_price = data.iloc[current_idx]['close']
        
        # Calculate entry channels
        entry_data = data.iloc[current_idx - self.entry_window:current_idx]
        upper_channel = entry_data['high'].max()
        lower_channel = entry_data['low'].min()
        
        # Calculate exit channels
        exit_data = data.iloc[current_idx - self.exit_window:current_idx]
        exit_upper = exit_data['high'].max()
        exit_lower = exit_data['low'].min()
        
        # Check existing positions
        has_long_position = any(pos.signal == Signal.BUY for pos in self.positions)
        has_short_position = any(pos.signal == Signal.SELL for pos in self.positions)
        
        # Entry signals
        if not has_long_position and not has_short_position:
            if current_price > upper_channel:
                return Signal.BUY  # Upward breakout
            elif current_price < lower_channel:
                return Signal.SELL  # Downward breakout
        
        # Exit signals
        elif has_long_position and current_price < exit_lower:
            return Signal.SELL  # Exit long position
        elif has_short_position and current_price > exit_upper:
            return Signal.BUY  # Exit short position
        
        return Signal.HOLD
    
    def calculate_position_size(
        self, 
        signal: Signal, 
        price: float, 
        data: pd.DataFrame, 
        current_idx: int
    ) -> float:
        """Calculate ATR-based position size."""
        if signal == Signal.HOLD:
            return 0.0
        
        atr = self.calculate_atr(data, current_idx)
        if atr == 0:
            return super().calculate_position_size(signal, price, data, current_idx)
        
        # Risk-based position sizing
        risk_per_share = atr * self.risk_multiple
        risk_amount = self.cash * 0.02  # Risk 2% of capital per trade
        
        position_size = risk_amount / risk_per_share
        return position_size if signal == Signal.BUY else -position_size
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'entry_window': (15, 40),
            'exit_window': (5, 20),
            'atr_window': (10, 20),
            'risk_multiple': (1.0, 3.0)
        }


class TrendFilterStrategy(BaseStrategy):
    """
    Trend following strategy with multiple trend filters.
    """
    
    def __init__(
        self,
        short_ma: int = 10,
        long_ma: int = 50,
        adx_window: int = 14,
        adx_threshold: float = 25,
        rsi_window: int = 14,
        rsi_overbought: float = 70,
        rsi_oversold: float = 30,
        **kwargs
    ):
        """
        Initialize trend filter strategy.
        
        Args:
            short_ma: Short moving average window
            long_ma: Long moving average window
            adx_window: ADX calculation window
            adx_threshold: Minimum ADX for trend confirmation
            rsi_window: RSI calculation window
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
        """
        super().__init__(
            name="TrendFilter",
            parameters={
                'short_ma': short_ma,
                'long_ma': long_ma,
                'adx_window': adx_window,
                'adx_threshold': adx_threshold,
                'rsi_window': rsi_window,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold
            },
            **kwargs
        )
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.adx_window = adx_window
        self.adx_threshold = adx_threshold
        self.rsi_window = rsi_window
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
    
    def calculate_adx(self, data: pd.DataFrame, current_idx: int) -> float:
        """Calculate Average Directional Index."""
        if current_idx < self.adx_window + 1:
            return 25  # Default neutral value
        
        window_data = data.iloc[current_idx - self.adx_window:current_idx + 1]
        
        # Calculate True Range and Directional Movement
        tr_list = []
        dm_plus_list = []
        dm_minus_list = []
        
        for i in range(1, len(window_data)):
            high = window_data.iloc[i]['high']
            low = window_data.iloc[i]['low']
            close = window_data.iloc[i]['close']
            prev_high = window_data.iloc[i-1]['high']
            prev_low = window_data.iloc[i-1]['low']
            prev_close = window_data.iloc[i-1]['close']
            
            # True Range
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
            
            # Directional Movement
            dm_plus = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
            dm_minus = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
            
            dm_plus_list.append(dm_plus)
            dm_minus_list.append(dm_minus)
        
        # Calculate smoothed values
        if not tr_list:
            return 25
        
        atr = np.mean(tr_list)
        di_plus = np.mean(dm_plus_list) / atr * 100 if atr > 0 else 0
        di_minus = np.mean(dm_minus_list) / atr * 100 if atr > 0 else 0
        
        # Calculate ADX
        if di_plus + di_minus == 0:
            return 25
        
        dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
        return dx  # Simplified ADX calculation
    
    def calculate_rsi(self, data: pd.DataFrame, current_idx: int) -> float:
        """Calculate RSI."""
        if current_idx < self.rsi_window:
            return 50
        
        prices = data.iloc[current_idx - self.rsi_window:current_idx + 1]['close']
        deltas = prices.diff().dropna()
        
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate trend filter signals."""
        if current_idx < max(self.long_ma, self.adx_window, self.rsi_window):
            return Signal.HOLD
        
        # Calculate moving averages
        prices = data.iloc[current_idx - self.long_ma:current_idx + 1]['close']
        short_ma_val = prices.rolling(window=self.short_ma).mean().iloc[-1]
        long_ma_val = prices.rolling(window=self.long_ma).mean().iloc[-1]
        
        # Calculate trend strength (ADX)
        adx = self.calculate_adx(data, current_idx)
        
        # Calculate momentum (RSI)
        rsi = self.calculate_rsi(data, current_idx)
        
        # Trend direction
        uptrend = short_ma_val > long_ma_val
        strong_trend = adx > self.adx_threshold
        
        # Generate signals with all filters
        if uptrend and strong_trend and rsi < self.rsi_overbought:
            return Signal.BUY
        elif not uptrend and strong_trend and rsi > self.rsi_oversold:
            return Signal.SELL
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'short_ma': (5, 20),
            'long_ma': (30, 80),
            'adx_window': (10, 20),
            'adx_threshold': (20, 35),
            'rsi_window': (10, 20),
            'rsi_overbought': (65, 80),
            'rsi_oversold': (20, 35)
        }
