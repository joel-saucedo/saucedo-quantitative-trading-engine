"""
Mean Reversion Trading Strategies

This module implements various mean reversion strategies including
statistical arbitrage, bollinger bands, and pairs trading approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy, Signal


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands mean reversion strategy.
    Buys when price touches lower band, sells when touching upper band.
    """
    
    def __init__(
        self,
        window: int = 20,
        num_stds: float = 2.0,
        rsi_window: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        **kwargs
    ):
        """
        Initialize Bollinger Bands strategy.
        
        Args:
            window: Moving average window
            num_stds: Number of standard deviations for bands
            rsi_window: RSI calculation window
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
        """
        super().__init__(
            name="BollingerBands",
            parameters={
                'window': window,
                'num_stds': num_stds,
                'rsi_window': rsi_window,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought
            },
            **kwargs
        )
        self.window = window
        self.num_stds = num_stds
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
    
    def calculate_rsi(self, prices: pd.Series, window: int) -> float:
        """Calculate RSI for given price series."""
        if len(prices) < window + 1:
            return 50  # Neutral RSI
            
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.rolling(window=window).mean().iloc[-1]
        avg_loss = losses.rolling(window=window).mean().iloc[-1]
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate Bollinger Bands mean reversion signals."""
        if current_idx < max(self.window, self.rsi_window):
            return Signal.HOLD
        
        # Get recent price data
        prices = data.iloc[max(0, current_idx - self.window):current_idx + 1]['close']
        current_price = prices.iloc[-1]
        
        # Calculate Bollinger Bands
        sma = prices.rolling(window=self.window).mean().iloc[-1]
        std = prices.rolling(window=self.window).std().iloc[-1]
        upper_band = sma + (self.num_stds * std)
        lower_band = sma - (self.num_stds * std)
        
        # Calculate RSI for confirmation
        rsi_prices = data.iloc[max(0, current_idx - self.rsi_window - 10):current_idx + 1]['close']
        rsi = self.calculate_rsi(rsi_prices, self.rsi_window)
        
        # Generate signals
        if current_price <= lower_band and rsi <= self.rsi_oversold:
            return Signal.BUY  # Oversold, expect reversion up
        elif current_price >= upper_band and rsi >= self.rsi_overbought:
            return Signal.SELL  # Overbought, expect reversion down
        elif abs(current_price - sma) / sma < 0.01:  # Close to mean
            return Signal.HOLD  # Exit signal
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'window': (10, 50),
            'num_stds': (1.5, 3.0),
            'rsi_window': (10, 20),
            'rsi_oversold': (20, 35),
            'rsi_overbought': (65, 80)
        }


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical arbitrage strategy using z-score mean reversion.
    """
    
    def __init__(
        self,
        lookback_window: int = 60,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        stop_loss_threshold: float = 3.0,
        **kwargs
    ):
        """
        Initialize statistical arbitrage strategy.
        
        Args:
            lookback_window: Window for calculating statistics
            entry_threshold: Z-score threshold for entry
            exit_threshold: Z-score threshold for exit
            stop_loss_threshold: Z-score threshold for stop loss
        """
        super().__init__(
            name="StatisticalArbitrage",
            parameters={
                'lookback_window': lookback_window,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'stop_loss_threshold': stop_loss_threshold
            },
            **kwargs
        )
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
    
    def calculate_zscore(self, prices: pd.Series) -> float:
        """Calculate z-score of current price vs historical mean."""
        if len(prices) < 2:
            return 0.0
            
        mean = prices[:-1].mean()  # Exclude current price from mean
        std = prices[:-1].std()
        current_price = prices.iloc[-1]
        
        if std == 0:
            return 0.0
            
        return (current_price - mean) / std
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate statistical arbitrage signals based on z-score."""
        if current_idx < self.lookback_window:
            return Signal.HOLD
        
        # Get price window
        prices = data.iloc[current_idx - self.lookback_window:current_idx + 1]['close']
        z_score = self.calculate_zscore(prices)
        
        # Check if we have existing positions
        has_long_position = any(pos.signal == Signal.BUY for pos in self.positions)
        has_short_position = any(pos.signal == Signal.SELL for pos in self.positions)
        
        # Entry signals
        if not has_long_position and not has_short_position:
            if z_score <= -self.entry_threshold:
                return Signal.BUY  # Price is cheap relative to history
            elif z_score >= self.entry_threshold:
                return Signal.SELL  # Price is expensive relative to history
        
        # Exit signals for existing positions
        elif has_long_position:
            if z_score >= -self.exit_threshold or z_score <= -self.stop_loss_threshold:
                return Signal.SELL  # Close long position
        elif has_short_position:
            if z_score <= self.exit_threshold or z_score >= self.stop_loss_threshold:
                return Signal.BUY  # Close short position
        
        return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'lookback_window': (30, 120),
            'entry_threshold': (1.5, 3.0),
            'exit_threshold': (0.2, 1.0),
            'stop_loss_threshold': (2.5, 4.0)
        }


class PairsStrategy(BaseStrategy):
    """
    Pairs trading strategy using cointegration.
    Note: This is a simplified version for single asset demonstration.
    """
    
    def __init__(
        self,
        formation_period: int = 60,
        trading_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize pairs trading strategy.
        
        Args:
            formation_period: Period to form pairs relationship
            trading_period: Period to trade the pair
            entry_threshold: Standard deviations for entry
            exit_threshold: Standard deviations for exit
        """
        super().__init__(
            name="PairsTrading",
            parameters={
                'formation_period': formation_period,
                'trading_period': trading_period,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold
            },
            **kwargs
        )
        self.formation_period = formation_period
        self.trading_period = trading_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
        # Pairs-specific state
        self.formation_complete = False
        self.spread_mean = 0
        self.spread_std = 1
        self.last_formation_update = 0
        
        # Would typically have second asset data
        self.pair_data = None
    
    def set_pair_data(self, pair_data: pd.DataFrame):
        """Set data for the paired asset."""
        self.pair_data = pair_data
    
    def calculate_spread(self, data: pd.DataFrame, current_idx: int) -> float:
        """
        Calculate spread between assets.
        Simplified version using price ratio.
        """
        if self.pair_data is None or current_idx >= len(self.pair_data):
            # Fallback: use price vs its own moving average
            if current_idx < 20:
                return 0
            prices = data.iloc[current_idx-19:current_idx+1]['close']
            current_price = prices.iloc[-1]
            ma = prices.mean()
            return np.log(current_price / ma)
        
        # Real pairs calculation would be here
        asset1_price = data.iloc[current_idx]['close']
        asset2_price = self.pair_data.iloc[current_idx]['close']
        return np.log(asset1_price / asset2_price)
    
    def update_formation_statistics(self, data: pd.DataFrame, current_idx: int):
        """Update formation period statistics."""
        if current_idx - self.last_formation_update >= self.trading_period:
            # Recalculate formation statistics
            start_idx = max(0, current_idx - self.formation_period)
            spreads = []
            
            for i in range(start_idx, current_idx):
                spread = self.calculate_spread(data, i)
                spreads.append(spread)
            
            if spreads:
                self.spread_mean = np.mean(spreads)
                self.spread_std = np.std(spreads)
                self.formation_complete = True
                self.last_formation_update = current_idx
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate pairs trading signals."""
        if current_idx < self.formation_period:
            return Signal.HOLD
        
        # Update formation statistics periodically
        self.update_formation_statistics(data, current_idx)
        
        if not self.formation_complete or self.spread_std == 0:
            return Signal.HOLD
        
        # Calculate current spread z-score
        current_spread = self.calculate_spread(data, current_idx)
        z_score = (current_spread - self.spread_mean) / self.spread_std
        
        # Check existing positions
        has_long_position = any(pos.signal == Signal.BUY for pos in self.positions)
        has_short_position = any(pos.signal == Signal.SELL for pos in self.positions)
        
        # Entry signals
        if not has_long_position and not has_short_position:
            if z_score <= -self.entry_threshold:
                return Signal.BUY  # Spread is low, expect convergence
            elif z_score >= self.entry_threshold:
                return Signal.SELL  # Spread is high, expect convergence
        
        # Exit signals
        elif has_long_position and abs(z_score) <= self.exit_threshold:
            return Signal.SELL  # Close long position
        elif has_short_position and abs(z_score) <= self.exit_threshold:
            return Signal.BUY  # Close short position
        
        return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'formation_period': (40, 100),
            'trading_period': (10, 40),
            'entry_threshold': (1.5, 3.0),
            'exit_threshold': (0.2, 1.0)
        }


class OrnsteinUhlenbeckStrategy(BaseStrategy):
    """
    Mean reversion strategy based on Ornstein-Uhlenbeck process.
    Models price as mean-reverting process and trades based on expected reversion.
    """
    
    def __init__(
        self,
        estimation_window: int = 100,
        confidence_level: float = 0.95,
        reestimation_frequency: int = 20,
        **kwargs
    ):
        """
        Initialize Ornstein-Uhlenbeck strategy.
        
        Args:
            estimation_window: Window for parameter estimation
            confidence_level: Confidence level for trading bands
            reestimation_frequency: How often to reestimate parameters
        """
        super().__init__(
            name="OrnsteinUhlenbeck",
            parameters={
                'estimation_window': estimation_window,
                'confidence_level': confidence_level,
                'reestimation_frequency': reestimation_frequency
            },
            **kwargs
        )
        self.estimation_window = estimation_window
        self.confidence_level = confidence_level
        self.reestimation_frequency = reestimation_frequency
        
        # OU process parameters
        self.theta = 0  # Mean reversion speed
        self.mu = 0     # Long-term mean
        self.sigma = 1  # Volatility
        self.last_estimation = 0
    
    def estimate_ou_parameters(self, prices: pd.Series):
        """Estimate Ornstein-Uhlenbeck process parameters."""
        if len(prices) < 3:
            return
            
        log_prices = np.log(prices)
        returns = log_prices.diff().dropna()
        lagged_prices = log_prices.shift(1).dropna()
        
        if len(returns) != len(lagged_prices):
            returns = returns[:len(lagged_prices)]
        
        # OLS regression: dr = theta * (mu - r_lagged) * dt + sigma * dW
        # Simplified: dr = alpha + beta * r_lagged + error
        X = np.column_stack([np.ones(len(lagged_prices)), lagged_prices])
        y = returns.values
        
        try:
            params = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha, beta = params
            
            # Convert to OU parameters
            self.theta = -beta if beta < 0 else 0.1
            self.mu = -alpha / beta if beta != 0 else log_prices.mean()
            self.sigma = np.std(returns)
            
        except np.linalg.LinAlgError:
            # Fallback to simple estimates
            self.theta = 0.1
            self.mu = log_prices.mean()
            self.sigma = np.std(returns)
    
    def calculate_expected_price(self, current_price: float, dt: float = 1.0) -> float:
        """Calculate expected price after time dt."""
        log_current = np.log(current_price)
        log_expected = self.mu + (log_current - self.mu) * np.exp(-self.theta * dt)
        return np.exp(log_expected)
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate OU-based mean reversion signals."""
        if current_idx < self.estimation_window:
            return Signal.HOLD
        
        # Reestimate parameters periodically
        if current_idx - self.last_estimation >= self.reestimation_frequency:
            prices = data.iloc[current_idx - self.estimation_window:current_idx]['close']
            self.estimate_ou_parameters(prices)
            self.last_estimation = current_idx
        
        if self.theta <= 0:  # No mean reversion detected
            return Signal.HOLD
        
        current_price = data.iloc[current_idx]['close']
        expected_price = self.calculate_expected_price(current_price)
        
        # Calculate confidence bands
        log_current = np.log(current_price)
        log_mu = np.log(self.mu) if self.mu > 0 else log_current
        
        # Standard deviation of the process
        process_std = self.sigma / np.sqrt(2 * self.theta)
        z_alpha = 1.96 if self.confidence_level == 0.95 else 2.58  # 95% or 99%
        
        upper_band = np.exp(log_mu + z_alpha * process_std)
        lower_band = np.exp(log_mu - z_alpha * process_std)
        
        # Generate signals
        if current_price < lower_band:
            return Signal.BUY  # Price below confidence band
        elif current_price > upper_band:
            return Signal.SELL  # Price above confidence band
        else:
            return Signal.HOLD
    
    def get_parameter_bounds(self) -> Dict[str, tuple]:
        """Return parameter bounds for optimization."""
        return {
            'estimation_window': (50, 200),
            'confidence_level': (0.90, 0.99),
            'reestimation_frequency': (10, 50)
        }


class MeanReversionStrategy(BaseStrategy):
    """
    Basic mean reversion strategy.
    Buys when price is below moving average, sells when above.
    """
    
    def __init__(self, lookback_period: int = 20, **kwargs):
        super().__init__(
            name="MeanReversion",
            parameters={'lookback_period': lookback_period},
            **kwargs
        )
        self.lookback_period = lookback_period

    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        if current_idx < self.lookback_period:
            return Signal.HOLD
        
        prices = data.iloc[max(0, current_idx - self.lookback_period):current_idx + 1]['close']
        current_price = prices.iloc[-1]
        moving_average = prices.mean()
        
        if current_price < moving_average * 0.98:  # 2% below MA
            return Signal.BUY
        elif current_price > moving_average * 1.02:  # 2% above MA
            return Signal.SELL
        else:
            return Signal.HOLD
