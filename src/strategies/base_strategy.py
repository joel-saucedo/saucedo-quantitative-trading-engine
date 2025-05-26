"""
Base Strategy Class for Trading Strategy Framework

This module provides the abstract base class that all trading strategies must inherit from.
It defines the interface and common functionality for strategy implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Trading signals"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    size: float
    entry_price: float
    entry_time: pd.Timestamp
    signal: Signal
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    duration: pd.Timedelta
    signal: Signal
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement the generate_signals method and can optionally
    override other methods for custom behavior.
    """
    
    def __init__(
        self,
        name: str,
        parameters: Optional[Dict[str, Any]] = None,
        transaction_cost: float = 0.001,
        slippage: float = 0.0001,
        adversarial_slippage: bool = False,
        max_positions: int = 1,
        min_holding_period: int = 1
    ):
        """
        Initialize base strategy.
        
        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Slippage as fraction of trade value
            max_positions: Maximum number of concurrent positions
            min_holding_period: Minimum holding period in periods
        """
        self.name = name
        self.parameters = parameters or {}
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_positions = max_positions
        self.min_holding_period = min_holding_period
        
        # Strategy state
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.signals_history: List[Tuple[pd.Timestamp, Signal]] = []
        self.portfolio_value: List[float] = []
        self.cash: float = 100000.0  # Starting cash
        self.initial_capital = self.cash
        
        # Performance tracking
        self.returns: Optional[pd.Series] = None
        self.equity_curve: Optional[pd.Series] = None
        self.metrics: Optional[Dict[str, float]] = None
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """
        Generate trading signal based on current market data.
        
        CRITICAL: This method must NEVER access data beyond current_idx to prevent look-ahead bias.
        Only use data.iloc[:current_idx+1] or data.iloc[current_idx-lookback:current_idx+1]
        
        Args:
            data: Historical price data (only use up to current_idx)
            current_idx: Current time index (do NOT access beyond this)
            
        Returns:
            Trading signal (BUY, SELL, HOLD)
        """
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data before strategy execution.
        Override this method for custom preprocessing.
        
        Args:
            data: Raw price data
            
        Returns:
            Preprocessed data
        """
        # Default: ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return data.copy()
    
    def calculate_position_size(
        self, 
        signal: Signal, 
        price: float, 
        data: pd.DataFrame, 
        current_idx: int
    ) -> float:
        """
        Calculate position size for a given signal.
        Override this method for custom position sizing.
        
        Args:
            signal: Trading signal
            price: Current price
            data: Historical data
            current_idx: Current time index
            
        Returns:
            Position size (number of shares)
        """
        if signal == Signal.HOLD:
            return 0.0
            
        # Default: use fixed fraction of available cash
        available_cash = self.cash * 0.95  # Reserve 5% for costs
        position_value = available_cash / self.max_positions
        
        # Account for transaction costs
        effective_price = price * (1 + self.transaction_cost + self.slippage)
        position_size = position_value / effective_price
        
        return position_size if signal == Signal.BUY else -position_size
    
    def execute_trade(
        self, 
        signal: Signal, 
        price: float, 
        timestamp: pd.Timestamp, 
        symbol: str = "ASSET"
    ) -> bool:
        """
        Execute a trade based on signal.
        
        Args:
            signal: Trading signal
            price: Execution price
            timestamp: Trade timestamp
            symbol: Asset symbol
            
        Returns:
            True if trade was executed, False otherwise
        """
        if signal == Signal.HOLD:
            return False
            
        # Check if we can open new position
        if signal in [Signal.BUY, Signal.SELL] and len(self.positions) >= self.max_positions:
            return False
            
        position_size = self.calculate_position_size(signal, price, None, 0)
        
        if abs(position_size) < 0.01:  # Minimum position size
            return False
            
        # Calculate costs
        trade_value = abs(position_size * price)
        total_cost = trade_value * (self.transaction_cost + self.slippage)
        
        # Check if we have enough cash
        if signal == Signal.BUY and (trade_value + total_cost) > self.cash:
            return False
            
        # Create position
        position = Position(
            symbol=symbol,
            size=position_size,
            entry_price=price,
            entry_time=timestamp,
            signal=signal
        )
        
        self.positions.append(position)
        
        # Update cash
        if signal == Signal.BUY:
            self.cash -= (trade_value + total_cost)
        else:  # SELL signal - we need margin/collateral for short position
            self.cash -= total_cost  # Only deduct transaction costs for short
            
        return True
    
    def close_position(
        self, 
        position: Position, 
        price: float, 
        timestamp: pd.Timestamp
    ) -> Trade:
        """
        Close an existing position.
        
        Args:
            position: Position to close
            price: Exit price
            timestamp: Exit timestamp
            
        Returns:
            Completed trade object
        """
        # Calculate PnL
        if position.signal == Signal.BUY:
            pnl = (price - position.entry_price) * position.size
        else:
            pnl = (position.entry_price - price) * abs(position.size)
            
        # Account for exit costs
        trade_value = abs(position.size * price)
        exit_cost = trade_value * (self.transaction_cost + self.slippage)
        pnl -= exit_cost
        
        # Calculate return percentage
        initial_investment = abs(position.size * position.entry_price)
        return_pct = pnl / initial_investment if initial_investment > 0 else 0.0
        
        # Create trade record
        trade = Trade(
            symbol=position.symbol,
            entry_time=position.entry_time,
            exit_time=timestamp,
            entry_price=position.entry_price,
            exit_price=price,
            size=position.size,
            pnl=pnl,
            return_pct=return_pct,
            duration=timestamp - position.entry_time,
            signal=position.signal
        )
        
        self.trades.append(trade)
        
        # Update cash - return the position proceeds
        if position.signal == Signal.BUY:
            # For long positions: get back the sale proceeds
            self.cash += trade_value - exit_cost + pnl
        else:
            # For short positions: pay back the borrowed amount and keep profit/loss
            self.cash += pnl  # Only add/subtract the profit/loss
        
        # Remove position
        self.positions.remove(position)
        
        return trade
    
    def update_portfolio_value(self, current_prices: Dict[str, float]):
        """Update current portfolio value based on current prices."""
        total_value = self.cash
        
        for position in self.positions:
            current_price = current_prices.get(position.symbol, position.entry_price)
            if position.signal == Signal.BUY:
                # Long position: current value = shares * current_price
                position_value = position.size * current_price
            else:
                # Short position: profit/loss = shares * (entry_price - current_price)
                unrealized_pnl = abs(position.size) * (position.entry_price - current_price)
                position_value = unrealized_pnl  # Only add the unrealized P&L
            total_value += position_value
            
        self.portfolio_value.append(total_value)
        
    def backtest(
        self, 
        data: pd.DataFrame, 
        symbol: str = "ASSET",
        rebalance_frequency: str = "1D"
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data.
        
        Args:
            data: Historical price data
            symbol: Asset symbol
            rebalance_frequency: How often to rebalance
            
        Returns:
            Backtest results dictionary
        """
        # Reset strategy state
        self.positions = []
        self.trades = []
        self.signals_history = []
        self.portfolio_value = []
        self.cash = self.initial_capital
        
        # Preprocess data
        data = self.preprocess_data(data)
        
        # Run strategy
        for i, (timestamp, row) in enumerate(data.iterrows()):
            current_price = row['close']
            
            # CRITICAL: Only pass historical data up to current point to prevent look-ahead bias
            historical_data = data.iloc[:i+1]
            
            # Generate signal using only historical data
            signal = self.generate_signals(historical_data, i)
            self.signals_history.append((timestamp, signal))
            
            # Execute trades based on signal
            if signal != Signal.HOLD:
                # Close existing positions if signal changes
                for position in self.positions.copy():
                    if position.signal != signal:
                        self.close_position(position, current_price, timestamp)
                
                # Open new position if signal is not hold
                self.execute_trade(signal, current_price, timestamp, symbol)
            
            # Update portfolio value
            self.update_portfolio_value({symbol: current_price})
        
        # Close any remaining positions at the end
        if self.positions and len(data) > 0:
            final_price = data.iloc[-1]['close']
            final_timestamp = data.index[-1]
            for position in self.positions.copy():
                self.close_position(position, final_price, final_timestamp)
        
        # Calculate returns and metrics
        self._calculate_performance_metrics(data)
        
        return {
            'strategy': self.name,
            'parameters': self.parameters,
            'trades': self.trades,
            'portfolio_value': self.portfolio_value,
            'returns': self.returns,
            'metrics': self.metrics,
            'signals': self.signals_history
        }
    
    def _calculate_performance_metrics(self, data: pd.DataFrame):
        """Calculate performance metrics after backtest."""
        if not self.portfolio_value:
            return
            
        # Create equity curve
        portfolio_series = pd.Series(
            self.portfolio_value, 
            index=data.index[:len(self.portfolio_value)]
        )
        self.equity_curve = portfolio_series
        
        # Calculate returns
        self.returns = portfolio_series.pct_change().dropna()
        
        # Calculate basic metrics
        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(self.returns)) - 1
        volatility = self.returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        peak = portfolio_series.expanding(min_periods=1).max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        self.metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trades),
            'avg_trade_return': np.mean([t.return_pct for t in self.trades]) if self.trades else 0
        }
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get parameter bounds for optimization.
        Override this method to define parameter search space.
        
        Returns:
            Dictionary of parameter bounds
        """
        return {}
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """Update strategy parameters."""
        self.parameters.update(parameters)
    
    def __str__(self) -> str:
        return f"{self.name}({self.parameters})"
    
    def __repr__(self) -> str:
        return self.__str__()
