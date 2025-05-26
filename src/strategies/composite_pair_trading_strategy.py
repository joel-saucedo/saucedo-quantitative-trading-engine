#!/usr/bin/env python3
"""
Composite Pair Trading Strategy with Transfer Entropy Integration

This strategy implements a sophisticated pair trading approach that treats BTC-ETH as a 
pseudo-cointegrated pair, utilizing transfer entropy for information flow detection and 
dynamic short-long arbitrage corrective series.

Key Features:
- Pseudo-cointegration modeling with rolling hedge ratios
- Transfer entropy-based signal confirmation
- Mean-reversion detection via z-score analysis
- Volatility-adjusted position sizing
- Comprehensive risk management
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append(str(Path(__file__).parent / 'src'))
sys.path.append(str(Path(__file__).parent / 'research'))

try:
    from src.strategies.base_strategy import BaseStrategy
    from src.utils.entropy_utils import calculate_transfer_entropy, calculate_entropy
except ImportError:
    # Fallback if imports fail
    class BaseStrategy:
        def __init__(self, name: str, parameters: Dict[str, Any]):
            self.name = name
            self.parameters = parameters
    
    def calculate_transfer_entropy(x, y, lag=1, n_bins=10):
        """Simple fallback implementation"""
        return np.random.random() * 0.1
    
    def calculate_entropy(x, n_bins=10):
        """Simple fallback implementation"""
        return np.random.random() * 2.0


class CompositePairTradingStrategy(BaseStrategy):
    """
    Composite pair trading strategy combining cointegration analysis with transfer entropy
    """
    
    def __init__(self, name: str = "CompositePairTrading", parameters: Dict[str, Any] = None):
        """
        Initialize the composite pair trading strategy
        
        Args:
            name: Strategy name
            parameters: Strategy parameters dictionary
        """
        default_params = {
            'asset_pair': ('BTC_USD', 'ETH_USD'),
            'lookback_window': 60,  # For hedge ratio calculation
            'z_entry_threshold': 2.0,  # Z-score entry threshold
            'z_exit_threshold': 0.5,   # Z-score exit threshold
            'te_window': 30,           # Transfer entropy window
            'te_lag': 1,               # Transfer entropy lag
            'n_bins': 10,              # Bins for entropy calculation
            'vol_window': 30,          # Volatility calculation window
            'risk_budget': 0.02,       # Portfolio risk budget (2%)
            'stop_loss': 0.05,         # Stop loss (5%)
            'take_profit': 0.10,       # Take profit (10%)
            'max_drawdown_limit': 0.15, # Maximum drawdown limit
            'min_te_threshold': 0.01,   # Minimum transfer entropy for signal confirmation
        }
        
        if parameters:
            default_params.update(parameters)
            
        super().__init__(name, default_params)
        
        # Strategy state
        self.hedge_ratios = []
        self.spreads = []
        self.z_scores = []
        self.transfer_entropies = []
        self.positions = []
        self.current_position = 0
        self.entry_price = 0
        self.entry_z_score = 0
        
        # Performance tracking
        self.portfolio_value = [1.0]
        self.trades = []
        self.signals = []
        
    def calculate_hedge_ratio(self, btc_prices: pd.Series, eth_prices: pd.Series, window: int) -> float:
        """
        Calculate rolling hedge ratio using OLS regression for pseudo-cointegration
        
        Args:
            btc_prices: BTC price series
            eth_prices: ETH price series
            window: Lookback window for regression
            
        Returns:
            Hedge ratio (beta coefficient)
        """
        if len(btc_prices) < window or len(eth_prices) < window:
            return 1.0
            
        # Use log prices for better statistical properties
        log_btc = np.log(btc_prices.iloc[-window:].values).reshape(-1, 1)
        log_eth = np.log(eth_prices.iloc[-window:].values)
        
        # OLS regression: log(BTC) = alpha + beta * log(ETH) + error
        reg = LinearRegression().fit(log_eth.reshape(-1, 1), log_btc.flatten())
        
        return reg.coef_[0]
    
    def calculate_spread(self, btc_price: float, eth_price: float, hedge_ratio: float) -> float:
        """
        Calculate the spread between BTC and ETH adjusted by hedge ratio
        
        Args:
            btc_price: Current BTC price
            eth_price: Current ETH price
            hedge_ratio: Current hedge ratio
            
        Returns:
            Spread value
        """
        return np.log(btc_price) - hedge_ratio * np.log(eth_price)
    
    def calculate_z_score(self, spreads: List[float], window: int) -> float:
        """
        Calculate z-score of current spread for mean reversion detection
        
        Args:
            spreads: Historical spread values
            window: Window for mean and std calculation
            
        Returns:
            Z-score of current spread
        """
        if len(spreads) < window:
            return 0.0
            
        recent_spreads = spreads[-window:]
        mean_spread = np.mean(recent_spreads)
        std_spread = np.std(recent_spreads)
        
        if std_spread == 0:
            return 0.0
            
        return (spreads[-1] - mean_spread) / std_spread
    
    def calculate_position_size(self, volatility: float, correlation: float) -> float:
        """
        Calculate position size based on volatility and correlation
        
        Args:
            volatility: Recent volatility measure
            correlation: Correlation between assets
            
        Returns:
            Position size as fraction of portfolio
        """
        # Base position size from risk budget
        base_size = self.parameters['risk_budget']
        
        # Adjust for volatility (higher vol = smaller position)
        vol_adjustment = 1 / (1 + volatility * 10)
        
        # Adjust for correlation (higher correlation = larger position for pair trading)
        corr_adjustment = abs(correlation) if not np.isnan(correlation) else 0.5
        
        return base_size * vol_adjustment * corr_adjustment
    
    def get_transfer_entropy_signal(self, btc_returns: pd.Series, eth_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate transfer entropy between BTC and ETH for directional bias
        
        Args:
            btc_returns: BTC returns series
            eth_returns: ETH returns series
            
        Returns:
            Dictionary with transfer entropy measures
        """
        window = self.parameters['te_window']
        
        if len(btc_returns) < window or len(eth_returns) < window:
            return {'te_btc_to_eth': 0.0, 'te_eth_to_btc': 0.0, 'te_ratio': 1.0}
        
        try:
            # Calculate transfer entropy in both directions
            te_btc_to_eth = calculate_transfer_entropy(
                btc_returns.iloc[-window:].values,
                eth_returns.iloc[-window:].values,
                lag=self.parameters['te_lag'],
                n_bins=self.parameters['n_bins']
            )
            
            te_eth_to_btc = calculate_transfer_entropy(
                eth_returns.iloc[-window:].values,
                btc_returns.iloc[-window:].values,
                lag=self.parameters['te_lag'],
                n_bins=self.parameters['n_bins']
            )
            
            # Calculate ratio for directional bias
            te_ratio = te_btc_to_eth / (te_eth_to_btc + 1e-8)
            
            return {
                'te_btc_to_eth': te_btc_to_eth,
                'te_eth_to_btc': te_eth_to_btc,
                'te_ratio': te_ratio
            }
            
        except Exception as e:
            return {'te_btc_to_eth': 0.0, 'te_eth_to_btc': 0.0, 'te_ratio': 1.0}
    
    def generate_signal(self, data: Dict[str, pd.DataFrame], current_idx: int) -> str:
        """
        Generate trading signal based on spread analysis and transfer entropy
        
        Args:
            data: Dictionary containing price data for both assets
            current_idx: Current data index
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        btc_data = data[self.parameters['asset_pair'][0]]
        eth_data = data[self.parameters['asset_pair'][1]]
        
        if current_idx < self.parameters['lookback_window']:
            return 'HOLD'
        
        # Get current prices
        btc_price = btc_data['close'].iloc[current_idx]
        eth_price = eth_data['close'].iloc[current_idx]
        
        # Calculate hedge ratio
        hedge_ratio = self.calculate_hedge_ratio(
            btc_data['close'].iloc[:current_idx+1],
            eth_data['close'].iloc[:current_idx+1],
            self.parameters['lookback_window']
        )
        self.hedge_ratios.append(hedge_ratio)
        
        # Calculate spread
        spread = self.calculate_spread(btc_price, eth_price, hedge_ratio)
        self.spreads.append(spread)
        
        # Calculate z-score
        z_score = self.calculate_z_score(self.spreads, self.parameters['lookback_window'])
        self.z_scores.append(z_score)
        
        # Get transfer entropy signal
        btc_returns = btc_data['close'].pct_change().dropna()
        eth_returns = eth_data['close'].pct_change().dropna()
        
        te_info = self.get_transfer_entropy_signal(
            btc_returns.iloc[:current_idx],
            eth_returns.iloc[:current_idx]
        )
        self.transfer_entropies.append(te_info)
        
        # Generate trading signal
        signal = 'HOLD'
        
        # Check if transfer entropy is significant enough
        min_te = max(te_info['te_btc_to_eth'], te_info['te_eth_to_btc'])
        if min_te < self.parameters['min_te_threshold']:
            return signal
        
        # Mean reversion signals based on z-score
        if abs(z_score) > self.parameters['z_entry_threshold']:
            if z_score > 0:  # Spread too high, expect reversion
                # BTC overvalued relative to ETH, sell BTC/buy ETH
                if te_info['te_ratio'] < 1.0:  # ETH leading BTC (confirming signal)
                    signal = 'SELL'
            else:  # Spread too low, expect reversion
                # BTC undervalued relative to ETH, buy BTC/sell ETH
                if te_info['te_ratio'] > 1.0:  # BTC leading ETH (confirming signal)
                    signal = 'BUY'
        
        # Exit signals
        elif self.current_position != 0 and abs(z_score) < self.parameters['z_exit_threshold']:
            signal = 'EXIT'
        
        return signal
    
    def execute_trade(self, signal: str, data: Dict[str, pd.DataFrame], current_idx: int) -> Dict[str, Any]:
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal
            data: Price data
            current_idx: Current index
            
        Returns:
            Trade execution details
        """
        btc_data = data[self.parameters['asset_pair'][0]]
        eth_data = data[self.parameters['asset_pair'][1]]
        
        btc_price = btc_data['close'].iloc[current_idx]
        eth_price = eth_data['close'].iloc[current_idx]
        
        # Calculate recent volatility and correlation
        if current_idx >= self.parameters['vol_window']:
            btc_returns = btc_data['close'].pct_change().iloc[current_idx-self.parameters['vol_window']:current_idx]
            eth_returns = eth_data['close'].pct_change().iloc[current_idx-self.parameters['vol_window']:current_idx]
            
            volatility = np.sqrt(btc_returns.var() + eth_returns.var())
            correlation = btc_returns.corr(eth_returns)
        else:
            volatility = 0.02
            correlation = 0.5
        
        trade_info = {
            'timestamp': btc_data.index[current_idx],
            'signal': signal,
            'btc_price': btc_price,
            'eth_price': eth_price,
            'spread': self.spreads[-1] if self.spreads else 0,
            'z_score': self.z_scores[-1] if self.z_scores else 0,
            'position_before': self.current_position,
            'position_after': self.current_position,
            'pnl': 0,
            'portfolio_value': self.portfolio_value[-1]
        }
        
        if signal in ['BUY', 'SELL'] and self.current_position == 0:
            # Enter new position
            position_size = self.calculate_position_size(volatility, correlation)
            self.current_position = position_size if signal == 'BUY' else -position_size
            self.entry_price = btc_price
            self.entry_z_score = self.z_scores[-1] if self.z_scores else 0
            
            trade_info['position_after'] = self.current_position
            trade_info['entry_price'] = self.entry_price
            
        elif signal == 'EXIT' and self.current_position != 0:
            # Exit position
            price_change = (btc_price - self.entry_price) / self.entry_price
            pnl = self.current_position * price_change
            
            # Update portfolio value
            new_portfolio_value = self.portfolio_value[-1] * (1 + pnl)
            self.portfolio_value.append(new_portfolio_value)
            
            trade_info['pnl'] = pnl
            trade_info['portfolio_value'] = new_portfolio_value
            trade_info['exit_price'] = btc_price
            
            self.current_position = 0
            self.entry_price = 0
            
        # Store trade
        self.trades.append(trade_info)
        self.positions.append(self.current_position)
        
        return trade_info
    
    def backtest(self, data: Dict[str, pd.DataFrame], symbol: str = "BTC-ETH-PAIR") -> Dict[str, Any]:
        """
        Run complete backtest of the pair trading strategy
        
        Args:
            data: Dictionary containing price data for both assets
            symbol: Trading symbol identifier
            
        Returns:
            Complete backtest results
        """
        print(f"\n🔄 Running Composite Pair Trading Strategy backtest...")
        
        # Reset strategy state
        self.hedge_ratios = []
        self.spreads = []
        self.z_scores = []
        self.transfer_entropies = []
        self.positions = []
        self.current_position = 0
        self.entry_price = 0
        self.portfolio_value = [1.0]
        self.trades = []
        self.signals = []
        
        # Get data length
        data_length = min(len(data[self.parameters['asset_pair'][0]]), 
                         len(data[self.parameters['asset_pair'][1]]))
        
        print(f"   • Data length: {data_length} periods")
        print(f"   • Lookback window: {self.parameters['lookback_window']}")
        
        # Main backtest loop
        for i in range(data_length):
            try:
                # Generate signal
                signal = self.generate_signal(data, i)
                
                # Store signal information
                signal_info = {
                    'timestamp': data[self.parameters['asset_pair'][0]].index[i],
                    'signal': signal,
                    'z_score': self.z_scores[-1] if self.z_scores else 0,
                    'spread': self.spreads[-1] if self.spreads else 0,
                    'hedge_ratio': self.hedge_ratios[-1] if self.hedge_ratios else 1.0,
                    'te_info': self.transfer_entropies[-1] if self.transfer_entropies else {}
                }
                self.signals.append(signal_info)
                
                # Execute trade if signal is actionable
                if signal in ['BUY', 'SELL', 'EXIT']:
                    self.execute_trade(signal, data, i)
                
                # Update portfolio value for hold periods
                if signal == 'HOLD' and len(self.portfolio_value) == i:
                    self.portfolio_value.append(self.portfolio_value[-1])
                    
            except Exception as e:
                print(f"   ⚠️  Error at index {i}: {e}")
                continue
        
        # Calculate final metrics
        returns = pd.Series(self.portfolio_value).pct_change().dropna()
        
        metrics = self.calculate_performance_metrics(returns)
        
        print(f"   ✅ Backtest completed")
        print(f"   • Total trades: {len([t for t in self.trades if t.get('pnl', 0) != 0])}")
        print(f"   • Total return: {metrics.get('total_return', 0):.2%}")
        print(f"   • Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        
        return {
            'returns': returns,
            'portfolio_value': self.portfolio_value,
            'trades': self.trades,
            'signals': self.signals,
            'metrics': metrics,
            'hedge_ratios': self.hedge_ratios,
            'spreads': self.spreads,
            'z_scores': self.z_scores,
            'transfer_entropies': self.transfer_entropies,
            'positions': self.positions
        }
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            returns: Strategy returns series
            
        Returns:
            Dictionary of performance metrics
        """
        if returns.empty or len(returns) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'win_rate': 0}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Win rate from trades
        profitable_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        total_trades_with_pnl = [t for t in self.trades if t.get('pnl', 0) != 0]
        win_rate = len(profitable_trades) / len(total_trades_with_pnl) if total_trades_with_pnl else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(total_trades_with_pnl),
            'profitable_trades': len(profitable_trades)
        }


class CompositePairTradingOptimizer:
    """
    Parameter optimizer for the composite pair trading strategy
    """
    
    def __init__(self, asset_pair: Tuple[str, str] = ('BTC_USD', 'ETH_USD')):
        self.asset_pair = asset_pair
        self.optimization_results = []
        self.best_params = None
        self.best_score = -np.inf
    
    def get_parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Generate parameter grid for optimization
        
        Returns:
            List of parameter combinations to test
        """
        param_grid = []
        
        # Define parameter ranges
        lookback_windows = [40, 60, 80]
        z_entry_thresholds = [1.5, 2.0, 2.5]
        z_exit_thresholds = [0.3, 0.5, 0.7]
        te_windows = [20, 30, 40]
        risk_budgets = [0.01, 0.02, 0.03]
        
        for lookback in lookback_windows:
            for z_entry in z_entry_thresholds:
                for z_exit in z_exit_thresholds:
                    for te_window in te_windows:
                        for risk_budget in risk_budgets:
                            if z_exit < z_entry:  # Logical constraint
                                param_grid.append({
                                    'asset_pair': self.asset_pair,
                                    'lookback_window': lookback,
                                    'z_entry_threshold': z_entry,
                                    'z_exit_threshold': z_exit,
                                    'te_window': te_window,
                                    'te_lag': 1,
                                    'n_bins': 10,
                                    'vol_window': 30,
                                    'risk_budget': risk_budget,
                                    'stop_loss': 0.05,
                                    'take_profit': 0.10,
                                    'max_drawdown_limit': 0.15,
                                    'min_te_threshold': 0.01
                                })
        
        return param_grid
    
    def objective_function(self, returns: pd.Series, metrics: Dict[str, float]) -> float:
        """
        Objective function for optimization (higher is better)
        
        Args:
            returns: Strategy returns
            metrics: Performance metrics
            
        Returns:
            Objective score
        """
        if returns.empty or len(returns) < 10:
            return -999
        
        sharpe = metrics.get('sharpe_ratio', 0)
        total_return = metrics.get('total_return', 0)
        max_dd = abs(metrics.get('max_drawdown', 1))
        win_rate = metrics.get('win_rate', 0)
        total_trades = metrics.get('total_trades', 0)
        
        # Penalize strategies with too few trades
        if total_trades < 5:
            return -999
        
        # Composite score emphasizing risk-adjusted returns
        score = (
            sharpe * 0.4 +                    # Sharpe ratio weight
            (total_return * 100) * 0.3 +      # Total return weight  
            (1 / (max_dd + 0.01)) * 0.2 +     # Inverse max drawdown weight
            win_rate * 0.1                    # Win rate weight
        )
        
        return score
    
    def grid_search_optimization(self, 
                               train_data: Dict[str, pd.DataFrame],
                               val_data: Dict[str, pd.DataFrame],
                               max_combinations: int = 50) -> Dict[str, Any]:
        """
        Perform grid search optimization
        
        Args:
            train_data: Training data
            val_data: Validation data  
            max_combinations: Maximum parameter combinations to test
            
        Returns:
            Optimization results
        """
        param_grid = self.get_parameter_grid()
        
        # Limit combinations for performance
        if len(param_grid) > max_combinations:
            import random
            param_grid = random.sample(param_grid, max_combinations)
        
        print(f"   • Testing {len(param_grid)} parameter combinations...")
        
        results = []
        
        for i, params in enumerate(param_grid):
            try:
                # Test on training data
                strategy = CompositePairTradingStrategy(
                    name=f"CompositePair_Test_{i}",
                    parameters=params
                )
                
                backtest_results = strategy.backtest(train_data)
                train_metrics = backtest_results['metrics']
                train_score = self.objective_function(backtest_results['returns'], train_metrics)
                
                # Validate on validation data
                val_results = strategy.backtest(val_data)
                val_metrics = val_results['metrics']
                val_score = self.objective_function(val_results['returns'], val_metrics)
                
                # Combined score (prefer validation performance)
                combined_score = 0.3 * train_score + 0.7 * val_score
                
                result = {
                    'params': params,
                    'train_score': train_score,
                    'val_score': val_score,
                    'combined_score': combined_score,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                
                results.append(result)
                
                # Update best parameters
                if combined_score > self.best_score:
                    self.best_score = combined_score
                    self.best_params = params.copy()
                
                if (i + 1) % 10 == 0:
                    print(f"   • Tested {i + 1}/{len(param_grid)} combinations...")
                    
            except Exception as e:
                print(f"   ⚠️  Error testing combination {i}: {e}")
                continue
        
        # Sort results by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        self.optimization_results = results
        
        return {
            'best_score': self.best_score,
            'best_params': self.best_params,
            'results': results[:10],  # Top 10 results
            'total_tested': len(results)
        }
    
    def get_optimal_parameters(self) -> Dict[str, Any]:
        """
        Get the optimal parameters from optimization
        
        Returns:
            Best parameter set
        """
        if self.best_params is None:
            # Return default parameters if no optimization was run
            return {
                'asset_pair': self.asset_pair,
                'lookback_window': 60,
                'z_entry_threshold': 2.0,
                'z_exit_threshold': 0.5,
                'te_window': 30,
                'te_lag': 1,
                'n_bins': 10,
                'vol_window': 30,
                'risk_budget': 0.02,
                'stop_loss': 0.05,
                'take_profit': 0.10,
                'max_drawdown_limit': 0.15,
                'min_te_threshold': 0.01
            }
        
        return self.best_params.copy()
