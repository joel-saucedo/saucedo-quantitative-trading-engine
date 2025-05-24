"""
Risk Metrics Module

Advanced risk analysis including tail risk measures, regime analysis,
drawdown statistics, and stress testing scenarios.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import warnings


class RiskMetrics:
    """
    Comprehensive risk metrics calculator for bootstrap analysis.
    
    Implements advanced risk measures including:
    - Tail risk metrics (VaR, CVaR, Expected Shortfall)
    - Drawdown analysis
    - Regime-based risk measures
    - Stress testing scenarios
    """
    
    def __init__(self, confidence_levels: List[float] = None):
        """
        Initialize risk metrics calculator.
        
        Args:
            confidence_levels: Confidence levels for VaR/CVaR calculations
        """
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99, 0.999]
    
    def value_at_risk(self, returns: np.ndarray, 
                     confidence_levels: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate Value at Risk at multiple confidence levels.
        
        Args:
            returns: Array of returns
            confidence_levels: Confidence levels (if None, uses instance default)
            
        Returns:
            VaR values at different confidence levels
        """
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
            
        var_results = {}
        
        for conf in confidence_levels:
            alpha = 1 - conf
            var_value = np.percentile(returns, alpha * 100)
            var_results[f'VaR_{int(conf*100)}'] = var_value
            
        return var_results
    
    def conditional_value_at_risk(self, returns: np.ndarray,
                                 confidence_levels: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).
        
        Args:
            returns: Array of returns
            confidence_levels: Confidence levels
            
        Returns:
            CVaR values at different confidence levels
        """
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
            
        cvar_results = {}
        
        for conf in confidence_levels:
            alpha = 1 - conf
            var_threshold = np.percentile(returns, alpha * 100)
            tail_returns = returns[returns <= var_threshold]
            
            if len(tail_returns) > 0:
                cvar_value = np.mean(tail_returns)
            else:
                cvar_value = var_threshold
                
            cvar_results[f'CVaR_{int(conf*100)}'] = cvar_value
            
        return cvar_results
    
    def expected_drawdown(self, equity_curve: np.ndarray,
                         confidence_levels: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Calculate Expected Drawdown at various confidence levels.
        
        Args:
            equity_curve: Equity curve array
            confidence_levels: Confidence levels
            
        Returns:
            Expected drawdown values
        """
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
            
        # Calculate drawdown series
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        
        # Remove zero drawdowns for tail analysis
        non_zero_dd = drawdowns[drawdowns < 0]
        
        ed_results = {}
        
        for conf in confidence_levels:
            if len(non_zero_dd) > 0:
                alpha = 1 - conf
                dd_threshold = np.percentile(non_zero_dd, alpha * 100)
                tail_drawdowns = non_zero_dd[non_zero_dd <= dd_threshold]
                ed_value = np.mean(tail_drawdowns) if len(tail_drawdowns) > 0 else dd_threshold
            else:
                ed_value = 0.0
                
            ed_results[f'ExpectedDD_{int(conf*100)}'] = ed_value
            
        return ed_results
    
    def ulcer_index(self, equity_curve: np.ndarray) -> float:
        """
        Calculate Ulcer Index (RMS of drawdowns).
        
        Args:
            equity_curve: Equity curve array
            
        Returns:
            Ulcer Index value
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        return np.sqrt(np.mean(drawdowns ** 2))
    
    def pain_index(self, equity_curve: np.ndarray) -> float:
        """
        Calculate Pain Index (mean of drawdowns).
        
        Args:
            equity_curve: Equity curve array
            
        Returns:
            Pain Index value
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        return np.mean(np.abs(drawdowns))
    
    def burke_ratio(self, returns: np.ndarray, equity_curve: np.ndarray, 
                   risk_free_rate: float = 0.0) -> float:
        """
        Calculate Burke Ratio.
        
        Args:
            returns: Return array
            equity_curve: Equity curve array
            risk_free_rate: Risk-free rate
            
        Returns:
            Burke Ratio value
        """
        excess_returns = returns - risk_free_rate
        mean_excess = np.mean(excess_returns)
        
        # Calculate drawdowns
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        
        # Burke ratio denominator
        burke_denominator = np.sqrt(np.sum(drawdowns ** 2))
        
        if burke_denominator > 0:
            return mean_excess / burke_denominator
        else:
            return np.inf if mean_excess > 0 else np.nan
    
    def sterling_ratio(self, returns: np.ndarray, equity_curve: np.ndarray,
                      period_years: float = 1.0) -> float:
        """
        Calculate Sterling Ratio.
        
        Args:
            returns: Return array
            equity_curve: Equity curve array
            period_years: Period in years
            
        Returns:
            Sterling Ratio value
        """
        total_return = equity_curve[-1] / equity_curve[0] - 1
        annualized_return = (1 + total_return) ** (1 / period_years) - 1
        
        # Average drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        avg_drawdown = np.mean(np.abs(drawdowns))
        
        if avg_drawdown > 0:
            return annualized_return / avg_drawdown
        else:
            return np.inf if annualized_return > 0 else np.nan
    
    def omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio.
        
        Args:
            returns: Return array
            threshold: Threshold return level
            
        Returns:
            Omega Ratio value
        """
        excess_returns = returns - threshold
        
        gains = np.sum(np.maximum(excess_returns, 0))
        losses = np.sum(np.abs(np.minimum(excess_returns, 0)))
        
        if losses > 0:
            return gains / losses
        else:
            return np.inf if gains > 0 else 1.0
    
    def gain_to_pain_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Gain-to-Pain Ratio.
        
        Args:
            returns: Return array
            
        Returns:
            Gain-to-Pain Ratio value
        """
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            gain = np.sum(positive_returns)
            pain = np.sum(np.abs(negative_returns))
            return gain / pain if pain > 0 else np.inf
        else:
            return np.nan
    
    def tail_ratio(self, returns: np.ndarray, percentile: float = 95) -> float:
        """
        Calculate Tail Ratio (ratio of right tail to left tail).
        
        Args:
            returns: Return array
            percentile: Percentile for tail calculation
            
        Returns:
            Tail Ratio value
        """
        upper_tail = np.percentile(returns, percentile)
        lower_tail = np.percentile(returns, 100 - percentile)
        
        if lower_tail < 0:
            return upper_tail / np.abs(lower_tail)
        else:
            return np.inf
    
    def common_sense_ratio(self, returns: np.ndarray, equity_curve: np.ndarray) -> float:
        """
        Calculate Common Sense Ratio.
        
        Args:
            returns: Return array
            equity_curve: Equity curve array
            
        Returns:
            Common Sense Ratio value
        """
        total_return = equity_curve[-1] / equity_curve[0] - 1
        
        # Calculate max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Tail ratio component
        tail_ratio = self.tail_ratio(returns)
        
        if max_drawdown < 0 and tail_ratio > 0:
            return total_return * tail_ratio / np.abs(max_drawdown)
        else:
            return np.nan
    
    def drawdown_duration_analysis(self, equity_curve: np.ndarray) -> Dict[str, Any]:
        """
        Analyze drawdown durations and recovery times.
        
        Args:
            equity_curve: Equity curve array
            
        Returns:
            Drawdown duration statistics
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        
        # Find drawdown periods
        in_drawdown = drawdowns < -1e-6  # Use small threshold to avoid numerical issues
        
        # Calculate duration statistics
        durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
                
        # Add final duration if still in drawdown
        if current_duration > 0:
            durations.append(current_duration)
        
        if len(durations) > 0:
            return {
                'num_drawdown_periods': len(durations),
                'avg_drawdown_duration': np.mean(durations),
                'max_drawdown_duration': np.max(durations),
                'median_drawdown_duration': np.median(durations),
                'total_drawdown_periods': np.sum(durations),
                'pct_time_in_drawdown': np.sum(durations) / len(equity_curve) * 100
            }
        else:
            return {
                'num_drawdown_periods': 0,
                'avg_drawdown_duration': 0,
                'max_drawdown_duration': 0,
                'median_drawdown_duration': 0,
                'total_drawdown_periods': 0,
                'pct_time_in_drawdown': 0
            }
    
    def regime_risk_analysis(self, returns: np.ndarray, 
                           regime_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Analyze risk metrics across different volatility regimes.
        
        Args:
            returns: Return array
            regime_threshold: Threshold for high volatility regime
            
        Returns:
            Regime-based risk analysis
        """
        # Calculate rolling volatility (21-day window)
        window = min(21, len(returns) // 4)
        if window < 3:
            return {'error': 'Insufficient data for regime analysis'}
            
        rolling_vol = pd.Series(returns).rolling(window).std()
        
        # Define regimes
        high_vol_regime = rolling_vol > regime_threshold
        low_vol_regime = rolling_vol <= regime_threshold
        
        # Remove NaN values
        valid_idx = ~rolling_vol.isna()
        returns_valid = returns[valid_idx]
        high_vol_valid = high_vol_regime[valid_idx]
        low_vol_valid = low_vol_regime[valid_idx]
        
        results = {}
        
        # High volatility regime
        high_vol_returns = returns_valid[high_vol_valid]
        if len(high_vol_returns) > 0:
            results['high_vol_regime'] = {
                'count': len(high_vol_returns),
                'mean_return': np.mean(high_vol_returns),
                'volatility': np.std(high_vol_returns),
                'skewness': stats.skew(high_vol_returns),
                'kurtosis': stats.kurtosis(high_vol_returns),
                'var_95': np.percentile(high_vol_returns, 5),
                'cvar_95': np.mean(high_vol_returns[high_vol_returns <= np.percentile(high_vol_returns, 5)])
            }
        
        # Low volatility regime
        low_vol_returns = returns_valid[low_vol_valid]
        if len(low_vol_returns) > 0:
            results['low_vol_regime'] = {
                'count': len(low_vol_returns),
                'mean_return': np.mean(low_vol_returns),
                'volatility': np.std(low_vol_returns),
                'skewness': stats.skew(low_vol_returns),
                'kurtosis': stats.kurtosis(low_vol_returns),
                'var_95': np.percentile(low_vol_returns, 5),
                'cvar_95': np.mean(low_vol_returns[low_vol_returns <= np.percentile(low_vol_returns, 5)])
            }
        
        return results
    
    def stress_test_scenarios(self, returns: np.ndarray, 
                            scenarios: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Apply stress test scenarios to returns.
        
        Args:
            returns: Return array
            scenarios: Dictionary of stress scenarios
            
        Returns:
            Stress test results
        """
        if scenarios is None:
            scenarios = {
                'market_crash': {'shock': -0.20, 'probability': 0.01},
                'volatility_spike': {'vol_multiplier': 3.0, 'duration': 10},
                'extended_drawdown': {'negative_drift': -0.001, 'duration': 60},
                'black_swan': {'shock': -0.40, 'probability': 0.001}
            }
        
        results = {}
        
        for scenario_name, params in scenarios.items():
            stressed_returns = returns.copy()
            
            if scenario_name == 'market_crash':
                # Apply single large negative shock
                shock_prob = params.get('probability', 0.01)
                shock_magnitude = params.get('shock', -0.20)
                
                n_shocks = int(len(returns) * shock_prob)
                if n_shocks > 0:
                    shock_indices = np.random.choice(len(returns), n_shocks, replace=False)
                    stressed_returns[shock_indices] += shock_magnitude
                    
            elif scenario_name == 'volatility_spike':
                # Increase volatility for certain periods
                vol_mult = params.get('vol_multiplier', 3.0)
                duration = params.get('duration', 10)
                
                # Apply to random periods
                n_periods = len(returns) // duration
                for _ in range(max(1, n_periods // 10)):  # 10% of periods affected
                    start_idx = np.random.randint(0, len(returns) - duration)
                    end_idx = start_idx + duration
                    period_returns = stressed_returns[start_idx:end_idx]
                    period_mean = np.mean(period_returns)
                    stressed_returns[start_idx:end_idx] = (period_returns - period_mean) * vol_mult + period_mean
                    
            elif scenario_name == 'extended_drawdown':
                # Apply negative drift for extended period
                negative_drift = params.get('negative_drift', -0.001)
                duration = params.get('duration', 60)
                
                start_idx = np.random.randint(0, max(1, len(returns) - duration))
                end_idx = min(start_idx + duration, len(returns))
                stressed_returns[start_idx:end_idx] += negative_drift
                
            elif scenario_name == 'black_swan':
                # Extreme negative event
                shock_prob = params.get('probability', 0.001)
                shock_magnitude = params.get('shock', -0.40)
                
                if np.random.random() < shock_prob * len(returns):
                    shock_idx = np.random.randint(0, len(returns))
                    stressed_returns[shock_idx] += shock_magnitude
            
            # Calculate stressed metrics
            stressed_equity = np.cumprod(1 + stressed_returns)
            
            results[scenario_name] = {
                'total_return': stressed_equity[-1] - 1,
                'volatility': np.std(stressed_returns),
                'max_drawdown': self._calculate_max_drawdown_from_returns(stressed_returns),
                'var_95': np.percentile(stressed_returns, 5),
                'cvar_95': np.mean(stressed_returns[stressed_returns <= np.percentile(stressed_returns, 5)]),
                'ulcer_index': self.ulcer_index(stressed_equity)
            }
        
        return results
    
    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """Helper method to calculate max drawdown from returns."""
        equity = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        return np.min(drawdowns)
    
    def calculate_all_metrics(self, bootstrap_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for bootstrap results.
        
        Args:
            bootstrap_results: Results from bootstrap simulation
            
        Returns:
            Complete risk metrics
        """
        simulated_stats = bootstrap_results['simulated_stats']
        
        # Extract return arrays for analysis
        if 'simulated_equity_curves' in bootstrap_results and bootstrap_results['simulated_equity_curves'] is not None:
            equity_curves = bootstrap_results['simulated_equity_curves'].values
        else:
            # Generate from statistics if equity curves not available
            equity_curves = None
        
        # Aggregate metrics across simulations
        all_returns = []
        all_equity = []
        
        # This is a simplified approach - in practice we'd need the actual return series
        # from each simulation to calculate proper risk metrics
        
        # For now, calculate based on available statistics
        results = {
            'var_metrics': {},
            'cvar_metrics': {},
            'tail_risk': {},
            'drawdown_analysis': {},
            'regime_analysis': {},
            'stress_tests': {}
        }
        
        # Extract metrics from simulated statistics
        if simulated_stats:
            # VaR metrics from simulated statistics
            for metric in ['CumulativeReturn', 'MaxDrawdown', 'Sharpe']:
                if metric in simulated_stats[0]:
                    values = [stat[metric] for stat in simulated_stats if not np.isnan(stat.get(metric, np.nan))]
                    if values:
                        results['var_metrics'][metric] = self.value_at_risk(np.array(values))
                        results['cvar_metrics'][metric] = self.conditional_value_at_risk(np.array(values))
            
            # Tail risk analysis
            sharpe_values = [stat.get('Sharpe', np.nan) for stat in simulated_stats]
            sharpe_values = [x for x in sharpe_values if not np.isnan(x)]
            
            if sharpe_values:
                results['tail_risk'] = {
                    'sharpe_tail_ratio': self.tail_ratio(np.array(sharpe_values)),
                    'sharpe_omega_ratio': self.omega_ratio(np.array(sharpe_values))
                }
        
        return results
