"""
Risk Analysis Module

Comprehensive risk assessment for trading strategies including:
- Value at Risk (VaR) and Expected Shortfall (ES)
- Risk decomposition and attribution
- Tail risk analysis
- Stress testing and scenario analysis
- Risk-adjusted performance metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.optimize import minimize
import warnings
from dataclasses import dataclass

from ..utils.metrics import RiskMetrics


@dataclass
class RiskReport:
    """Risk analysis report"""
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    max_drawdown: float
    volatility: float
    downside_deviation: float
    tail_ratio: float
    risk_metrics: Dict[str, float]
    stress_test_results: Dict[str, float]


class RiskAnalyzer:
    """
    Comprehensive risk analysis for trading strategies
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize risk analyzer
        
        Args:
            confidence_levels: Confidence levels for VaR/ES calculations
        """
        self.confidence_levels = confidence_levels
        # Initialize RiskMetrics with default values
        self.risk_metrics = RiskMetrics(
            var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
            beta=0.0, alpha=0.0, information_ratio=0.0, tracking_error=0.0,
            downside_deviation=0.0, ulcer_index=0.0, pain_index=0.0,
            skewness=0.0, kurtosis=0.0, tail_ratio=0.0
        )
        
    def calculate_var(self, 
                     returns: Union[pd.Series, np.ndarray], 
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            method: 'historical', 'parametric', or 'monte_carlo'
            
        Returns:
            VaR value
        """
        returns = np.array(returns)
        
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            mu = np.mean(returns)
            sigma = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            return mu + z_score * sigma
        
        elif method == 'monte_carlo':
            # Monte Carlo simulation for VaR
            n_simulations = 10000
            mu = np.mean(returns)
            sigma = np.std(returns)
            
            simulated_returns = np.random.normal(mu, sigma, n_simulations)
            return np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_expected_shortfall(self, 
                                   returns: Union[pd.Series, np.ndarray],
                                   confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            returns: Return series
            confidence_level: Confidence level
            
        Returns:
            Expected Shortfall value
        """
        returns = np.array(returns)
        var = self.calculate_var(returns, confidence_level, 'historical')
        
        # ES is the mean of returns below VaR
        tail_returns = returns[returns <= var]
        
        if len(tail_returns) == 0:
            return var
        
        return np.mean(tail_returns)
    
    def calculate_component_var(self, 
                              portfolio_returns: pd.DataFrame,
                              weights: np.ndarray,
                              confidence_level: float = 0.95) -> pd.Series:
        """
        Calculate Component VaR for portfolio decomposition
        
        Args:
            portfolio_returns: DataFrame with asset returns
            weights: Portfolio weights
            confidence_level: Confidence level
            
        Returns:
            Component VaR for each asset
        """
        # Calculate portfolio returns
        portfolio_ret = (portfolio_returns * weights).sum(axis=1)
        portfolio_var = self.calculate_var(portfolio_ret, confidence_level)
        
        # Calculate marginal VaR using finite differences
        epsilon = 0.01
        component_vars = []
        
        for i in range(len(weights)):
            # Perturb weight slightly
            perturbed_weights = weights.copy()
            perturbed_weights[i] += epsilon
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize
            
            perturbed_returns = (portfolio_returns * perturbed_weights).sum(axis=1)
            perturbed_var = self.calculate_var(perturbed_returns, confidence_level)
            
            # Marginal VaR
            marginal_var = (perturbed_var - portfolio_var) / epsilon
            
            # Component VaR
            component_var = marginal_var * weights[i]
            component_vars.append(component_var)
        
        return pd.Series(component_vars, index=portfolio_returns.columns)
    
    def stress_test(self, 
                   returns: Union[pd.Series, np.ndarray],
                   scenarios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Perform stress testing under various scenarios
        
        Args:
            returns: Return series
            scenarios: Dictionary of stress scenarios
                      e.g., {'market_crash': {'mean_shock': -0.1, 'vol_shock': 2.0}}
            
        Returns:
            Stress test results
        """
        returns = np.array(returns)
        original_mean = np.mean(returns)
        original_std = np.std(returns)
        
        results = {}
        
        for scenario_name, shocks in scenarios.items():
            # Apply shocks
            shocked_mean = original_mean + shocks.get('mean_shock', 0)
            shocked_std = original_std * shocks.get('vol_shock', 1.0)
            
            # Generate stressed returns
            n_simulations = 1000
            stressed_returns = np.random.normal(shocked_mean, shocked_std, n_simulations)
            
            # Calculate metrics under stress
            stressed_var_95 = self.calculate_var(stressed_returns, 0.95)
            stressed_es_95 = self.calculate_expected_shortfall(stressed_returns, 0.95)
            
            results[scenario_name] = {
                'var_95': stressed_var_95,
                'es_95': stressed_es_95,
                'expected_return': shocked_mean,
                'volatility': shocked_std
            }
        
        return results
    
    def tail_risk_analysis(self, returns: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Analyze tail risk characteristics
        
        Args:
            returns: Return series
            
        Returns:
            Tail risk metrics
        """
        returns = np.array(returns)
        
        # Tail ratio (95th percentile / 5th percentile)
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        tail_ratio = p95 / abs(p5) if p5 != 0 else np.inf
        
        # Skewness and kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        
        # Tail index estimation (Hill estimator)
        sorted_returns = np.sort(returns)
        n = len(returns)
        k = max(int(n * 0.1), 10)  # Use top 10% for tail estimation
        
        tail_losses = -sorted_returns[:k]  # Negative returns (losses)
        if len(tail_losses) > 1 and np.min(tail_losses) > 0:
            log_ratios = np.log(tail_losses[:-1] / tail_losses[-1])
            hill_estimator = np.mean(log_ratios)
        else:
            hill_estimator = np.nan
        
        return {
            'tail_ratio': tail_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_stat': jb_stat,
            'jarque_bera_pvalue': jb_pvalue,
            'hill_estimator': hill_estimator,
            'is_normal': jb_pvalue > 0.05
        }
    
    def calculate_risk_adjusted_metrics(self, 
                                      returns: Union[pd.Series, np.ndarray],
                                      risk_free_rate: float = 0.0) -> Dict[str, float]:
        """
        Calculate comprehensive risk-adjusted performance metrics
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate
            
        Returns:
            Risk-adjusted metrics
        """
        returns = np.array(returns)
        excess_returns = returns - risk_free_rate
        
        # Basic metrics
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        # Corrected Sortino: use excess_returns.mean()
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0
        
        # Calmar ratio
        # max_dd = self.risk_metrics.calculate_max_drawdown(returns) # This was problematic
        max_dd = self._calculate_max_drawdown(returns) # Changed to helper method
        calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else 0
        
        # Information ratio (assuming benchmark return is 0)
        tracking_error = np.std(returns) * np.sqrt(252)
        information_ratio = annualized_return / tracking_error if tracking_error > 0 else 0
        
        # Treynor ratio (assuming beta = 1 for simplicity)
        treynor_ratio = annualized_return / 1.0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'max_drawdown': max_dd
        }
    
    def rolling_risk_analysis(self, 
                            returns: Union[pd.Series, np.ndarray],
                            window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling risk metrics
        
        Args:
            returns: Return series
            window: Rolling window size
            
        Returns:
            DataFrame with rolling risk metrics
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        
        results = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns.iloc[i-window:i]
            
            var_95 = self.calculate_var(window_returns, 0.95)
            es_95 = self.calculate_expected_shortfall(window_returns, 0.95)
            vol = np.std(window_returns) * np.sqrt(252)
            # max_dd = self.risk_metrics.calculate_max_drawdown(window_returns.values) # This was problematic
            max_dd = self._calculate_max_drawdown(window_returns.values) # Changed to helper method
            
            tail_metrics = self.tail_risk_analysis(window_returns)
            
            results.append({
                'date': returns.index[i-1] if hasattr(returns, 'index') else i-1,
                'var_95': var_95,
                'es_95': es_95,
                'volatility': vol,
                'max_drawdown': max_dd,
                'skewness': tail_metrics['skewness'],
                'kurtosis': tail_metrics['kurtosis']
            })
        
        return pd.DataFrame(results)
    
    def generate_risk_report(self, 
                           returns: Union[pd.Series, np.ndarray],
                           risk_free_rate: float = 0.0) -> RiskReport:
        """
        Generate comprehensive risk report
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate
            
        Returns:
            RiskReport object
        """
        returns = np.array(returns)
        
        # Basic VaR and ES
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)
        es_95 = self.calculate_expected_shortfall(returns, 0.95)
        es_99 = self.calculate_expected_shortfall(returns, 0.99)
        
        # Risk metrics
        # max_drawdown = self.risk_metrics.calculate_max_drawdown(returns) # This was problematic
        max_drawdown = self._calculate_max_drawdown(returns) # Changed to helper method
        volatility = np.std(returns) * np.sqrt(252)
        # downside_deviation = self.risk_metrics.calculate_downside_deviation(returns) # This was problematic
        downside_deviation = self._calculate_downside_deviation(returns) # Changed to helper method
        
        # Tail analysis
        tail_metrics = self.tail_risk_analysis(returns)
        tail_ratio = tail_metrics['tail_ratio']
        
        # Risk-adjusted metrics
        risk_adj_metrics = self.calculate_risk_adjusted_metrics(returns, risk_free_rate)
        
        # Stress testing
        stress_scenarios = {
            'market_crash': {'mean_shock': -0.1, 'vol_shock': 2.0},
            'high_volatility': {'mean_shock': 0.0, 'vol_shock': 1.5},
            'deflation': {'mean_shock': -0.05, 'vol_shock': 1.2}
        }
        stress_results = self.stress_test(returns, stress_scenarios)
        
        return RiskReport(
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            max_drawdown=max_drawdown,
            volatility=volatility,
            downside_deviation=downside_deviation,
            tail_ratio=tail_ratio,
            risk_metrics=risk_adj_metrics,
            stress_test_results=stress_results
        )
    
    def compare_risk_profiles(self, 
                            returns_dict: Dict[str, Union[pd.Series, np.ndarray]],
                            risk_free_rate: float = 0.0) -> pd.DataFrame:
        """
        Compare risk profiles of multiple strategies
        
        Args:
            returns_dict: Dictionary of strategy returns
            risk_free_rate: Risk-free rate
            
        Returns:
            DataFrame comparing risk metrics
        """
        results = []
        
        for strategy_name, returns in returns_dict.items():
            report = self.generate_risk_report(returns, risk_free_rate)
            
            result = {
                'strategy': strategy_name,
                'var_95': report.var_95,
                'var_99': report.var_99,
                'es_95': report.es_95,
                'es_99': report.es_99,
                'max_drawdown': report.max_drawdown,
                'volatility': report.volatility,
                'downside_deviation': report.downside_deviation,
                'tail_ratio': report.tail_ratio,
                **report.risk_metrics
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    # Helper methods for calculations that were previously (incorrectly) on self.risk_metrics
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate max drawdown from a numpy array of returns."""
        if len(returns) == 0:
            return 0.0
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min() if len(drawdown) > 0 else 0.0

    def _calculate_downside_deviation(self, returns: np.ndarray, risk_free_rate_annual: float = 0.0) -> float:
        """Calculate downside deviation from a numpy array of returns."""
        if len(returns) == 0:
            return 0.0
        periods_per_year = 252 # Assuming daily data
        risk_free_rate_period = risk_free_rate_annual / periods_per_year
        downside_returns = returns[returns < risk_free_rate_period]
        if len(downside_returns) == 0:
            return 0.0
        return np.std(downside_returns) * np.sqrt(periods_per_year)
