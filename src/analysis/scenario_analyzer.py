"""
Scenario Analysis Module

Comprehensive scenario analysis including:
- Historical scenario analysis
- Monte Carlo simulations
- Stress testing
- What-if analysis
- Factor model simulations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy import stats
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.metrics import PerformanceMetrics, RiskMetrics
from .risk_analyzer import RiskAnalyzer


@dataclass
class ScenarioResult:
    """Scenario analysis result"""
    scenario_name: str
    probability: float
    portfolio_return: float
    portfolio_volatility: float
    max_drawdown: float
    var_95: float
    expected_shortfall: float
    scenario_details: Dict[str, float]


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""
    mean_return: float
    std_return: float
    percentiles: Dict[str, float]
    probability_of_loss: float
    expected_shortfall: float
    paths: np.ndarray
    final_values: np.ndarray


class ScenarioAnalyzer:
    """
    Comprehensive scenario analysis for trading strategies and portfolios
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize scenario analyzer
        
        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.performance_metrics = PerformanceMetrics()
        self.risk_metrics = RiskMetrics()
        self.risk_analyzer = RiskAnalyzer()
        
    def historical_scenario_analysis(self, 
                                   returns: pd.DataFrame,
                                   portfolio_weights: np.ndarray,
                                   scenario_periods: Dict[str, Tuple[str, str]]) -> List[ScenarioResult]:
        """
        Analyze portfolio performance during historical scenarios
        
        Args:
            returns: DataFrame with asset returns
            portfolio_weights: Portfolio weights
            scenario_periods: Dictionary mapping scenario names to (start_date, end_date) tuples
            
        Returns:
            List of ScenarioResult objects
        """
        results = []
        
        for scenario_name, (start_date, end_date) in scenario_periods.items():
            # Filter returns for scenario period
            if isinstance(returns.index, pd.DatetimeIndex):
                scenario_returns = returns.loc[start_date:end_date]
            else:
                # If index is not datetime, assume it's the scenario period indices
                start_idx = int(start_date) if isinstance(start_date, str) and start_date.isdigit() else 0
                end_idx = int(end_date) if isinstance(end_date, str) and end_date.isdigit() else len(returns)
                scenario_returns = returns.iloc[start_idx:end_idx]
            
            if len(scenario_returns) == 0:
                continue
            
            # Calculate portfolio returns for scenario
            portfolio_returns = (scenario_returns * portfolio_weights).sum(axis=1)
            
            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            max_dd = self.risk_metrics.calculate_max_drawdown(portfolio_returns.values)
            var_95 = self.risk_analyzer.calculate_var(portfolio_returns, 0.95)
            es_95 = self.risk_analyzer.calculate_expected_shortfall(portfolio_returns, 0.95)
            
            # Additional scenario details
            scenario_details = {
                'duration_days': len(portfolio_returns),
                'positive_days': (portfolio_returns > 0).sum(),
                'negative_days': (portfolio_returns < 0).sum(),
                'worst_day': portfolio_returns.min(),
                'best_day': portfolio_returns.max(),
                'sharpe_ratio': (portfolio_returns.mean() - self.risk_free_rate/252) / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
            }
            
            results.append(ScenarioResult(
                scenario_name=scenario_name,
                probability=1.0,  # Historical scenarios have occurred
                portfolio_return=total_return,
                portfolio_volatility=volatility,
                max_drawdown=max_dd,
                var_95=var_95,
                expected_shortfall=es_95,
                scenario_details=scenario_details
            ))
        
        return results
    
    def monte_carlo_simulation(self, 
                             returns: pd.DataFrame,
                             portfolio_weights: np.ndarray,
                             n_simulations: int = 10000,
                             time_horizon: int = 252,
                             method: str = 'parametric') -> MonteCarloResult:
        """
        Perform Monte Carlo simulation of portfolio returns
        
        Args:
            returns: Historical returns DataFrame
            portfolio_weights: Portfolio weights
            n_simulations: Number of simulation paths
            time_horizon: Time horizon in days
            method: 'parametric', 'bootstrap', or 'factor_model'
            
        Returns:
            MonteCarloResult object
        """
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        
        if method == 'parametric':
            # Parametric simulation using normal distribution
            mu = portfolio_returns.mean()
            sigma = portfolio_returns.std()
            
            # Generate random paths
            random_returns = np.random.normal(mu, sigma, (n_simulations, time_horizon))
            
        elif method == 'bootstrap':
            # Bootstrap simulation from historical returns
            random_returns = np.random.choice(
                portfolio_returns.values, 
                size=(n_simulations, time_horizon), 
                replace=True
            )
            
        elif method == 'factor_model':
            # Factor model simulation (simplified single factor)
            # Estimate factor loadings
            market_returns = returns.mean(axis=1)  # Simple market proxy
            
            # Regression to get beta
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(market_returns, portfolio_returns)
            
            # Generate factor returns
            market_mu = market_returns.mean()
            market_sigma = market_returns.std()
            
            simulated_market = np.random.normal(market_mu, market_sigma, (n_simulations, time_horizon))
            
            # Generate idiosyncratic returns
            residuals = portfolio_returns - (intercept + slope * market_returns)
            idiosyncratic_sigma = residuals.std()
            
            simulated_idiosyncratic = np.random.normal(0, idiosyncratic_sigma, (n_simulations, time_horizon))
            
            # Combine
            random_returns = intercept + slope * simulated_market + simulated_idiosyncratic
            
        else:
            raise ValueError(f"Unknown simulation method: {method}")
        
        # Calculate cumulative returns for each path
        cumulative_returns = np.cumprod(1 + random_returns, axis=1)
        final_values = cumulative_returns[:, -1]
        
        # Calculate statistics
        final_returns = final_values - 1
        mean_return = np.mean(final_returns)
        std_return = np.std(final_returns)
        
        # Calculate percentiles
        percentiles = {
            'p5': np.percentile(final_returns, 5),
            'p10': np.percentile(final_returns, 10),
            'p25': np.percentile(final_returns, 25),
            'p50': np.percentile(final_returns, 50),
            'p75': np.percentile(final_returns, 75),
            'p90': np.percentile(final_returns, 90),
            'p95': np.percentile(final_returns, 95)
        }
        
        # Probability of loss
        prob_loss = (final_returns < 0).mean()
        
        # Expected shortfall (5% tail)
        tail_returns = final_returns[final_returns <= percentiles['p5']]
        expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else percentiles['p5']
        
        return MonteCarloResult(
            mean_return=mean_return,
            std_return=std_return,
            percentiles=percentiles,
            probability_of_loss=prob_loss,
            expected_shortfall=expected_shortfall,
            paths=cumulative_returns,
            final_values=final_values
        )
    
    def stress_testing(self, 
                      returns: pd.DataFrame,
                      portfolio_weights: np.ndarray,
                      stress_scenarios: Dict[str, Dict[str, float]]) -> List[ScenarioResult]:
        """
        Perform stress testing under various scenarios
        
        Args:
            returns: Historical returns DataFrame
            portfolio_weights: Portfolio weights
            stress_scenarios: Dictionary of stress scenarios
                            e.g., {'market_crash': {'shock_magnitude': -0.2, 'shock_correlation': 0.8}}
            
        Returns:
            List of ScenarioResult objects
        """
        results = []
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Extract scenario parameters
            shock_magnitude = scenario_params.get('shock_magnitude', -0.1)
            shock_correlation = scenario_params.get('shock_correlation', 0.5)
            shock_duration = scenario_params.get('shock_duration', 21)  # Days
            recovery_factor = scenario_params.get('recovery_factor', 0.5)
            
            # Generate stressed returns
            n_simulations = 1000
            stressed_paths = []
            
            for _ in range(n_simulations):
                # Start with historical return characteristics
                base_returns = np.random.choice(portfolio_returns.values, size=shock_duration, replace=True)
                
                # Apply stress
                stress_factor = np.random.uniform(shock_correlation, 1.0)
                stressed_returns = base_returns * stress_factor + shock_magnitude / shock_duration
                
                # Add recovery
                if recovery_factor > 0:
                    recovery_returns = np.random.choice(portfolio_returns.values, size=shock_duration, replace=True)
                    recovery_returns = recovery_returns * recovery_factor
                    stressed_returns = np.concatenate([stressed_returns, recovery_returns])
                
                stressed_paths.append(stressed_returns)
            
            # Calculate metrics across simulations
            path_returns = []
            path_volatilities = []
            path_drawdowns = []
            
            for path in stressed_paths:
                total_return = (1 + path).prod() - 1
                volatility = np.std(path) * np.sqrt(252)
                max_dd = self.risk_metrics.calculate_max_drawdown(path)
                
                path_returns.append(total_return)
                path_volatilities.append(volatility)
                path_drawdowns.append(max_dd)
            
            # Aggregate statistics
            avg_return = np.mean(path_returns)
            avg_volatility = np.mean(path_volatilities)
            avg_max_dd = np.mean(path_drawdowns)
            
            # Calculate VaR and ES from path returns
            var_95 = np.percentile(path_returns, 5)
            es_95 = np.mean([r for r in path_returns if r <= var_95])
            
            # Scenario probability (subjective estimate)
            probability = scenario_params.get('probability', 0.05)
            
            scenario_details = {
                'shock_magnitude': shock_magnitude,
                'shock_duration': shock_duration,
                'worst_case_return': np.min(path_returns),
                'best_case_return': np.max(path_returns),
                'recovery_probability': np.mean([r > 0 for r in path_returns])
            }
            
            results.append(ScenarioResult(
                scenario_name=scenario_name,
                probability=probability,
                portfolio_return=avg_return,
                portfolio_volatility=avg_volatility,
                max_drawdown=avg_max_dd,
                var_95=var_95,
                expected_shortfall=es_95,
                scenario_details=scenario_details
            ))
        
        return results
    
    def what_if_analysis(self, 
                        returns: pd.DataFrame,
                        base_weights: np.ndarray,
                        weight_changes: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Perform what-if analysis with different portfolio allocations
        
        Args:
            returns: Historical returns DataFrame
            base_weights: Base portfolio weights
            weight_changes: Dictionary of weight change scenarios
                          e.g., {'increase_equity': {'stocks': 0.1, 'bonds': -0.1}}
            
        Returns:
            Dictionary of scenario results
        """
        results = {}
        
        # Calculate base portfolio metrics
        base_portfolio_returns = (returns * base_weights).sum(axis=1)
        base_metrics = self._calculate_portfolio_metrics(base_portfolio_returns)
        results['base_case'] = base_metrics
        
        for scenario_name, changes in weight_changes.items():
            # Apply weight changes
            new_weights = base_weights.copy()
            
            for asset, change in changes.items():
                if asset in returns.columns:
                    asset_idx = returns.columns.get_loc(asset)
                    new_weights[asset_idx] += change
            
            # Ensure weights sum to 1
            new_weights = new_weights / new_weights.sum()
            
            # Calculate new portfolio metrics
            new_portfolio_returns = (returns * new_weights).sum(axis=1)
            new_metrics = self._calculate_portfolio_metrics(new_portfolio_returns)
            
            # Calculate differences from base case
            differences = {f"{key}_diff": new_metrics[key] - base_metrics[key] 
                          for key in new_metrics.keys() if key in base_metrics}
            
            new_metrics.update(differences)
            results[scenario_name] = new_metrics
        
        return results
    
    def factor_scenario_analysis(self, 
                               returns: pd.DataFrame,
                               portfolio_weights: np.ndarray,
                               factor_scenarios: Dict[str, Dict[str, float]]) -> List[ScenarioResult]:
        """
        Analyze portfolio performance under different factor scenarios
        
        Args:
            returns: Historical returns DataFrame
            portfolio_weights: Portfolio weights
            factor_scenarios: Dictionary of factor scenarios
                            e.g., {'high_inflation': {'interest_rate_factor': 0.02, 'equity_factor': -0.1}}
            
        Returns:
            List of ScenarioResult objects
        """
        # Simplified factor model (can be extended with actual factor loadings)
        portfolio_returns = (returns * portfolio_weights).sum(axis=1)
        
        # Estimate factor sensitivities (simplified)
        # In practice, you would use proper factor models (Fama-French, etc.)
        market_factor = returns.mean(axis=1)  # Market factor proxy
        
        results = []
        
        for scenario_name, factor_shocks in factor_scenarios.items():
            # Apply factor shocks to generate scenario returns
            shocked_returns = portfolio_returns.copy()
            
            # Apply market factor shock
            market_shock = factor_shocks.get('market_factor', 0)
            shocked_returns += market_shock
            
            # Apply volatility shock
            vol_shock = factor_shocks.get('volatility_factor', 1.0)
            shocked_returns = shocked_returns * vol_shock
            
            # Add regime change effects
            regime_shock = factor_shocks.get('regime_factor', 0)
            if regime_shock != 0:
                # Apply different shock to different periods
                n_periods = len(shocked_returns)
                regime_periods = np.random.choice([0, 1], size=n_periods, p=[0.7, 0.3])
                shocked_returns[regime_periods == 1] += regime_shock
            
            # Calculate scenario metrics
            total_return = (1 + shocked_returns).prod() - 1
            volatility = shocked_returns.std() * np.sqrt(252)
            max_dd = self.risk_metrics.calculate_max_drawdown(shocked_returns.values)
            var_95 = self.risk_analyzer.calculate_var(shocked_returns, 0.95)
            es_95 = self.risk_analyzer.calculate_expected_shortfall(shocked_returns, 0.95)
            
            probability = factor_scenarios.get('probability', 0.1)
            
            scenario_details = {
                'factor_shocks': factor_shocks,
                'correlation_with_base': np.corrcoef(portfolio_returns, shocked_returns)[0, 1]
            }
            
            results.append(ScenarioResult(
                scenario_name=scenario_name,
                probability=probability,
                portfolio_return=total_return,
                portfolio_volatility=volatility,
                max_drawdown=max_dd,
                var_95=var_95,
                expected_shortfall=es_95,
                scenario_details=scenario_details
            ))
        
        return results
    
    def _calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Helper method to calculate portfolio metrics"""
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_drawdown = self.risk_metrics.calculate_max_drawdown(returns.values)
        var_95 = self.risk_analyzer.calculate_var(returns, 0.95)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95
        }
    
    def scenario_comparison_report(self, 
                                 scenario_results: List[ScenarioResult]) -> pd.DataFrame:
        """
        Generate comparison report of scenario results
        
        Args:
            scenario_results: List of ScenarioResult objects
            
        Returns:
            DataFrame with scenario comparison
        """
        data = []
        
        for result in scenario_results:
            data.append({
                'Scenario': result.scenario_name,
                'Probability': result.probability,
                'Portfolio Return': result.portfolio_return,
                'Volatility': result.portfolio_volatility,
                'Max Drawdown': result.max_drawdown,
                'VaR 95%': result.var_95,
                'Expected Shortfall': result.expected_shortfall,
                'Risk-Adjusted Return': result.portfolio_return / result.portfolio_volatility if result.portfolio_volatility > 0 else 0
            })
        
        df = pd.DataFrame(data)
        
        # Sort by probability (most likely scenarios first)
        df = df.sort_values('Probability', ascending=False)
        
        return df
    
    def plot_scenario_analysis(self, 
                             scenario_results: List[ScenarioResult],
                             save_path: Optional[str] = None) -> None:
        """
        Plot scenario analysis results
        
        Args:
            scenario_results: List of ScenarioResult objects
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        scenarios = [r.scenario_name for r in scenario_results]
        returns = [r.portfolio_return for r in scenario_results]
        volatilities = [r.portfolio_volatility for r in scenario_results]
        max_drawdowns = [r.max_drawdown for r in scenario_results]
        probabilities = [r.probability for r in scenario_results]
        
        # Return vs Volatility scatter
        scatter = axes[0, 0].scatter(volatilities, returns, s=[p*1000 for p in probabilities], 
                                   c=returns, cmap='RdYlGn', alpha=0.7)
        axes[0, 0].set_xlabel('Volatility')
        axes[0, 0].set_ylabel('Portfolio Return')
        axes[0, 0].set_title('Return vs Volatility by Scenario')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Add scenario labels
        for i, scenario in enumerate(scenarios):
            axes[0, 0].annotate(scenario, (volatilities[i], returns[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Max Drawdown comparison
        axes[0, 1].bar(range(len(scenarios)), max_drawdowns, color='red', alpha=0.7)
        axes[0, 1].set_xlabel('Scenario')
        axes[0, 1].set_ylabel('Max Drawdown')
        axes[0, 1].set_title('Maximum Drawdown by Scenario')
        axes[0, 1].set_xticks(range(len(scenarios)))
        axes[0, 1].set_xticklabels(scenarios, rotation=45, ha='right')
        
        # Probability vs Return
        axes[1, 0].scatter(probabilities, returns, s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].set_ylabel('Portfolio Return')
        axes[1, 0].set_title('Return vs Probability')
        
        for i, scenario in enumerate(scenarios):
            axes[1, 0].annotate(scenario, (probabilities[i], returns[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Risk-Return efficiency
        risk_adj_returns = [r/v if v > 0 else 0 for r, v in zip(returns, volatilities)]
        axes[1, 1].bar(range(len(scenarios)), risk_adj_returns, alpha=0.7)
        axes[1, 1].set_xlabel('Scenario')
        axes[1, 1].set_ylabel('Risk-Adjusted Return')
        axes[1, 1].set_title('Risk-Adjusted Returns by Scenario')
        axes[1, 1].set_xticks(range(len(scenarios)))
        axes[1, 1].set_xticklabels(scenarios, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_monte_carlo_results(self, 
                               mc_result: MonteCarloResult,
                               save_path: Optional[str] = None) -> None:
        """
        Plot Monte Carlo simulation results
        
        Args:
            mc_result: MonteCarloResult object
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Path evolution
        n_paths_to_plot = min(100, mc_result.paths.shape[0])
        sample_paths = mc_result.paths[:n_paths_to_plot]
        
        for i in range(n_paths_to_plot):
            axes[0, 0].plot(sample_paths[i], alpha=0.1, color='blue')
        
        # Plot mean path
        mean_path = np.mean(mc_result.paths, axis=0)
        axes[0, 0].plot(mean_path, color='red', linewidth=2, label='Mean Path')
        axes[0, 0].set_xlabel('Time (Days)')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].set_title('Monte Carlo Simulation Paths')
        axes[0, 0].legend()
        
        # Final value distribution
        axes[0, 1].hist(mc_result.final_values, bins=50, alpha=0.7, density=True)
        axes[0, 1].axvline(np.mean(mc_result.final_values), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(mc_result.final_values):.3f}')
        axes[0, 1].axvline(mc_result.percentiles['p5'], color='orange', linestyle='--',
                         label=f'5th percentile: {mc_result.percentiles["p5"]:.3f}')
        axes[0, 1].set_xlabel('Final Portfolio Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Distribution of Final Values')
        axes[0, 1].legend()
        
        # Percentile chart
        percentile_names = list(mc_result.percentiles.keys())
        percentile_values = list(mc_result.percentiles.values())
        
        axes[1, 0].bar(percentile_names, percentile_values, alpha=0.7)
        axes[1, 0].set_xlabel('Percentile')
        axes[1, 0].set_ylabel('Return')
        axes[1, 0].set_title('Return Percentiles')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Risk metrics
        risk_metrics = {
            'Mean Return': mc_result.mean_return,
            'Std Return': mc_result.std_return,
            'Prob of Loss': mc_result.probability_of_loss,
            'Expected Shortfall': mc_result.expected_shortfall
        }
        
        metric_names = list(risk_metrics.keys())
        metric_values = list(risk_metrics.values())
        
        bars = axes[1, 1].bar(metric_names, metric_values, alpha=0.7)
        axes[1, 1].set_xlabel('Metric')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Risk Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
