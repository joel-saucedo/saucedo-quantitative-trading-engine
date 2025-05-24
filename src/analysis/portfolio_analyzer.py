"""
Portfolio Analysis Module

Comprehensive portfolio analysis including:
- Portfolio optimization
- Asset allocation analysis
- Rebalancing strategies
- Portfolio attribution
- Multi-asset portfolio construction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..utils.metrics import PerformanceMetrics, RiskMetrics
from .risk_analyzer import RiskAnalyzer


@dataclass
class PortfolioAllocation:
    """Portfolio allocation result"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    allocation_details: Dict[str, float]


@dataclass
class AttributionResult:
    """Portfolio attribution analysis result"""
    total_return: float
    asset_contribution: Dict[str, float]
    allocation_effect: Dict[str, float]
    selection_effect: Dict[str, float]
    interaction_effect: Dict[str, float]


class PortfolioAnalyzer:
    """
    Comprehensive portfolio analysis and optimization
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio analyzer
        
        Args:
            risk_free_rate: Risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.risk_analyzer = RiskAnalyzer()
        
    def calculate_portfolio_metrics(self, 
                                  returns: pd.DataFrame,
                                  weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics
        
        Args:
            returns: DataFrame with asset returns
            weights: Portfolio weights
            
        Returns:
            Portfolio metrics
        """
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Basic metrics
        total_return = (1 + portfolio_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        # Risk metrics
        max_drawdown = self.risk_analyzer._calculate_max_drawdown(portfolio_returns)
        var_95 = self.risk_analyzer.calculate_var(portfolio_returns, 0.95)
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95
        }
    
    def mean_variance_optimization(self, 
                                 returns: pd.DataFrame,
                                 method: str = 'sharpe',
                                 target_return: Optional[float] = None,
                                 constraints: Optional[Dict] = None) -> PortfolioAllocation:
        """
        Perform mean-variance optimization
        
        Args:
            returns: DataFrame with asset returns
            method: Optimization method ('sharpe', 'min_vol', 'target_return')
            target_return: Target return for constrained optimization
            constraints: Additional constraints
            
        Returns:
            PortfolioAllocation object
        """
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        n_assets = len(expected_returns)
        
        # Default constraints
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 1.0,
                'max_concentration': 0.4  # Max 40% in any single asset
            }
        
        # Constraint functions
        def constraint_sum_weights(weights):
            return np.sum(weights) - 1.0
        
        def constraint_min_weights(weights):
            return weights - constraints['min_weight']
        
        def constraint_max_weights(weights):
            return constraints['max_weight'] - weights
        
        def constraint_max_concentration(weights):
            return constraints['max_concentration'] - np.max(weights)
        
        # Set up constraints
        cons = [
            {'type': 'eq', 'fun': constraint_sum_weights},
            {'type': 'ineq', 'fun': constraint_min_weights},
            {'type': 'ineq', 'fun': constraint_max_weights}
        ]
        
        if 'max_concentration' in constraints:
            cons.append({'type': 'ineq', 'fun': constraint_max_concentration})
        
        # Objective functions
        def objective_sharpe(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        def objective_min_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        def objective_target_return(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_vol  # Minimize volatility subject to return constraint
        
        # Add target return constraint if needed
        if method == 'target_return' and target_return is not None:
            def constraint_target_return(weights):
                return np.sum(weights * expected_returns) - target_return
            cons.append({'type': 'eq', 'fun': constraint_target_return})
        
        # Choose objective function
        if method == 'sharpe':
            objective = objective_sharpe
        elif method == 'min_vol':
            objective = objective_min_vol
        elif method == 'target_return':
            objective = objective_target_return
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Bounds
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) 
                      for _ in range(n_assets))
        
        # Optimization
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
        
        optimal_weights = result.x
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Create allocation details
        allocation_details = {asset: weight for asset, weight in zip(returns.columns, optimal_weights)}
        
        return PortfolioAllocation(
            weights=optimal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            allocation_details=allocation_details
        )
    
    def efficient_frontier(self, 
                          returns: pd.DataFrame,
                          n_points: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Args:
            returns: DataFrame with asset returns
            n_points: Number of points on the frontier
            
        Returns:
            DataFrame with efficient frontier points
        """
        expected_returns = returns.mean() * 252
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier_results = []
        
        for target_ret in target_returns:
            try:
                allocation = self.mean_variance_optimization(
                    returns, 
                    method='target_return', 
                    target_return=target_ret
                )
                
                frontier_results.append({
                    'target_return': target_ret,
                    'volatility': allocation.volatility,
                    'sharpe_ratio': allocation.sharpe_ratio,
                    'weights': allocation.weights
                })
            except:
                continue
        
        return pd.DataFrame(frontier_results)
    
    def black_litterman_optimization(self, 
                                   returns: pd.DataFrame,
                                   market_caps: Optional[pd.Series] = None,
                                   views: Optional[Dict[str, float]] = None,
                                   view_confidence: float = 0.5) -> PortfolioAllocation:
        """
        Black-Litterman optimization
        
        Args:
            returns: DataFrame with asset returns
            market_caps: Market capitalizations for assets
            views: Dictionary of investor views {asset: expected_return}
            view_confidence: Confidence in views (0-1)
            
        Returns:
            PortfolioAllocation object
        """
        # Calculate sample statistics
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Market cap weights (if not provided, use equal weights)
        if market_caps is None:
            market_weights = np.array([1.0 / len(returns.columns)] * len(returns.columns))
        else:
            market_weights = market_caps / market_caps.sum()
            market_weights = market_weights.values
        
        # Risk aversion parameter (implied from market portfolio)
        market_return = np.sum(market_weights * expected_returns)
        market_variance = np.dot(market_weights.T, np.dot(cov_matrix, market_weights))
        risk_aversion = (market_return - self.risk_free_rate) / market_variance
        
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(cov_matrix, market_weights)
        
        if views is not None:
            # Incorporate views
            P = np.zeros((len(views), len(returns.columns)))
            Q = np.zeros(len(views))
            
            for i, (asset, view_return) in enumerate(views.items()):
                asset_idx = returns.columns.get_loc(asset)
                P[i, asset_idx] = 1.0
                Q[i] = view_return
            
            # View uncertainty matrix
            omega = view_confidence * np.dot(P, np.dot(cov_matrix, P.T))
            
            # Black-Litterman formula
            tau = 1.0  # Scaling factor
            M1 = np.linalg.inv(tau * cov_matrix)
            M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
            M3 = np.dot(np.linalg.inv(tau * cov_matrix), pi)
            M4 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
            
            bl_returns = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
            bl_cov = np.linalg.inv(M1 + M2)
        else:
            bl_returns = pi
            bl_cov = cov_matrix
        
        # Optimize with Black-Litterman inputs
        n_assets = len(bl_returns)
        
        def objective(weights):
            portfolio_return = np.sum(weights * bl_returns)
            portfolio_var = np.dot(weights.T, np.dot(bl_cov, weights))
            return -(portfolio_return - 0.5 * risk_aversion * portfolio_var)
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = market_weights
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        # Calculate metrics
        portfolio_return = np.sum(optimal_weights * bl_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(bl_cov, optimal_weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        allocation_details = {asset: weight for asset, weight in zip(returns.columns, optimal_weights)}
        
        return PortfolioAllocation(
            weights=optimal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            allocation_details=allocation_details
        )
    
    def risk_parity_optimization(self, returns: pd.DataFrame) -> PortfolioAllocation:
        """
        Risk parity optimization (equal risk contribution)
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            PortfolioAllocation object
        """
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)
        
        def risk_contribution(weights, cov_matrix):
            """Calculate risk contributions"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            return risk_contrib
        
        def objective(weights):
            """Minimize sum of squared deviations from equal risk contribution"""
            risk_contrib = risk_contribution(weights, cov_matrix)
            target_risk = 1.0 / n_assets
            return np.sum((risk_contrib - target_risk) ** 2)
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        bounds = tuple((0.001, 1) for _ in range(n_assets))  # Small minimum to avoid numerical issues
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        
        # Calculate metrics
        expected_returns = returns.mean() * 252
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        allocation_details = {asset: weight for asset, weight in zip(returns.columns, optimal_weights)}
        
        return PortfolioAllocation(
            weights=optimal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            allocation_details=allocation_details
        )
    
    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> PortfolioAllocation:
        """
        Hierarchical Risk Parity (HRP) optimization
        
        Args:
            returns: DataFrame with asset returns
            
        Returns:
            PortfolioAllocation object
        """
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Convert correlation to distance
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # Hierarchical clustering
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        def get_cluster_variance(cov_matrix, items):
            """Calculate cluster variance"""
            cluster_cov = cov_matrix.loc[items, items]
            inv_diag = 1 / np.diag(cluster_cov)
            w = inv_diag / inv_diag.sum()
            return np.dot(w, np.dot(cluster_cov, w))
        
        def get_quasi_diag(linkage_matrix):
            """Get quasi-diagonal matrix from linkage"""
            link = linkage_matrix.copy()
            quasi_diag = [link[-1, 0], link[-1, 1]]  # Start with top cluster
            num_items = link[-1, 3]
            
            while len(quasi_diag) < num_items:
                cluster_tms = []
                for i in quasi_diag:
                    if i < num_items:  # Original item
                        cluster_tms.append([i])
                    else:  # Cluster
                        cluster_idx = int(i - num_items)
                        cluster_tms.append([link[cluster_idx, 0], link[cluster_idx, 1]])
                
                new_quasi_diag = []
                for cluster in cluster_tms:
                    new_quasi_diag.extend(cluster)
                quasi_diag = new_quasi_diag
            
            return [int(i) for i in quasi_diag]
        
        # Get sorting order
        cov_matrix = returns.cov() * 252
        sort_idx = get_quasi_diag(linkage_matrix)
        sorted_assets = [returns.columns[i] for i in sort_idx]
        
        # Recursive bisection
        def get_rec_bipart(cov, sortIx):
            w = pd.Series(1.0, index=sortIx)
            cItems = [sortIx]  # Cluster items
            
            while len(cItems) > 0:
                cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
                
                for i in range(0, len(cItems), 2):
                    cItems0 = cItems[i]
                    cItems1 = cItems[i + 1]
                    cVar0 = get_cluster_variance(cov, cItems0)
                    cVar1 = get_cluster_variance(cov, cItems1)
                    alpha = 1 - cVar0 / (cVar0 + cVar1)
                    
                    w[cItems0] *= alpha
                    w[cItems1] *= 1 - alpha
            
            return w
        
        # Calculate HRP weights
        hrp_weights = get_rec_bipart(cov_matrix, sorted_assets)
        
        # Reorder to original asset order
        optimal_weights = np.array([hrp_weights[asset] for asset in returns.columns])
        
        # Calculate metrics
        expected_returns = returns.mean() * 252
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        allocation_details = {asset: weight for asset, weight in zip(returns.columns, optimal_weights)}
        
        return PortfolioAllocation(
            weights=optimal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            allocation_details=allocation_details
        )
    
    def portfolio_attribution(self, 
                            portfolio_returns: pd.DataFrame,
                            benchmark_returns: pd.DataFrame,
                            weights: np.ndarray) -> AttributionResult:
        """
        Perform portfolio attribution analysis
        
        Args:
            portfolio_returns: DataFrame with portfolio asset returns
            benchmark_returns: DataFrame with benchmark asset returns
            weights: Portfolio weights
            
        Returns:
            AttributionResult object
        """
        # Calculate total returns
        portfolio_total = (portfolio_returns * weights).sum(axis=1).sum()
        benchmark_total = benchmark_returns.mean(axis=1).sum()  # Equal-weighted benchmark
        
        # Asset-level analysis
        asset_contribution = {}
        allocation_effect = {}
        selection_effect = {}
        interaction_effect = {}
        
        # Benchmark weights (equal-weighted)
        benchmark_weights = np.array([1.0 / len(benchmark_returns.columns)] * len(benchmark_returns.columns))
        
        for i, asset in enumerate(portfolio_returns.columns):
            if asset in benchmark_returns.columns:
                # Portfolio metrics
                w_p = weights[i]
                r_p = portfolio_returns[asset].sum()
                
                # Benchmark metrics
                w_b = benchmark_weights[benchmark_returns.columns.get_loc(asset)]
                r_b = benchmark_returns[asset].sum()
                
                # Attribution effects
                asset_contribution[asset] = w_p * r_p
                allocation_effect[asset] = (w_p - w_b) * r_b
                selection_effect[asset] = w_b * (r_p - r_b)
                interaction_effect[asset] = (w_p - w_b) * (r_p - r_b)
        
        return AttributionResult(
            total_return=portfolio_total,
            asset_contribution=asset_contribution,
            allocation_effect=allocation_effect,
            selection_effect=selection_effect,
            interaction_effect=interaction_effect
        )
    
    def rebalancing_analysis(self, 
                           returns: pd.DataFrame,
                           target_weights: np.ndarray,
                           rebalance_frequency: str = 'monthly',
                           transaction_cost: float = 0.001) -> Dict[str, float]:
        """
        Analyze rebalancing strategy performance
        
        Args:
            returns: DataFrame with asset returns
            target_weights: Target portfolio weights
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'
            transaction_cost: Transaction cost per trade
            
        Returns:
            Rebalancing analysis results
        """
        # Determine rebalancing periods
        freq_map = {
            'daily': 1,
            'weekly': 5,
            'monthly': 21,
            'quarterly': 63
        }
        
        rebalance_freq = freq_map.get(rebalance_frequency, 21)
        
        # Initialize
        current_weights = target_weights.copy()
        portfolio_values = [1.0]
        transaction_costs = []
        
        for i in range(len(returns)):
            # Calculate period return
            period_returns = returns.iloc[i].values
            
            # Update portfolio value and weights
            portfolio_value = portfolio_values[-1] * (1 + np.sum(current_weights * period_returns))
            current_weights = current_weights * (1 + period_returns)
            current_weights = current_weights / current_weights.sum()  # Normalize
            
            portfolio_values.append(portfolio_value)
            
            # Check if rebalancing is needed
            if (i + 1) % rebalance_freq == 0:
                # Calculate rebalancing cost
                weight_changes = np.abs(current_weights - target_weights)
                cost = np.sum(weight_changes) * transaction_cost * portfolio_value
                transaction_costs.append(cost)
                
                # Rebalance
                current_weights = target_weights.copy()
                portfolio_values[-1] -= cost  # Subtract transaction cost
        
        # Calculate metrics
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = portfolio_values[-1] - 1.0
        total_costs = sum(transaction_costs)
        
        # Buy-and-hold comparison
        bah_weights = target_weights.copy()
        bah_value = 1.0
        for i in range(len(returns)):
            period_returns = returns.iloc[i].values
            bah_value *= (1 + np.sum(bah_weights * period_returns))
            bah_weights = bah_weights * (1 + period_returns)
            bah_weights = bah_weights / bah_weights.sum()
        
        bah_return = bah_value - 1.0
        
        return {
            'total_return': total_return,
            'buy_and_hold_return': bah_return,
            'rebalancing_benefit': total_return - bah_return,
            'total_transaction_costs': total_costs,
            'net_benefit': total_return - bah_return - total_costs,
            'number_of_rebalances': len(transaction_costs),
            'average_cost_per_rebalance': np.mean(transaction_costs) if transaction_costs else 0
        }
