"""
Portfolio utility functions for portfolio management and rebalancing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import optimize
from dataclasses import dataclass
import warnings


@dataclass
class RebalanceResult:
    """Results from portfolio rebalancing."""
    new_weights: Dict[str, float]
    transactions: Dict[str, float]  # Amount to buy (+) or sell (-)
    turnover: float
    cost: float
    success: bool
    message: str


class PortfolioUtils:
    """
    Utility class for portfolio management operations.
    """
    
    def __init__(self, transaction_cost: float = 0.001):
        """
        Initialize portfolio utilities.
        
        Parameters:
        -----------
        transaction_cost : float
            Transaction cost as fraction of trade value
        """
        self.transaction_cost = transaction_cost
    
    def calculate_portfolio_value(self, 
                                weights: Dict[str, float],
                                prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Parameters:
        -----------
        weights : dict
            Asset weights (quantities)
        prices : dict
            Current asset prices
            
        Returns:
        --------
        float
            Total portfolio value
        """
        total_value = 0.0
        for asset, weight in weights.items():
            if asset in prices:
                total_value += weight * prices[asset]
        return total_value
    
    def calculate_asset_allocation(self,
                                 weights: Dict[str, float],
                                 prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate asset allocation percentages.
        
        Parameters:
        -----------
        weights : dict
            Asset weights (quantities)
        prices : dict
            Current asset prices
            
        Returns:
        --------
        dict
            Asset allocation percentages
        """
        portfolio_value = self.calculate_portfolio_value(weights, prices)
        if portfolio_value == 0:
            return {asset: 0.0 for asset in weights}
        
        allocation = {}
        for asset, weight in weights.items():
            if asset in prices:
                allocation[asset] = (weight * prices[asset]) / portfolio_value
            else:
                allocation[asset] = 0.0
        
        return allocation
    
    def calculate_turnover(self,
                          old_weights: Dict[str, float],
                          new_weights: Dict[str, float]) -> float:
        """
        Calculate portfolio turnover.
        
        Parameters:
        -----------
        old_weights : dict
            Previous portfolio weights
        new_weights : dict
            New portfolio weights
            
        Returns:
        --------
        float
            Portfolio turnover (0 to 1)
        """
        all_assets = set(old_weights.keys()) | set(new_weights.keys())
        total_change = 0.0
        
        for asset in all_assets:
            old_weight = old_weights.get(asset, 0.0)
            new_weight = new_weights.get(asset, 0.0)
            total_change += abs(new_weight - old_weight)
        
        return total_change / 2.0  # Divide by 2 since buys = sells
    
    def calculate_transaction_costs(self,
                                  transactions: Dict[str, float],
                                  prices: Dict[str, float]) -> float:
        """
        Calculate transaction costs.
        
        Parameters:
        -----------
        transactions : dict
            Transaction amounts (positive = buy, negative = sell)
        prices : dict
            Asset prices
            
        Returns:
        --------
        float
            Total transaction costs
        """
        total_cost = 0.0
        for asset, amount in transactions.items():
            if asset in prices and amount != 0:
                trade_value = abs(amount) * prices[asset]
                total_cost += trade_value * self.transaction_cost
        
        return total_cost


def rebalance_portfolio(current_weights: Dict[str, float],
                       target_weights: Dict[str, float],
                       prices: Dict[str, float],
                       portfolio_value: float,
                       transaction_cost: float = 0.001,
                       min_trade_size: float = 0.0) -> RebalanceResult:
    """
    Rebalance portfolio to target weights.
    
    Parameters:
    -----------
    current_weights : dict
        Current asset allocation percentages
    target_weights : dict
        Target asset allocation percentages
    prices : dict
        Current asset prices
    portfolio_value : float
        Total portfolio value
    transaction_cost : float
        Transaction cost as fraction of trade value
    min_trade_size : float
        Minimum trade size (ignore smaller trades)
        
    Returns:
    --------
    RebalanceResult
        Rebalancing results including new weights and costs
    """
    try:
        # Calculate target quantities
        target_quantities = {}
        for asset, target_pct in target_weights.items():
            if asset in prices and prices[asset] > 0:
                target_value = portfolio_value * target_pct
                target_quantities[asset] = target_value / prices[asset]
            else:
                target_quantities[asset] = 0.0
        
        # Calculate current quantities
        current_quantities = {}
        for asset in target_quantities:
            current_pct = current_weights.get(asset, 0.0)
            if asset in prices and prices[asset] > 0:
                current_value = portfolio_value * current_pct
                current_quantities[asset] = current_value / prices[asset]
            else:
                current_quantities[asset] = 0.0
        
        # Calculate transactions needed
        transactions = {}
        for asset in target_quantities:
            diff = target_quantities[asset] - current_quantities[asset]
            if abs(diff * prices.get(asset, 0)) >= min_trade_size:
                transactions[asset] = diff
            else:
                transactions[asset] = 0.0
        
        # Calculate costs
        utils = PortfolioUtils(transaction_cost)
        cost = utils.calculate_transaction_costs(transactions, prices)
        
        # Calculate new weights after transactions
        new_quantities = {}
        for asset in target_quantities:
            new_quantities[asset] = current_quantities[asset] + transactions[asset]
        
        # Convert back to percentages
        new_portfolio_value = portfolio_value - cost
        new_weights = {}
        for asset, quantity in new_quantities.items():
            if asset in prices:
                asset_value = quantity * prices[asset]
                new_weights[asset] = asset_value / new_portfolio_value if new_portfolio_value > 0 else 0.0
            else:
                new_weights[asset] = 0.0
        
        # Calculate turnover
        turnover = utils.calculate_turnover(current_weights, target_weights)
        
        return RebalanceResult(
            new_weights=new_weights,
            transactions=transactions,
            turnover=turnover,
            cost=cost,
            success=True,
            message="Rebalancing completed successfully"
        )
        
    except Exception as e:
        return RebalanceResult(
            new_weights=current_weights,
            transactions={},
            turnover=0.0,
            cost=0.0,
            success=False,
            message=f"Rebalancing failed: {str(e)}"
        )


def calculate_optimal_portfolio_size(returns: pd.DataFrame,
                                   risk_free_rate: float = 0.02,
                                   max_leverage: float = 1.0) -> Dict[str, float]:
    """
    Calculate optimal portfolio size using Kelly criterion.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    risk_free_rate : float
        Risk-free rate
    max_leverage : float
        Maximum leverage allowed
        
    Returns:
    --------
    dict
        Optimal portfolio weights
    """
    try:
        excess_returns = returns.mean() - risk_free_rate / 252
        cov_matrix = returns.cov()
        
        # Kelly weights: mu / (sigma^2)
        kelly_weights = np.linalg.solve(cov_matrix.values, excess_returns.values)
        
        # Scale to maximum leverage
        total_leverage = np.sum(np.abs(kelly_weights))
        if total_leverage > max_leverage:
            kelly_weights = kelly_weights * (max_leverage / total_leverage)
        
        return dict(zip(returns.columns, kelly_weights))
        
    except Exception as e:
        warnings.warn(f"Kelly calculation failed: {e}")
        # Return equal weights as fallback
        n_assets = len(returns.columns)
        equal_weight = max_leverage / n_assets
        return {col: equal_weight for col in returns.columns}


def risk_budget_portfolio(returns: pd.DataFrame,
                         risk_budgets: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Calculate risk budget portfolio weights.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
    risk_budgets : dict, optional
        Target risk budgets for each asset
        
    Returns:
    --------
    dict
        Portfolio weights
    """
    n_assets = len(returns.columns)
    
    if risk_budgets is None:
        # Equal risk budgets
        risk_budgets = {col: 1.0/n_assets for col in returns.columns}
    
    cov_matrix = returns.cov().values
    target_budgets = np.array([risk_budgets[col] for col in returns.columns])
    
    def risk_budget_objective(weights):
        """Objective function for risk budgeting."""
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        contrib = weights * marginal_contrib / portfolio_var
        return np.sum((contrib - target_budgets) ** 2)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Sum to 1
    ]
    
    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    
    # Initial guess
    x0 = np.array([1.0/n_assets] * n_assets)
    
    try:
        result = optimize.minimize(
            risk_budget_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            return dict(zip(returns.columns, weights))
        else:
            # Fallback to equal weights
            return {col: 1.0/n_assets for col in returns.columns}
            
    except Exception as e:
        warnings.warn(f"Risk budgeting optimization failed: {e}")
        return {col: 1.0/n_assets for col in returns.columns}


def calculate_diversification_ratio(weights: np.ndarray, 
                                   cov_matrix: np.ndarray) -> float:
    """
    Calculate portfolio diversification ratio.
    
    Parameters:
    -----------
    weights : np.ndarray
        Portfolio weights
    cov_matrix : np.ndarray
        Covariance matrix
        
    Returns:
    --------
    float
        Diversification ratio
    """
    # Portfolio volatility
    portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    
    # Weighted average volatility
    individual_vols = np.sqrt(np.diag(cov_matrix))
    weighted_avg_vol = np.dot(weights, individual_vols)
    
    return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 0.0


def maximum_diversification_portfolio(returns: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate maximum diversification portfolio.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Historical returns
        
    Returns:
    --------
    dict
        Portfolio weights
    """
    n_assets = len(returns.columns)
    cov_matrix = returns.cov().values
    
    def neg_diversification_ratio(weights):
        """Negative diversification ratio for minimization."""
        return -calculate_diversification_ratio(weights, cov_matrix)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Sum to 1
    ]
    
    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    
    # Initial guess
    x0 = np.array([1.0/n_assets] * n_assets)
    
    try:
        result = optimize.minimize(
            neg_diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            return dict(zip(returns.columns, weights))
        else:
            # Fallback to equal weights
            return {col: 1.0/n_assets for col in returns.columns}
            
    except Exception as e:
        warnings.warn(f"Maximum diversification optimization failed: {e}")
        return {col: 1.0/n_assets for col in returns.columns}
