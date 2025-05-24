"""
Performance and Risk Metrics Calculation

This module provides comprehensive metrics for evaluating trading strategy
performance including returns, risk, and risk-adjusted metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Union, List
from scipy import stats
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    num_trades: int
    hit_ratio: float


@dataclass 
class RiskMetrics:
    """Container for risk metrics."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    beta: float
    alpha: float
    information_ratio: float
    tracking_error: float
    downside_deviation: float
    ulcer_index: float
    pain_index: float
    skewness: float
    kurtosis: float
    tail_ratio: float


def calculate_basic_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Calculate basic performance metrics.
    
    Args:
        returns: Strategy returns series
        benchmark_returns: Benchmark returns for comparison
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        Dictionary of basic metrics
    """
    if len(returns) == 0:
        return {}
    
    # Remove any NaN values
    returns = returns.dropna()
    
    # Basic statistics
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    periods_per_year = 252  # Assuming daily data
    
    # Annualized metrics
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    volatility = returns.std() * np.sqrt(periods_per_year)
    
    # Risk-adjusted metrics
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
    
    # Downside deviation and Sortino ratio
    negative_returns = returns[returns < 0]
    downside_deviation = negative_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio = excess_returns.mean() / negative_returns.std() * np.sqrt(periods_per_year) if len(negative_returns) > 0 and negative_returns.std() > 0 else 0
    
    # Drawdown analysis
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Drawdown duration
    drawdown_duration = 0
    current_dd_duration = 0
    max_dd_duration = 0
    
    for dd in drawdown:
        if dd < 0:
            current_dd_duration += 1
            max_dd_duration = max(max_dd_duration, current_dd_duration)
        else:
            current_dd_duration = 0
    
    # Calmar ratio
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_duration': max_dd_duration,
        'downside_deviation': downside_deviation
    }
    
    # Benchmark-relative metrics
    if benchmark_returns is not None:
        benchmark_returns = benchmark_returns.dropna()
        if len(benchmark_returns) > 0:
            # Align returns
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 1:
                # Beta and Alpha
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                benchmark_annualized = (1 + aligned_benchmark).prod() ** (periods_per_year / len(aligned_benchmark)) - 1
                alpha = annualized_return - (risk_free_rate + beta * (benchmark_annualized - risk_free_rate))
                
                # Tracking error and information ratio
                active_returns = aligned_returns - aligned_benchmark
                tracking_error = active_returns.std() * np.sqrt(periods_per_year)
                information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(periods_per_year) if active_returns.std() > 0 else 0
                
                metrics.update({
                    'beta': beta,
                    'alpha': alpha,
                    'tracking_error': tracking_error,
                    'information_ratio': information_ratio
                })
    
    return metrics


def calculate_trade_metrics(trades: List) -> Dict[str, float]:
    """
    Calculate trade-level metrics.
    
    Args:
        trades: List of Trade objects
        
    Returns:
        Dictionary of trade metrics
    """
    if not trades:
        return {}
    
    # Extract trade data
    pnls = [trade.pnl for trade in trades]
    returns = [trade.return_pct for trade in trades]
    
    winning_trades = [pnl for pnl in pnls if pnl > 0]
    losing_trades = [pnl for pnl in pnls if pnl < 0]
    
    # Basic trade statistics
    num_trades = len(trades)
    num_winning = len(winning_trades)
    num_losing = len(losing_trades)
    win_rate = num_winning / num_trades if num_trades > 0 else 0
    
    # Profit metrics
    total_profit = sum(winning_trades) if winning_trades else 0
    total_loss = abs(sum(losing_trades)) if losing_trades else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    # Hit ratio (avg win / avg loss)
    hit_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Trade duration analysis
    durations = [trade.duration.days for trade in trades if hasattr(trade, 'duration')]
    avg_duration = np.mean(durations) if durations else 0
    
    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'hit_ratio': hit_ratio,
        'avg_duration': avg_duration,
        'total_pnl': sum(pnls)
    }


def calculate_risk_metrics(
    returns: pd.Series,
    confidence_levels: List[float] = [0.95, 0.99]
) -> Dict[str, float]:
    """
    Calculate comprehensive risk metrics.
    
    Args:
        returns: Returns series
        confidence_levels: VaR confidence levels
        
    Returns:
        Dictionary of risk metrics
    """
    if len(returns) == 0:
        return {}
    
    returns = returns.dropna()
    metrics = {}
    
    # Value at Risk (VaR)
    for conf in confidence_levels:
        var_value = np.percentile(returns, (1 - conf) * 100)
        cvar_value = returns[returns <= var_value].mean()
        
        metrics[f'var_{int(conf*100)}'] = var_value
        metrics[f'cvar_{int(conf*100)}'] = cvar_value
    
    # Distribution metrics
    metrics['skewness'] = stats.skew(returns)
    metrics['kurtosis'] = stats.kurtosis(returns)
    
    # Tail ratio
    left_tail = np.percentile(returns, 5)
    right_tail = np.percentile(returns, 95)
    metrics['tail_ratio'] = abs(right_tail / left_tail) if left_tail != 0 else 0
    
    # Ulcer Index
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - running_max) / running_max
    ulcer_index = np.sqrt((drawdown ** 2).mean())
    metrics['ulcer_index'] = ulcer_index
    
    # Pain Index
    pain_index = drawdown.mean()
    metrics['pain_index'] = pain_index
    
    return metrics


def calculate_metrics(
    returns: pd.Series,
    trades: Optional[List] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, Union[PerformanceMetrics, RiskMetrics, Dict]]:
    """
    Calculate comprehensive performance and risk metrics.
    
    Args:
        returns: Strategy returns
        trades: List of trades
        benchmark_returns: Benchmark returns
        risk_free_rate: Risk-free rate
        
    Returns:
        Dictionary containing all metrics
    """
    # Basic performance metrics
    basic_metrics = calculate_basic_metrics(returns, benchmark_returns, risk_free_rate)
    
    # Trade metrics
    trade_metrics = calculate_trade_metrics(trades) if trades else {}
    
    # Risk metrics
    risk_metrics = calculate_risk_metrics(returns)
    
    # Combine all metrics
    all_metrics = {**basic_metrics, **trade_metrics, **risk_metrics}
    
    # Create structured objects
    perf_metrics = PerformanceMetrics(
        total_return=all_metrics.get('total_return', 0),
        annualized_return=all_metrics.get('annualized_return', 0),
        volatility=all_metrics.get('volatility', 0),
        sharpe_ratio=all_metrics.get('sharpe_ratio', 0),
        sortino_ratio=all_metrics.get('sortino_ratio', 0),
        calmar_ratio=all_metrics.get('calmar_ratio', 0),
        max_drawdown=all_metrics.get('max_drawdown', 0),
        max_drawdown_duration=all_metrics.get('max_drawdown_duration', 0),
        win_rate=all_metrics.get('win_rate', 0),
        profit_factor=all_metrics.get('profit_factor', 0),
        avg_win=all_metrics.get('avg_win', 0),
        avg_loss=all_metrics.get('avg_loss', 0),
        num_trades=all_metrics.get('num_trades', 0),
        hit_ratio=all_metrics.get('hit_ratio', 0)
    )
    
    risk_metrics_obj = RiskMetrics(
        var_95=all_metrics.get('var_95', 0),
        var_99=all_metrics.get('var_99', 0),
        cvar_95=all_metrics.get('cvar_95', 0),
        cvar_99=all_metrics.get('cvar_99', 0),
        beta=all_metrics.get('beta', 0),
        alpha=all_metrics.get('alpha', 0),
        information_ratio=all_metrics.get('information_ratio', 0),
        tracking_error=all_metrics.get('tracking_error', 0),
        downside_deviation=all_metrics.get('downside_deviation', 0),
        ulcer_index=all_metrics.get('ulcer_index', 0),
        pain_index=all_metrics.get('pain_index', 0),
        skewness=all_metrics.get('skewness', 0),
        kurtosis=all_metrics.get('kurtosis', 0),
        tail_ratio=all_metrics.get('tail_ratio', 0)
    )
    
    return {
        'performance': perf_metrics,
        'risk': risk_metrics_obj,
        'raw_metrics': all_metrics
    }


def compare_strategies(
    strategy_results: Dict[str, pd.Series],
    benchmark_returns: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.
    
    Args:
        strategy_results: Dictionary of strategy name -> returns
        benchmark_returns: Benchmark for comparison
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for strategy_name, returns in strategy_results.items():
        metrics = calculate_basic_metrics(returns, benchmark_returns)
        metrics['strategy'] = strategy_name
        comparison_data.append(metrics)
    
    df = pd.DataFrame(comparison_data)
    df.set_index('strategy', inplace=True)
    
    # Sort by Sharpe ratio
    df = df.sort_values('sharpe_ratio', ascending=False)
    
    return df


def rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    metrics: List[str] = ['sharpe_ratio', 'volatility', 'max_drawdown']
) -> pd.DataFrame:
    """
    Calculate rolling metrics over time.
    
    Args:
        returns: Returns series
        window: Rolling window size
        metrics: List of metrics to calculate
        
    Returns:
        DataFrame with rolling metrics
    """
    rolling_data = {}
    
    for metric in metrics:
        if metric == 'sharpe_ratio':
            rolling_data[metric] = returns.rolling(window).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
        elif metric == 'volatility':
            rolling_data[metric] = returns.rolling(window).std() * np.sqrt(252)
        elif metric == 'max_drawdown':
            def rolling_max_dd(x):
                cum_ret = (1 + x).cumprod()
                running_max = cum_ret.expanding().max()
                dd = (cum_ret - running_max) / running_max
                return dd.min()
            
            rolling_data[metric] = returns.rolling(window).apply(rolling_max_dd)
    
    return pd.DataFrame(rolling_data, index=returns.index)
