"""
Performance Analysis Module

This module provides comprehensive performance analysis capabilities
for trading strategies including attribution, decomposition, and reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..utils.metrics import calculate_metrics, PerformanceMetrics, RiskMetrics


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    """
    
    def __init__(self, strategy_results: Dict[str, any]):
        """
        Initialize performance analyzer.
        
        Args:
            strategy_results: Dictionary containing strategy backtest results
        """
        self.strategy_results = strategy_results
        self.returns = strategy_results.get('returns')
        self.equity_curve = strategy_results.get('portfolio_value')
        self.trades = strategy_results.get('trades', [])
        self.metrics = strategy_results.get('metrics', {})
        
    def generate_performance_report(self) -> Dict[str, any]:
        """
        Generate comprehensive performance report.
        
        Returns:
            Dictionary containing performance analysis
        """
        if self.returns is None:
            return {}
        
        # Calculate detailed metrics
        metrics_result = calculate_metrics(self.returns, self.trades)
        
        # Performance attribution
        attribution = self.calculate_performance_attribution()
        
        # Period analysis
        period_analysis = self.analyze_performance_by_period()
        
        # Risk-return analysis
        risk_return = self.analyze_risk_return_profile()
        
        return {
            'summary_metrics': metrics_result,
            'attribution': attribution,
            'period_analysis': period_analysis,
            'risk_return': risk_return,
            'drawdown_analysis': self.analyze_drawdowns(),
            'trade_analysis': self.analyze_trades()
        }
    
    def calculate_performance_attribution(self) -> Dict[str, float]:
        """
        Calculate performance attribution by different factors.
        
        Returns:
            Dictionary of attribution factors
        """
        if self.returns is None or len(self.returns) == 0:
            return {}
        
        # Time-based attribution
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        yearly_returns = self.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        
        # Volatility regimes
        volatility = self.returns.rolling(window=20).std()
        high_vol_periods = volatility > volatility.quantile(0.7)
        low_vol_periods = volatility < volatility.quantile(0.3)
        
        high_vol_returns = self.returns[high_vol_periods].mean() if high_vol_periods.any() else 0
        low_vol_returns = self.returns[low_vol_periods].mean() if low_vol_periods.any() else 0
        
        # Market regime attribution (simplified)
        positive_market_days = self.returns > 0
        negative_market_days = self.returns < 0
        
        bull_market_performance = self.returns[positive_market_days].mean() if positive_market_days.any() else 0
        bear_market_performance = self.returns[negative_market_days].mean() if negative_market_days.any() else 0
        
        return {
            'monthly_consistency': monthly_returns.std(),
            'yearly_consistency': yearly_returns.std(),
            'high_vol_performance': high_vol_returns,
            'low_vol_performance': low_vol_returns,
            'bull_market_performance': bull_market_performance,
            'bear_market_performance': bear_market_performance,
            'positive_days_ratio': positive_market_days.sum() / len(self.returns)
        }
    
    def analyze_performance_by_period(self) -> Dict[str, any]:
        """
        Analyze performance across different time periods.
        
        Returns:
            Dictionary of period-based analysis
        """
        if self.returns is None:
            return {}
        
        # Monthly analysis
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_stats = {
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'positive_months': (monthly_returns > 0).sum(),
            'total_months': len(monthly_returns),
            'monthly_win_rate': (monthly_returns > 0).mean()
        }
        
        # Quarterly analysis
        quarterly_returns = self.returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
        quarterly_stats = {
            'best_quarter': quarterly_returns.max(),
            'worst_quarter': quarterly_returns.min(),
            'positive_quarters': (quarterly_returns > 0).sum(),
            'total_quarters': len(quarterly_returns)
        }
        
        # Yearly analysis
        yearly_returns = self.returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
        yearly_stats = {
            'best_year': yearly_returns.max(),
            'worst_year': yearly_returns.min(),
            'positive_years': (yearly_returns > 0).sum(),
            'total_years': len(yearly_returns)
        }
        
        return {
            'monthly': monthly_stats,
            'quarterly': quarterly_stats,
            'yearly': yearly_stats,
            'raw_monthly_returns': monthly_returns,
            'raw_quarterly_returns': quarterly_returns,
            'raw_yearly_returns': yearly_returns
        }
    
    def analyze_risk_return_profile(self) -> Dict[str, any]:
        """
        Analyze the risk-return profile of the strategy.
        
        Returns:
            Risk-return analysis results
        """
        if self.returns is None:
            return {}
        
        # Rolling risk metrics
        rolling_vol = self.returns.rolling(window=60).std() * np.sqrt(252)
        rolling_sharpe = self.returns.rolling(window=60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        # Return distribution analysis
        return_quantiles = self.returns.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        
        # Tail risk analysis
        extreme_positive = self.returns[self.returns > self.returns.quantile(0.95)]
        extreme_negative = self.returns[self.returns < self.returns.quantile(0.05)]
        
        return {
            'rolling_volatility': {
                'mean': rolling_vol.mean(),
                'std': rolling_vol.std(),
                'min': rolling_vol.min(),
                'max': rolling_vol.max()
            },
            'rolling_sharpe': {
                'mean': rolling_sharpe.mean(),
                'std': rolling_sharpe.std(),
                'min': rolling_sharpe.min(),
                'max': rolling_sharpe.max()
            },
            'return_distribution': return_quantiles.to_dict(),
            'tail_analysis': {
                'extreme_positive_mean': extreme_positive.mean(),
                'extreme_negative_mean': extreme_negative.mean(),
                'extreme_positive_freq': len(extreme_positive) / len(self.returns),
                'extreme_negative_freq': len(extreme_negative) / len(self.returns)
            }
        }
    
    def analyze_drawdowns(self) -> Dict[str, any]:
        """
        Comprehensive drawdown analysis.
        
        Returns:
            Drawdown analysis results
        """
        if self.returns is None:
            return {}
        
        # Calculate drawdowns
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        peak_value = 0
        trough_value = 0
        
        for date, value in drawdown.items():
            if value < 0 and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                start_date = date
                peak_value = running_max[date]
                trough_value = cumulative_returns[date]
            elif value < 0 and in_drawdown:
                # Continue drawdown
                if cumulative_returns[date] < trough_value:
                    trough_value = cumulative_returns[date]
            elif value >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                duration = (date - start_date).days
                magnitude = (trough_value - peak_value) / peak_value
                
                drawdown_periods.append({
                    'start': start_date,
                    'end': date,
                    'duration': duration,
                    'magnitude': magnitude,
                    'peak_value': peak_value,
                    'trough_value': trough_value
                })
        
        # Analyze drawdown periods
        if drawdown_periods:
            durations = [dd['duration'] for dd in drawdown_periods]
            magnitudes = [dd['magnitude'] for dd in drawdown_periods]
            
            drawdown_stats = {
                'max_drawdown': min(magnitudes),
                'avg_drawdown': np.mean(magnitudes),
                'max_duration': max(durations),
                'avg_duration': np.mean(durations),
                'num_drawdowns': len(drawdown_periods),
                'recovery_factor': abs(self.returns.sum() / min(magnitudes)) if magnitudes else 0
            }
        else:
            drawdown_stats = {
                'max_drawdown': 0,
                'avg_drawdown': 0,
                'max_duration': 0,
                'avg_duration': 0,
                'num_drawdowns': 0,
                'recovery_factor': 0
            }
        
        return {
            'statistics': drawdown_stats,
            'periods': drawdown_periods,
            'drawdown_series': drawdown
        }
    
    def analyze_trades(self) -> Dict[str, any]:
        """
        Comprehensive trade analysis.
        
        Returns:
            Trade analysis results
        """
        if not self.trades:
            return {}
        
        # Basic trade statistics
        pnls = [trade.pnl for trade in self.trades]
        returns = [trade.return_pct for trade in self.trades]
        durations = [trade.duration.days for trade in self.trades if hasattr(trade, 'duration')]
        
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        # Trade distribution analysis
        trade_stats = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'largest_win': max(winning_trades) if winning_trades else 0,
            'largest_loss': min(losing_trades) if losing_trades else 0,
            'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf'),
            'avg_duration': np.mean(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0
        }
        
        # Monthly trade analysis
        if hasattr(self.trades[0], 'entry_time'):
            trade_months = {}
            for trade in self.trades:
                month = trade.entry_time.strftime('%Y-%m')
                if month not in trade_months:
                    trade_months[month] = []
                trade_months[month].append(trade.pnl)
            
            monthly_trade_stats = {
                month: {
                    'count': len(pnls),
                    'total_pnl': sum(pnls),
                    'win_rate': len([p for p in pnls if p > 0]) / len(pnls)
                }
                for month, pnls in trade_months.items()
            }
        else:
            monthly_trade_stats = {}
        
        return {
            'overall': trade_stats,
            'monthly': monthly_trade_stats,
            'pnl_distribution': {
                'percentiles': np.percentile(pnls, [10, 25, 50, 75, 90]) if pnls else [],
                'mean': np.mean(pnls) if pnls else 0,
                'std': np.std(pnls) if pnls else 0
            }
        }
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate a summary table of key metrics.
        
        Returns:
            DataFrame with summary metrics
        """
        if self.returns is None:
            return pd.DataFrame()
        
        metrics_result = calculate_metrics(self.returns, self.trades)
        perf_metrics = metrics_result['performance']
        risk_metrics = metrics_result['risk']
        
        summary_data = {
            'Total Return': f"{perf_metrics.total_return:.2%}",
            'Annualized Return': f"{perf_metrics.annualized_return:.2%}",
            'Volatility': f"{perf_metrics.volatility:.2%}",
            'Sharpe Ratio': f"{perf_metrics.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{perf_metrics.sortino_ratio:.2f}",
            'Max Drawdown': f"{perf_metrics.max_drawdown:.2%}",
            'Calmar Ratio': f"{perf_metrics.calmar_ratio:.2f}",
            'Win Rate': f"{perf_metrics.win_rate:.2%}",
            'Number of Trades': perf_metrics.num_trades,
            'VaR (95%)': f"{risk_metrics.var_95:.2%}",
            'CVaR (95%)': f"{risk_metrics.cvar_95:.2%}"
        }
        
        return pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
    
    def plot_performance_dashboard(self, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Create comprehensive performance dashboard.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib Figure object
        """
        if self.returns is None:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Strategy Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Equity curve
        if self.equity_curve is not None:
            equity_series = pd.Series(self.equity_curve, index=self.returns.index[:len(self.equity_curve)])
            axes[0, 0].plot(equity_series.index, equity_series.values)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value')
        
        # 2. Drawdown
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        axes[0, 1].fill_between(drawdown.index, 0, drawdown.values, alpha=0.3, color='red')
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown %')
        
        # 3. Monthly returns
        monthly_returns = self.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        colors = ['green' if x > 0 else 'red' for x in monthly_returns.values]
        axes[0, 2].bar(range(len(monthly_returns)), monthly_returns.values, color=colors)
        axes[0, 2].set_title('Monthly Returns')
        axes[0, 2].set_ylabel('Return %')
        
        # 4. Return distribution
        axes[1, 0].hist(self.returns.values, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(self.returns.mean(), color='red', linestyle='--', label='Mean')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].legend()
        
        # 5. Rolling Sharpe ratio
        rolling_sharpe = self.returns.rolling(window=60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
        axes[1, 1].set_title('Rolling Sharpe Ratio (60-day)')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        
        # 6. Trade PnL (if available)
        if self.trades:
            trade_pnls = [trade.pnl for trade in self.trades]
            trade_dates = [trade.entry_time for trade in self.trades if hasattr(trade, 'entry_time')]
            
            if len(trade_dates) == len(trade_pnls):
                colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
                axes[1, 2].scatter(trade_dates, trade_pnls, c=colors, alpha=0.6)
                axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 2].set_title('Trade PnL')
                axes[1, 2].set_ylabel('PnL')
        
        plt.tight_layout()
        return fig
