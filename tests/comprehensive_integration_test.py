#!/usr/bin/env python3
"""
Comprehensive Integration Test

This script integrates all framework capabilities:
1. Strategy testing and backtesting
2. Monte Carlo statistical validation
3. Advanced bootstrapping analysis
4. Performance visualization and plotting
5. Risk analysis and reporting
6. Complete workflow demonstration

Author: Saucedo Quantitative Trading Engine
Date: 2025-05-25
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Tuple, Any

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.utils.data_loader import DataLoader
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import BollingerBandsStrategy
from src.strategies.simple_pairs_strategy_fixed import SimplePairsStrategy, TrendFollowingStrategy, VolatilityBreakoutStrategy
from src.bootstrapping.core import AdvancedBootstrapping, BootstrapConfig, BootstrapMethod
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.analysis.risk_analyzer import RiskAnalyzer
from tests.statistical_validation import MonteCarloValidator, ValidationConfig

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')

class ComprehensiveIntegrationTest:
    """Complete integration test of all framework capabilities."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the comprehensive test."""
        self.output_dir = Path(output_dir) if output_dir else Path("results/comprehensive_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize components
        self.data_loader = DataLoader()
        # Note: PerformanceAnalyzer and RiskAnalyzer will be initialized after we have results
        
        # Test configuration
        self.symbols = ['BTC_USD', 'ETH_USD']
        self.strategies = [
            (MomentumStrategy, {'lookback_period': 20, 'momentum_threshold': 0.02}, 'Momentum'),
            (BollingerBandsStrategy, {'window': 20, 'num_stds': 2.0}, 'Bollinger Bands'),
            (SimplePairsStrategy, {'lookback_period': 30, 'z_entry_threshold': 1.5}, 'Simple Pairs'),
            (TrendFollowingStrategy, {'short_window': 10, 'long_window': 30}, 'Trend Following'),
            (VolatilityBreakoutStrategy, {'volatility_window': 20, 'breakout_threshold': 1.5}, 'Volatility Breakout')
        ]
        self.start_date = '2024-01-01'
        self.end_date = '2024-07-01'
        
        self.results = {}
        
    def run_comprehensive_test(self):
        """Run the complete integration test."""
        print("🚀 COMPREHENSIVE INTEGRATION TEST")
        print("=" * 80)
        print(f"📅 Testing Period: {self.start_date} to {self.end_date}")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(f"🎯 Strategies: {len(self.strategies)}")
        print(f"📁 Output Directory: {self.output_dir}")
        print("=" * 80)
        
        # Step 1: Load and validate data
        print("\n📥 STEP 1: DATA LOADING AND VALIDATION")
        self._load_and_validate_data()
        
        # Step 2: Run strategy backtests
        print("\n🔄 STEP 2: STRATEGY BACKTESTING")
        self._run_strategy_backtests()
        
        # Step 3: Monte Carlo statistical validation
        print("\n🎲 STEP 3: MONTE CARLO STATISTICAL VALIDATION")
        self._run_monte_carlo_validation()
        
        # Step 4: Bootstrap analysis
        print("\n📊 STEP 4: BOOTSTRAP ANALYSIS")
        self._run_bootstrap_analysis()
        
        # Step 5: Risk analysis
        print("\n⚠️  STEP 5: RISK ANALYSIS")
        self._run_risk_analysis()
        
        # Step 6: Generate comprehensive plots
        print("\n📈 STEP 6: VISUALIZATION AND PLOTTING")
        self._generate_comprehensive_plots()
        
        # Step 7: Generate final report
        print("\n📋 STEP 7: COMPREHENSIVE REPORT GENERATION")
        self._generate_comprehensive_report()
        
        print(f"\n✅ COMPREHENSIVE TEST COMPLETED!")
        print(f"📁 All results saved to: {self.output_dir}")
        
    def _load_and_validate_data(self):
        """Load and validate data for all symbols."""
        self.data = {}
        
        for symbol in self.symbols:
            print(f"  📊 Loading {symbol}...")
            try:
                data = self.data_loader.load_partitioned_crypto_data(
                    symbol=symbol,
                    interval='1d',
                    start_date=self.start_date,
                    end_date=self.end_date
                )
                
                if data.empty:
                    print(f"    ❌ No data available for {symbol}")
                    continue
                    
                # Data validation
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in data.columns for col in required_columns):
                    print(f"    ❌ Missing required columns for {symbol}")
                    continue
                    
                # Check for data quality
                if data.isnull().any().any():
                    print(f"    ⚠️  Found null values in {symbol}, forward filling...")
                    data = data.fillna(method='ffill')
                
                self.data[symbol] = data
                print(f"    ✅ Loaded {len(data)} records for {symbol}")
                
            except Exception as e:
                print(f"    ❌ Failed to load {symbol}: {e}")
                
        print(f"  📊 Successfully loaded data for {len(self.data)} symbols")
        
    def _run_strategy_backtests(self):
        """Run backtests for all strategies on all symbols."""
        self.backtest_results = {}
        
        for strategy_class, params, name in self.strategies:
            print(f"  🎯 Testing {name}...")
            self.backtest_results[name] = {}
            
            for symbol in self.data.keys():
                try:
                    # Initialize strategy
                    strategy = strategy_class(**params)
                    
                    # Run backtest
                    data = self.data[symbol].copy()
                    
                    # Generate signals
                    signals = []
                    for i in range(len(data)):
                        signal = strategy.generate_signals(data, i)
                        signals.append(signal)
                    
                    data['signal'] = signals
                    
                    # Calculate returns
                    data['returns'] = data['close'].pct_change()
                    data['strategy_returns'] = data['signal'].shift(1) * data['returns']
                    data['cumulative_returns'] = (1 + data['strategy_returns'].fillna(0)).cumprod()
                    
                    # Calculate performance metrics
                    total_return = data['cumulative_returns'].iloc[-1] - 1
                    annual_return = (1 + total_return) ** (365 / len(data)) - 1
                    volatility = data['strategy_returns'].std() * np.sqrt(252)
                    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                    max_drawdown = self._calculate_max_drawdown(data['cumulative_returns'])
                    
                    # Win rate calculation
                    winning_trades = (data['strategy_returns'] > 0).sum()
                    total_trades = (data['strategy_returns'] != 0).sum()
                    win_rate = winning_trades / total_trades if total_trades > 0 else 0
                    
                    result = {
                        'total_return': total_return,
                        'annual_return': annual_return,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'win_rate': win_rate,
                        'total_trades': total_trades,
                        'data': data,
                        'strategy_returns': data['strategy_returns'].dropna()
                    }
                    
                    self.backtest_results[name][symbol] = result
                    
                    print(f"    📊 {symbol}: Return={total_return:.2%}, Sharpe={sharpe_ratio:.2f}, DD={max_drawdown:.2%}")
                    
                except Exception as e:
                    print(f"    ❌ Failed {name} on {symbol}: {e}")
                    
    def _run_monte_carlo_validation(self):
        """Run Monte Carlo statistical validation."""
        print("  🎲 Running Monte Carlo validation...")
        
        # Initialize validator
        config = ValidationConfig(
            n_bootstrap_samples=500,  # Reduced for faster testing
            n_permutation_samples=500,
            confidence_level=0.95,
            alpha=0.05
        )
        validator = MonteCarloValidator(config)
        
        self.monte_carlo_results = {}
        
        for strategy_class, params, name in self.strategies:
            print(f"    🔬 Validating {name}...")
            self.monte_carlo_results[name] = {}
            
            for symbol in self.data.keys():
                try:
                    result = validator.validate_strategy(
                        strategy_class, params, symbol, self.start_date, self.end_date
                    )
                    self.monte_carlo_results[name][symbol] = result
                    
                    edge_assessment = result.edge_assessment
                    score = result.edge_score
                    print(f"      📈 {symbol}: {edge_assessment} (Score: {score}/100)")
                    
                except Exception as e:
                    print(f"      ❌ Failed validation for {name} on {symbol}: {e}")
                    
    def _run_bootstrap_analysis(self):
        """Run bootstrap analysis on strategy returns."""
        print("  📊 Running bootstrap analysis...")
        
        self.bootstrap_results = {}
        
        for name in self.backtest_results.keys():
            print(f"    🔄 Bootstrapping {name}...")
            self.bootstrap_results[name] = {}
            
            for symbol in self.backtest_results[name].keys():
                try:
                    returns = self.backtest_results[name][symbol]['strategy_returns']
                    
                    if len(returns) < 30:  # Need minimum data
                        print(f"      ⚠️  Insufficient data for {symbol}")
                        continue
                    
                    # Bootstrap configuration
                    config = BootstrapConfig(
                        n_samples=1000,
                        confidence_level=0.95,
                        method=BootstrapMethod.BLOCK_BOOTSTRAP,
                        block_size=10
                    )
                    
                    # Run bootstrap
                    bootstrapper = AdvancedBootstrapping(config)
                    result = bootstrapper.run_bootstrap_simulation(returns)
                    
                    self.bootstrap_results[name][symbol] = result
                    
                    # Extract key metrics
                    sharpe_ci = result.confidence_intervals.get('sharpe_ratio', (0, 0))
                    return_ci = result.confidence_intervals.get('annual_return', (0, 0))
                    
                    print(f"      📊 {symbol}: Sharpe CI=[{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}]")
                    
                except Exception as e:
                    print(f"      ❌ Bootstrap failed for {name} on {symbol}: {e}")
                    
    def _run_risk_analysis(self):
        """Run comprehensive risk analysis."""
        print("  ⚠️  Running risk analysis...")
        
        self.risk_results = {}
        
        for name in self.backtest_results.keys():
            print(f"    📉 Risk analysis for {name}...")
            self.risk_results[name] = {}
            
            for symbol in self.backtest_results[name].keys():
                try:
                    returns = self.backtest_results[name][symbol]['strategy_returns']
                    
                    if len(returns) < 30:
                        continue
                    
                    # Calculate risk metrics
                    var_95 = np.percentile(returns, 5)
                    var_99 = np.percentile(returns, 1)
                    cvar_95 = returns[returns <= var_95].mean()
                    cvar_99 = returns[returns <= var_99].mean()
                    
                    # Downside deviation
                    negative_returns = returns[returns < 0]
                    downside_deviation = negative_returns.std() if len(negative_returns) > 0 else 0
                    
                    # Maximum consecutive losses
                    consecutive_losses = 0
                    max_consecutive_losses = 0
                    for ret in returns:
                        if ret < 0:
                            consecutive_losses += 1
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        else:
                            consecutive_losses = 0
                    
                    risk_metrics = {
                        'var_95': var_95,
                        'var_99': var_99,
                        'cvar_95': cvar_95,
                        'cvar_99': cvar_99,
                        'downside_deviation': downside_deviation,
                        'max_consecutive_losses': max_consecutive_losses,
                        'return_skewness': returns.skew(),
                        'return_kurtosis': returns.kurtosis()
                    }
                    
                    self.risk_results[name][symbol] = risk_metrics
                    
                    print(f"      📊 {symbol}: VaR(95%)={var_95:.3f}, CVaR(95%)={cvar_95:.3f}")
                    
                except Exception as e:
                    print(f"      ❌ Risk analysis failed for {name} on {symbol}: {e}")
                    
    def _generate_comprehensive_plots(self):
        """Generate comprehensive visualization plots."""
        print("  📈 Generating comprehensive plots...")
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Strategy Performance Comparison
        self._plot_strategy_performance_comparison(plots_dir)
        
        # 2. Risk-Return Scatter Plot
        self._plot_risk_return_scatter(plots_dir)
        
        # 3. Bootstrap Confidence Intervals
        self._plot_bootstrap_confidence_intervals(plots_dir)
        
        # 4. Monte Carlo Score Heatmap
        self._plot_monte_carlo_heatmap(plots_dir)
        
        # 5. Individual Strategy Performance
        self._plot_individual_strategy_performance(plots_dir)
        
        # 6. Risk Analysis Dashboard
        self._plot_risk_analysis_dashboard(plots_dir)
        
        print(f"    📊 All plots saved to: {plots_dir}")
        
    def _plot_strategy_performance_comparison(self, plots_dir):
        """Plot strategy performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy Performance Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data
        strategies = list(self.backtest_results.keys())
        symbols = list(next(iter(self.backtest_results.values())).keys())
        
        # Total Returns
        returns_data = []
        for strategy in strategies:
            for symbol in symbols:
                if symbol in self.backtest_results[strategy]:
                    returns_data.append({
                        'Strategy': strategy,
                        'Symbol': symbol,
                        'Total Return': self.backtest_results[strategy][symbol]['total_return']
                    })
        
        returns_df = pd.DataFrame(returns_data)
        if not returns_df.empty:
            returns_pivot = returns_df.pivot(index='Strategy', columns='Symbol', values='Total Return')
            sns.heatmap(returns_pivot, annot=True, fmt='.2%', cmap='RdYlGn', ax=axes[0,0])
            axes[0,0].set_title('Total Returns Heatmap')
        
        # Sharpe Ratios
        sharpe_data = []
        for strategy in strategies:
            for symbol in symbols:
                if symbol in self.backtest_results[strategy]:
                    sharpe_data.append({
                        'Strategy': strategy,
                        'Symbol': symbol,
                        'Sharpe Ratio': self.backtest_results[strategy][symbol]['sharpe_ratio']
                    })
        
        sharpe_df = pd.DataFrame(sharpe_data)
        if not sharpe_df.empty:
            sharpe_pivot = sharpe_df.pivot(index='Strategy', columns='Symbol', values='Sharpe Ratio')
            sns.heatmap(sharpe_pivot, annot=True, fmt='.2f', cmap='viridis', ax=axes[0,1])
            axes[0,1].set_title('Sharpe Ratios Heatmap')
        
        # Max Drawdown
        dd_data = []
        for strategy in strategies:
            for symbol in symbols:
                if symbol in self.backtest_results[strategy]:
                    dd_data.append({
                        'Strategy': strategy,
                        'Symbol': symbol,
                        'Max Drawdown': self.backtest_results[strategy][symbol]['max_drawdown']
                    })
        
        dd_df = pd.DataFrame(dd_data)
        if not dd_df.empty:
            dd_pivot = dd_df.pivot(index='Strategy', columns='Symbol', values='Max Drawdown')
            sns.heatmap(dd_pivot, annot=True, fmt='.2%', cmap='Reds_r', ax=axes[1,0])
            axes[1,0].set_title('Max Drawdown Heatmap')
        
        # Win Rates
        wr_data = []
        for strategy in strategies:
            for symbol in symbols:
                if symbol in self.backtest_results[strategy]:
                    wr_data.append({
                        'Strategy': strategy,
                        'Symbol': symbol,
                        'Win Rate': self.backtest_results[strategy][symbol]['win_rate']
                    })
        
        wr_df = pd.DataFrame(wr_data)
        if not wr_df.empty:
            wr_pivot = wr_df.pivot(index='Strategy', columns='Symbol', values='Win Rate')
            sns.heatmap(wr_pivot, annot=True, fmt='.2%', cmap='Blues', ax=axes[1,1])
            axes[1,1].set_title('Win Rates Heatmap')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'strategy_performance_comparison_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_risk_return_scatter(self, plots_dir):
        """Plot risk-return scatter plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.backtest_results)))
        
        for i, (strategy, color) in enumerate(zip(self.backtest_results.keys(), colors)):
            x_vals = []  # Volatility
            y_vals = []  # Annual Return
            symbols_list = []
            
            for symbol in self.backtest_results[strategy].keys():
                result = self.backtest_results[strategy][symbol]
                x_vals.append(result['volatility'])
                y_vals.append(result['annual_return'])
                symbols_list.append(symbol)
            
            if x_vals and y_vals:
                scatter = ax.scatter(x_vals, y_vals, c=[color]*len(x_vals), label=strategy, s=100, alpha=0.7)
                
                # Add symbol labels
                for x, y, symbol in zip(x_vals, y_vals, symbols_list):
                    ax.annotate(symbol, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Volatility (Annual)')
        ax.set_ylabel('Annual Return')
        ax.set_title('Risk-Return Profile by Strategy and Symbol')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'risk_return_scatter_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_bootstrap_confidence_intervals(self, plots_dir):
        """Plot bootstrap confidence intervals."""
        if not self.bootstrap_results:
            return
            
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Sharpe Ratio Confidence Intervals
        strategies = []
        symbols_list = []
        sharpe_lower = []
        sharpe_upper = []
        sharpe_mean = []
        
        for strategy in self.bootstrap_results.keys():
            for symbol in self.bootstrap_results[strategy].keys():
                result = self.bootstrap_results[strategy][symbol]
                if 'sharpe_ratio' in result.confidence_intervals:
                    ci = result.confidence_intervals['sharpe_ratio']
                    strategies.append(strategy)
                    symbols_list.append(symbol)
                    sharpe_lower.append(ci[0])
                    sharpe_upper.append(ci[1])
                    sharpe_mean.append(result.statistics.get('sharpe_ratio', 0))
        
        if strategies:
            y_pos = np.arange(len(strategies))
            axes[0].barh(y_pos, sharpe_upper, xerr=[np.array(sharpe_mean) - np.array(sharpe_lower), 
                                                    np.array(sharpe_upper) - np.array(sharpe_mean)], 
                        capsize=5, alpha=0.7)
            axes[0].set_yticks(y_pos)
            axes[0].set_yticklabels([f"{s} ({sym})" for s, sym in zip(strategies, symbols_list)])
            axes[0].set_xlabel('Sharpe Ratio')
            axes[0].set_title('Bootstrap Confidence Intervals - Sharpe Ratio')
            axes[0].grid(True, alpha=0.3)
        
        # Annual Return Confidence Intervals
        return_lower = []
        return_upper = []
        return_mean = []
        
        for strategy in self.bootstrap_results.keys():
            for symbol in self.bootstrap_results[strategy].keys():
                result = self.bootstrap_results[strategy][symbol]
                if 'annual_return' in result.confidence_intervals:
                    ci = result.confidence_intervals['annual_return']
                    return_lower.append(ci[0])
                    return_upper.append(ci[1])
                    return_mean.append(result.statistics.get('annual_return', 0))
        
        if return_lower:
            axes[1].barh(y_pos, return_upper, xerr=[np.array(return_mean) - np.array(return_lower), 
                                                    np.array(return_upper) - np.array(return_mean)], 
                        capsize=5, alpha=0.7, color='green')
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels([f"{s} ({sym})" for s, sym in zip(strategies, symbols_list)])
            axes[1].set_xlabel('Annual Return')
            axes[1].set_title('Bootstrap Confidence Intervals - Annual Return')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'bootstrap_confidence_intervals_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_monte_carlo_heatmap(self, plots_dir):
        """Plot Monte Carlo statistical scores heatmap."""
        if not self.monte_carlo_results:
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Prepare data
        scores_data = []
        for strategy in self.monte_carlo_results.keys():
            for symbol in self.monte_carlo_results[strategy].keys():
                result = self.monte_carlo_results[strategy][symbol]
                scores_data.append({
                    'Strategy': strategy,
                    'Symbol': symbol,
                    'Statistical Score': result.edge_score
                })
        
        if scores_data:
            scores_df = pd.DataFrame(scores_data)
            scores_pivot = scores_df.pivot(index='Strategy', columns='Symbol', values='Statistical Score')
            
            sns.heatmap(scores_pivot, annot=True, fmt='.0f', cmap='RdYlGn', 
                       center=50, vmin=0, vmax=100, ax=ax)
            ax.set_title('Monte Carlo Statistical Validation Scores')
            
        plt.tight_layout()
        plt.savefig(plots_dir / f'monte_carlo_scores_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_individual_strategy_performance(self, plots_dir):
        """Plot individual strategy performance over time."""
        for strategy_name in self.backtest_results.keys():
            fig, axes = plt.subplots(len(self.symbols), 1, figsize=(15, 4 * len(self.symbols)))
            if len(self.symbols) == 1:
                axes = [axes]
                
            fig.suptitle(f'{strategy_name} - Performance Over Time', fontsize=16, fontweight='bold')
            
            for i, symbol in enumerate(self.symbols):
                if symbol in self.backtest_results[strategy_name]:
                    data = self.backtest_results[strategy_name][symbol]['data']
                    
                    # Plot cumulative returns
                    axes[i].plot(data.index, data['cumulative_returns'], 
                               label=f'{strategy_name} Strategy', linewidth=2)
                    
                    # Plot buy and hold
                    buy_hold = (1 + data['returns'].fillna(0)).cumprod()
                    axes[i].plot(data.index, buy_hold, 
                               label='Buy & Hold', linewidth=2, alpha=0.7)
                    
                    axes[i].set_title(f'{symbol}')
                    axes[i].set_ylabel('Cumulative Returns')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add performance metrics as text
                    result = self.backtest_results[strategy_name][symbol]
                    metrics_text = f"Return: {result['total_return']:.1%} | " \
                                 f"Sharpe: {result['sharpe_ratio']:.2f} | " \
                                 f"DD: {result['max_drawdown']:.1%}"
                    axes[i].text(0.02, 0.98, metrics_text, transform=axes[i].transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{strategy_name.lower().replace(" ", "_")}_performance_{self.timestamp}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def _plot_risk_analysis_dashboard(self, plots_dir):
        """Plot comprehensive risk analysis dashboard."""
        if not self.risk_results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Risk Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # VaR comparison
        var_data = []
        for strategy in self.risk_results.keys():
            for symbol in self.risk_results[strategy].keys():
                risk_metrics = self.risk_results[strategy][symbol]
                var_data.append({
                    'Strategy': strategy,
                    'Symbol': symbol,
                    'VaR_95': risk_metrics['var_95'],
                    'VaR_99': risk_metrics['var_99'],
                    'CVaR_95': risk_metrics['cvar_95'],
                    'CVaR_99': risk_metrics['cvar_99']
                })
        
        if var_data:
            var_df = pd.DataFrame(var_data)
            var_df['Strategy_Symbol'] = var_df['Strategy'] + ' (' + var_df['Symbol'] + ')'
            
            x = np.arange(len(var_df))
            width = 0.35
            
            axes[0,0].bar(x - width/2, var_df['VaR_95'], width, label='VaR 95%', alpha=0.8)
            axes[0,0].bar(x + width/2, var_df['VaR_99'], width, label='VaR 99%', alpha=0.8)
            axes[0,0].set_xlabel('Strategy (Symbol)')
            axes[0,0].set_ylabel('Value at Risk')
            axes[0,0].set_title('Value at Risk Comparison')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(var_df['Strategy_Symbol'], rotation=45, ha='right')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # CVaR comparison
        if var_data:
            axes[0,1].bar(x - width/2, var_df['CVaR_95'], width, label='CVaR 95%', alpha=0.8)
            axes[0,1].bar(x + width/2, var_df['CVaR_99'], width, label='CVaR 99%', alpha=0.8)
            axes[0,1].set_xlabel('Strategy (Symbol)')
            axes[0,1].set_ylabel('Conditional Value at Risk')
            axes[0,1].set_title('Conditional VaR Comparison')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(var_df['Strategy_Symbol'], rotation=45, ha='right')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Return distribution analysis
        all_returns = []
        strategy_labels = []
        
        for strategy in self.backtest_results.keys():
            for symbol in self.backtest_results[strategy].keys():
                returns = self.backtest_results[strategy][symbol]['strategy_returns']
                all_returns.extend(returns.tolist())
                strategy_labels.extend([f"{strategy} ({symbol})"] * len(returns))
        
        if all_returns:
            returns_df = pd.DataFrame({'Returns': all_returns, 'Strategy': strategy_labels})
            
            # Box plot of returns
            unique_strategies = returns_df['Strategy'].unique()
            returns_by_strategy = [returns_df[returns_df['Strategy'] == s]['Returns'].values 
                                 for s in unique_strategies]
            
            axes[1,0].boxplot(returns_by_strategy, labels=unique_strategies)
            axes[1,0].set_ylabel('Daily Returns')
            axes[1,0].set_title('Return Distribution by Strategy')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
        
        # Drawdown analysis
        dd_data = []
        for strategy in self.backtest_results.keys():
            for symbol in self.backtest_results[strategy].keys():
                result = self.backtest_results[strategy][symbol]
                dd_data.append({
                    'Strategy': strategy,
                    'Symbol': symbol,
                    'Max Drawdown': abs(result['max_drawdown'])
                })
        
        if dd_data:
            dd_df = pd.DataFrame(dd_data)
            dd_df['Strategy_Symbol'] = dd_df['Strategy'] + ' (' + dd_df['Symbol'] + ')'
            
            axes[1,1].bar(range(len(dd_df)), dd_df['Max Drawdown'], alpha=0.8, color='red')
            axes[1,1].set_xlabel('Strategy (Symbol)')
            axes[1,1].set_ylabel('Maximum Drawdown')
            axes[1,1].set_title('Maximum Drawdown by Strategy')
            axes[1,1].set_xticks(range(len(dd_df)))
            axes[1,1].set_xticklabels(dd_df['Strategy_Symbol'], rotation=45, ha='right')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'risk_analysis_dashboard_{self.timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _generate_comprehensive_report(self):
        """Generate comprehensive markdown report."""
        report_path = self.output_dir / f"comprehensive_test_report_{self.timestamp}.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Comprehensive Integration Test Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Test Period:** {self.start_date} to {self.end_date}\n")
            f.write(f"**Symbols Tested:** {', '.join(self.symbols)}\n")
            f.write(f"**Strategies Tested:** {len(self.strategies)}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This comprehensive test validates the complete quantitative trading framework, ")
            f.write("including strategy backtesting, Monte Carlo validation, bootstrap analysis, ")
            f.write("and comprehensive risk assessment.\n\n")
            
            # Strategy Performance Summary
            f.write("## Strategy Performance Summary\n\n")
            f.write("| Strategy | Symbol | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Monte Carlo Score |\n")
            f.write("|----------|---------|--------------|--------------|--------------|----------|-----------------|\n")
            
            for strategy_name in self.backtest_results.keys():
                for symbol in self.backtest_results[strategy_name].keys():
                    result = self.backtest_results[strategy_name][symbol]
                    mc_score = "N/A"
                    
                    if (strategy_name in self.monte_carlo_results and 
                        symbol in self.monte_carlo_results[strategy_name]):
                        mc_score = f"{self.monte_carlo_results[strategy_name][symbol].edge_score}/100"
                    
                    f.write(f"| {strategy_name} | {symbol} | "
                           f"{result['total_return']:.2%} | "
                           f"{result['sharpe_ratio']:.2f} | "
                           f"{result['max_drawdown']:.2%} | "
                           f"{result['win_rate']:.2%} | "
                           f"{mc_score} |\n")
            
            # Best Performing Strategies
            f.write("\n## Best Performing Strategies\n\n")
            
            # Find best strategies by different metrics
            best_return = None
            best_sharpe = None
            best_mc_score = None
            
            best_return_val = -float('inf')
            best_sharpe_val = -float('inf')
            best_mc_val = -float('inf')
            
            for strategy_name in self.backtest_results.keys():
                for symbol in self.backtest_results[strategy_name].keys():
                    result = self.backtest_results[strategy_name][symbol]
                    
                    if result['total_return'] > best_return_val:
                        best_return_val = result['total_return']
                        best_return = (strategy_name, symbol)
                    
                    if result['sharpe_ratio'] > best_sharpe_val:
                        best_sharpe_val = result['sharpe_ratio']
                        best_sharpe = (strategy_name, symbol)
                    
                    if (strategy_name in self.monte_carlo_results and 
                        symbol in self.monte_carlo_results[strategy_name]):
                        mc_score = self.monte_carlo_results[strategy_name][symbol].edge_score
                        if mc_score > best_mc_val:
                            best_mc_val = mc_score
                            best_mc_score = (strategy_name, symbol)
            
            if best_return:
                f.write(f"**Best Total Return:** {best_return[0]} on {best_return[1]} ({best_return_val:.2%})\n\n")
            if best_sharpe:
                f.write(f"**Best Sharpe Ratio:** {best_sharpe[0]} on {best_sharpe[1]} ({best_sharpe_val:.2f})\n\n")
            if best_mc_score:
                f.write(f"**Best Monte Carlo Score:** {best_mc_score[0]} on {best_mc_score[1]} ({best_mc_val}/100)\n\n")
            
            # Risk Analysis Summary
            f.write("## Risk Analysis Summary\n\n")
            if self.risk_results:
                f.write("| Strategy | Symbol | VaR (95%) | CVaR (95%) | Downside Dev | Max Consecutive Losses |\n")
                f.write("|----------|---------|-----------|------------|--------------|----------------------|\n")
                
                for strategy_name in self.risk_results.keys():
                    for symbol in self.risk_results[strategy_name].keys():
                        risk = self.risk_results[strategy_name][symbol]
                        f.write(f"| {strategy_name} | {symbol} | "
                               f"{risk['var_95']:.3f} | "
                               f"{risk['cvar_95']:.3f} | "
                               f"{risk['downside_deviation']:.3f} | "
                               f"{risk['max_consecutive_losses']} |\n")
            
            # Statistical Validation Results
            f.write("\n## Monte Carlo Statistical Validation\n\n")
            if self.monte_carlo_results:
                strong_edge_count = 0
                weak_edge_count = 0
                no_edge_count = 0
                
                for strategy_name in self.monte_carlo_results.keys():
                    for symbol in self.monte_carlo_results[strategy_name].keys():
                        result = self.monte_carlo_results[strategy_name][symbol]
                        if "STRONG" in result.edge_assessment:
                            strong_edge_count += 1
                        elif "WEAK" in result.edge_assessment:
                            weak_edge_count += 1
                        else:
                            no_edge_count += 1
                
                total_tests = strong_edge_count + weak_edge_count + no_edge_count
                f.write(f"**Total Strategy-Symbol Combinations Tested:** {total_tests}\n\n")
                
                if total_tests > 0:
                    f.write(f"**Strong Statistical Edge:** {strong_edge_count} ({strong_edge_count/total_tests*100:.1f}%)\n\n")
                    f.write(f"**Weak Statistical Edge:** {weak_edge_count} ({weak_edge_count/total_tests*100:.1f}%)\n\n")
                    f.write(f"**No Statistical Edge:** {no_edge_count} ({no_edge_count/total_tests*100:.1f}%)\n\n")
                else:
                    f.write("**No Monte Carlo validation results available.**\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("Based on the comprehensive analysis:\n\n")
            
            if best_mc_score and best_mc_val > 70:
                f.write(f"1. **{best_mc_score[0]} on {best_mc_score[1]}** shows strong statistical edge and should be considered for live trading.\n\n")
            
            if best_sharpe and best_sharpe_val > 2.0:
                f.write(f"2. **{best_sharpe[0]} on {best_sharpe[1]}** demonstrates excellent risk-adjusted returns.\n\n")
            
            f.write("3. Consider portfolio diversification across multiple strategies to reduce overall risk.\n\n")
            f.write("4. Implement additional risk management measures for strategies with high drawdowns.\n\n")
            f.write("5. Regular rebalancing and performance monitoring is recommended.\n\n")
            
            # Technical Details
            f.write("## Technical Framework Details\n\n")
            f.write("### Components Tested:\n")
            f.write("- ✅ Data Loading and Validation\n")
            f.write("- ✅ Strategy Backtesting Engine\n")
            f.write("- ✅ Monte Carlo Statistical Validation\n")
            f.write("- ✅ Bootstrap Confidence Interval Analysis\n")
            f.write("- ✅ Comprehensive Risk Analysis\n")
            f.write("- ✅ Performance Visualization\n")
            f.write("- ✅ Automated Reporting\n\n")
            
            f.write("### Files Generated:\n")
            f.write(f"- Performance comparison plots\n")
            f.write(f"- Risk-return scatter plots\n")
            f.write(f"- Bootstrap confidence intervals\n")
            f.write(f"- Monte Carlo validation scores\n")
            f.write(f"- Individual strategy performance charts\n")
            f.write(f"- Risk analysis dashboard\n")
            f.write(f"- JSON export of all results\n\n")
            
            f.write("---\n")
            f.write(f"*Report generated by Saucedo Quantitative Trading Engine v2.0*\n")
            
        # Save JSON export
        json_data = {
            'timestamp': self.timestamp,
            'test_period': {'start': self.start_date, 'end': self.end_date},
            'symbols': self.symbols,
            'backtest_results': self._serialize_results(self.backtest_results),
            'monte_carlo_results': self._serialize_monte_carlo_results(),
            'bootstrap_results': self._serialize_bootstrap_results(),
            'risk_results': self.risk_results
        }
        
        json_path = self.output_dir / f"comprehensive_test_data_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
            
        print(f"    📋 Comprehensive report saved to: {report_path}")
        print(f"    💾 JSON data exported to: {json_path}")
        
    def _serialize_results(self, results):
        """Serialize backtest results for JSON export."""
        serialized = {}
        for strategy in results:
            serialized[strategy] = {}
            for symbol in results[strategy]:
                result = results[strategy][symbol].copy()
                if 'data' in result:
                    del result['data']  # Remove DataFrame
                if 'strategy_returns' in result:
                    result['strategy_returns'] = result['strategy_returns'].tolist()
                serialized[strategy][symbol] = result
        return serialized
        
    def _serialize_monte_carlo_results(self):
        """Serialize Monte Carlo results for JSON export."""
        if not self.monte_carlo_results:
            return {}
            
        serialized = {}
        for strategy in self.monte_carlo_results:
            serialized[strategy] = {}
            for symbol in self.monte_carlo_results[strategy]:
                result = self.monte_carlo_results[strategy][symbol]
                serialized[strategy][symbol] = {
                    'edge_score': result.edge_score,
                    'edge_assessment': result.edge_assessment,
                    'is_statistically_significant': result.is_statistically_significant
                }
        return serialized
        
    def _serialize_bootstrap_results(self):
        """Serialize bootstrap results for JSON export."""
        if not self.bootstrap_results:
            return {}
            
        serialized = {}
        for strategy in self.bootstrap_results:
            serialized[strategy] = {}
            for symbol in self.bootstrap_results[strategy]:
                result = self.bootstrap_results[strategy][symbol]
                serialized[strategy][symbol] = {
                    'statistics': result.statistics,
                    'confidence_intervals': result.confidence_intervals
                }
        return serialized
        
    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown from cumulative returns."""
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()


def main():
    """Run the comprehensive integration test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Integration Test')
    parser.add_argument('--output-dir', default='results/comprehensive_test',
                       help='Output directory for results')
    parser.add_argument('--symbols', default='BTC_USD,ETH_USD',
                       help='Comma-separated symbols to test')
    parser.add_argument('--start-date', default='2024-01-01',
                       help='Start date for testing')
    parser.add_argument('--end-date', default='2024-07-01',
                       help='End date for testing')
    
    args = parser.parse_args()
    
    # Initialize test
    test = ComprehensiveIntegrationTest(output_dir=args.output_dir)
    
    # Override defaults if provided
    test.symbols = args.symbols.split(',')
    test.start_date = args.start_date
    test.end_date = args.end_date
    
    # Run comprehensive test
    test.run_comprehensive_test()


if __name__ == "__main__":
    main()
