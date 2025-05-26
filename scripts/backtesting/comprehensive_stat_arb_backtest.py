#!/usr/bin/env python3
"""
Comprehensive Statistical Arbitrage Backtesting with Advanced Statistical Validation

This script implements a complete backtesting pipeline with:
1. In-Sample (2017-2022) and Out-of-Sample (2023-2024) data partitioning
2. Advanced statistical validation using bootstrap methods
3. Extensive visualizations including signals, performance, and statistical tests
4. Clear indicators of statistical edge presence

Author: Saucedo Quantitative Trading Engine
Date: 2025-05-25
"""

import sys
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'research'))

# Core imports
from utils.data_loader import DataLoader
from bootstrapping.core import AdvancedBootstrapping
from bootstrapping.statistical_tests import StatisticalTests
from utils.metrics import calculate_basic_metrics, calculate_trade_metrics
from utils.validation import DataValidator

# Import strategies
from src.strategies.composite_pair_trading_strategy import CompositePairTradingStrategy, CompositePairTradingOptimizer

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore', category=FutureWarning)

class StatisticalEdgeAnalyzer:
    """
    Comprehensive analyzer to determine if a strategy has statistical edge
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.edge_indicators = {}
        
    def analyze_edge(self, 
                    permutation_results: Dict[str, Any],
                    bootstrap_results: Dict[str, Any],
                    traditional_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive edge analysis combining multiple statistical tests
        """
        edge_score = 0
        max_score = 10  # Total possible points
        
        # 1. Permutation test on returns (2 points)
        if 'permutation_sign_test' in permutation_results:
            sign_test = permutation_results['permutation_sign_test']
            if sign_test['p_value'] < self.alpha:
                edge_score += 2
                self.edge_indicators['returns_significance'] = True
            else:
                self.edge_indicators['returns_significance'] = False
                
        # 2. Sharpe ratio permutation test (2 points)
        if 'permutation_sharpe_test' in permutation_results:
            sharpe_test = permutation_results['permutation_sharpe_test']
            if sharpe_test['p_value'] < self.alpha:
                edge_score += 2
                self.edge_indicators['sharpe_significance'] = True
            else:
                self.edge_indicators['sharpe_significance'] = False
                
        # 3. Profit factor test (2 points)
        if 'permutation_pf_test' in permutation_results:
            pf_test = permutation_results['permutation_pf_test']
            if pf_test['p_value'] < self.alpha:
                edge_score += 2
                self.edge_indicators['profit_factor_significance'] = True
            else:
                self.edge_indicators['profit_factor_significance'] = False
                
        # 4. Traditional metrics thresholds (2 points)
        metrics_score = 0
        if traditional_metrics.get('sharpe_ratio', 0) > 1.0:
            metrics_score += 1
        if traditional_metrics.get('win_rate', 0) > 0.55:
            metrics_score += 1
        edge_score += metrics_score
        self.edge_indicators['traditional_metrics_strong'] = metrics_score >= 1
        
        # 5. Bootstrap consistency (2 points)
        if 'simulated_stats' in bootstrap_results:
            simulated_sharpes = [s['Sharpe'] for s in bootstrap_results['simulated_stats'] if not np.isnan(s['Sharpe'])]
            if len(simulated_sharpes) > 0:
                positive_sharpe_pct = np.mean(np.array(simulated_sharpes) > 0)
                if positive_sharpe_pct > 0.7:  # 70% of bootstrap samples have positive Sharpe
                    edge_score += 2
                    self.edge_indicators['bootstrap_consistency'] = True
                else:
                    self.edge_indicators['bootstrap_consistency'] = False
        
        # Calculate final edge assessment
        edge_percentage = (edge_score / max_score) * 100
        
        if edge_percentage >= 80:
            edge_assessment = "STRONG STATISTICAL EDGE"
        elif edge_percentage >= 60:
            edge_assessment = "MODERATE STATISTICAL EDGE"
        elif edge_percentage >= 40:
            edge_assessment = "WEAK STATISTICAL EDGE"
        else:
            edge_assessment = "NO STATISTICAL EDGE"
            
        return {
            'edge_score': edge_score,
            'max_score': max_score,
            'edge_percentage': edge_percentage,
            'edge_assessment': edge_assessment,
            'indicators': self.edge_indicators,
            'recommendation': self._get_recommendation(edge_percentage)
        }
    
    def _get_recommendation(self, edge_percentage: float) -> str:
        """Get deployment recommendation based on edge percentage"""
        if edge_percentage >= 80:
            return "DEPLOY: Strong statistical evidence supports live trading"
        elif edge_percentage >= 60:
            return "PROCEED WITH CAUTION: Moderate evidence, consider paper trading first"
        elif edge_percentage >= 40:
            return "OPTIMIZE: Weak evidence, requires parameter tuning or strategy modification"
        else:
            return "REJECT: Insufficient statistical evidence for deployment"


class ComprehensiveBacktester:
    """
    Main backtesting class orchestrating the entire analysis pipeline
    """
    
    def __init__(self, strategy_type: str = "composite"):
        self.data_loader = DataLoader()
        self.results = {}
        self.plots_dir = Path("results/plots")
        self.reports_dir = Path("results/reports")
        self.strategy_type = strategy_type  # "composite" or "entropy"
        
        # Create directories
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Date ranges
        self.insample_start = "2017-01-01"
        self.insample_end = "2022-12-31"
        self.oos_start = "2023-01-01" 
        self.oos_end = "2024-12-31"
        
        print("=" * 80)
        print("COMPREHENSIVE STATISTICAL ARBITRAGE BACKTESTING PIPELINE")
        print("=" * 80)
        print(f"Strategy Type: {strategy_type.upper()}")
        print(f"In-Sample Period: {self.insample_start} to {self.insample_end}")
        print(f"Out-of-Sample Period: {self.oos_start} to {self.oos_end}")
        print("=" * 80)
        
    def load_and_prepare_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Load BTC_USD and ETH_USD data and prepare for entropy strategy
        """
        print("\n📊 PHASE 1: DATA LOADING AND PREPARATION")
        print("-" * 50)
        
        try:
            # Load individual assets using correct method and symbol format
            print("Loading BTC_USD data...")
            btc_data = self.data_loader.load_partitioned_crypto_data(
                "BTC_USD", "1d", "2017-01-01", "2024-12-31"
            )
            
            print("Loading ETH_USD data...")
            eth_data = self.data_loader.load_partitioned_crypto_data(
                "ETH_USD", "1d", "2017-01-01", "2024-12-31"
            )
            
            if btc_data.empty or eth_data.empty:
                raise ValueError("Failed to load data for BTC_USD or ETH_USD")
            
            print(f"BTC data shape: {btc_data.shape}")
            print(f"ETH data shape: {eth_data.shape}")
            print(f"Date range: {btc_data.index.min()} to {btc_data.index.max()}")
            
            # Prepare data dictionaries for entropy strategy
            full_data = {'BTC_USD': btc_data, 'ETH_USD': eth_data}
            
            # Split into in-sample and out-of-sample periods
            insample_data = {}
            oos_data = {}
            
            for symbol, df in full_data.items():
                insample_mask = (df.index >= self.insample_start) & (df.index <= self.insample_end)
                oos_mask = (df.index >= self.oos_start) & (df.index <= self.oos_end)
                
                insample_data[symbol] = df[insample_mask].copy()
                oos_data[symbol] = df[oos_mask].copy()
                
                print(f"{symbol} - In-sample: {len(insample_data[symbol])} records")
                print(f"{symbol} - Out-of-sample: {len(oos_data[symbol])} records")
            
            print("✅ Data loading completed successfully")
            return insample_data, oos_data
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
            
    def optimize_strategy_parameters(self, train_data: Dict[str, pd.DataFrame], val_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Optimize strategy parameters using in-sample data with validation
        """
        print("\n🔍 PARAMETER OPTIMIZATION")
        print("-" * 40)
        
        if self.strategy_type == "composite":
            # Initialize composite pair trading optimizer
            optimizer = CompositePairTradingOptimizer(asset_pair=('BTC_USD', 'ETH_USD'))
            
            try:
                print("🎯 Running composite pair trading optimization...")
                optimization_results = optimizer.grid_search_optimization(
                    train_data, val_data, max_combinations=30  # Limit for performance
                )
                
                print(f"✅ Optimization completed:")
                print(f"   • Best score: {optimization_results['best_score']:.4f}")
                print(f"   • Combinations tested: {optimization_results['total_tested']}")
                
                # Get optimal parameters
                optimal_params = optimizer.get_optimal_parameters()
                
                print("🎯 Optimal parameters found:")
                for key, value in optimal_params.items():
                    if key != 'asset_pair':
                        print(f"   • {key}: {value}")
                        
                return {
                    'optimal_params': optimal_params,
                    'optimization_results': optimization_results,
                    'optimizer': optimizer
                }
                
            except Exception as e:
                print(f"⚠️  Optimization failed: {e}")
                print("🔧 Using default parameters...")
                
                # Return default optimal parameters
                default_params = optimizer.get_optimal_parameters()
                return {
                    'optimal_params': default_params,
                    'optimization_results': None,
                    'optimizer': optimizer
                }
                
        else:  # entropy strategy
            # Initialize parameter optimizer
            optimizer = ParameterOptimizer(asset_pair=('BTC_USD', 'ETH_USD'))
            
            try:
                print("🎯 Running entropy strategy optimization...")
                optimization_results = optimizer.grid_search_optimization(
                    train_data, val_data, max_combinations=30  # Limit for performance
                )
                
                print(f"✅ Optimization completed:")
                print(f"   • Best score: {optimization_results['best_score']:.4f}")
                print(f"   • Combinations tested: {len(optimization_results['results'])}")
                
                # Get optimal parameters
                optimal_params = optimizer.get_optimal_parameters()
                
                print("🎯 Optimal parameters found:")
                for key, value in optimal_params.items():
                    if key != 'asset_pair':
                        print(f"   • {key}: {value}")
                        
                return {
                    'optimal_params': optimal_params,
                    'optimization_results': optimization_results,
                    'optimizer': optimizer
                }
                
            except Exception as e:
                print(f"⚠️  Optimization failed: {e}")
                print("🔧 Using default parameters...")
                
                # Return default optimal parameters
                default_params = optimizer.get_optimal_parameters()
                return {
                    'optimal_params': default_params,
                    'optimization_results': None,
                    'optimizer': optimizer
                }

    def run_strategy_backtest(self, data: Dict[str, pd.DataFrame], period_name: str, strategy_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run strategy backtest on given data period with optimized parameters
        """
        print(f"\n🚀 Running {period_name} backtest...")
        
        if self.strategy_type == "composite":
            # Use provided parameters or defaults for composite strategy
            if strategy_params is None:
                strategy_params = {
                    'asset_pair': ('BTC_USD', 'ETH_USD'),
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
            
            # Initialize composite pair trading strategy with optimized parameters
            strategy = CompositePairTradingStrategy(
                name=f"CompositePairTrading_{period_name}",
                parameters=strategy_params
            )
            
            # Run backtest
            backtest_results = strategy.backtest(data, symbol="BTC-ETH-PAIR")
            
        else:  # entropy strategy
            # Use provided parameters or defaults for entropy strategy
            if strategy_params is None:
                strategy_params = {
                    'asset_pair': ('BTC_USD', 'ETH_USD'),
                    'window_entropy': 30,
                    'window_transfer_entropy': 40,
                    'te_lag': 2,
                    'theta_entry': 0.8,
                    'theta_exit': 0.2,
                    'n_bins': 10,
                    'kappa': 0.5,
                    'risk_budget': 0.02,
                    'vol_window': 30,
                    'beta_window': 120
                }
            
            # Initialize enhanced entropy strategy with optimized parameters
            strategy = EntropyDrivenStatArbEnhanced(
                name=f"EntropyStatArbEnhanced_{period_name}",
                parameters=strategy_params
            )
            
            # Run backtest
            backtest_results = strategy.backtest(data, symbol="BTC-ETH-ENTROPY")
        
        # Extract key information
        returns = backtest_results['returns']
        equity_curve = backtest_results.get('portfolio_value', [])
        trades = backtest_results.get('trades', [])
        signals = backtest_results.get('signals', [])
        metrics = backtest_results.get('metrics', {})
        
        print(f"✅ {period_name} backtest completed:")
        print(f"   • Total trades: {len(trades)}")
        print(f"   • Total return: {metrics.get('total_return', 0):.2%}")
        print(f"   • Sharpe ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"   • Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   • Win rate: {metrics.get('win_rate', 0):.2%}")
        
        return {
            'returns': returns,
            'equity_curve': equity_curve,
            'trades': trades,
            'signals': signals,
            'metrics': metrics,
            'strategy': strategy
        }
    
    def calculate_buy_and_hold_benchmark(self, data: Dict[str, pd.DataFrame], period_name: str) -> Dict[str, Any]:
        """
        Calculate buy and hold benchmark for comparison
        """
        print(f"\n📈 Calculating Buy & Hold benchmark for {period_name}...")
        
        # Use BTC as the benchmark asset
        btc_prices = data['BTC_USD']['close']
        
        # Calculate daily returns
        bh_returns = btc_prices.pct_change().dropna()
        
        # Calculate cumulative returns
        cumulative_returns = (1 + bh_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = total_return / (len(bh_returns) / 252)
        volatility = bh_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Calculate maximum drawdown
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (bh_returns > 0).mean()
        };
        
        print(f"✅ Buy & Hold {period_name} metrics:")
        print(f"   • Total return: {total_return:.2%}")
        print(f"   • Sharpe ratio: {sharpe_ratio:.3f}")
        print(f"   • Max drawdown: {max_drawdown:.2%}")
        
        return {
            'returns': bh_returns,
            'equity_curve': cumulative_returns,
            'metrics': metrics
        }
    
    def execute_backtesting_phase(self, insample_data: Dict[str, pd.DataFrame], oos_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Execute complete backtesting on both in-sample and out-of-sample data with parameter optimization
        """
        print("\n🔥 PHASE 2: BACKTESTING EXECUTION")
        print("-" * 50)
        
        results = {}
        
        # 0. Parameter Optimization Phase
        print("\n🎯 Parameter Optimization Phase")
        
        # Split in-sample data for optimization (80% train, 20% validation)
        split_idx = int(len(insample_data['BTC_USD']) * 0.8)
        
        train_data = {}
        val_data = {}
        for symbol in insample_data.keys():
            train_data[symbol] = insample_data[symbol].iloc[:split_idx].copy()
            val_data[symbol] = insample_data[symbol].iloc[split_idx:].copy()
        
        print(f"   • Training data: {len(train_data['BTC_USD'])} records")
        print(f"   • Validation data: {len(val_data['BTC_USD'])} records")
        
        # Run parameter optimization
        optimization_results = self.optimize_strategy_parameters(train_data, val_data)
        optimal_params = optimization_results['optimal_params']
        
        # Store optimization results
        results['parameter_optimization'] = optimization_results
        
        # 1. In-Sample Backtesting with optimized parameters
        print("\n📊 In-Sample Backtesting (2017-2022)")
        results['insample_strategy'] = self.run_strategy_backtest(insample_data, "In-Sample", optimal_params)
        results['insample_benchmark'] = self.calculate_buy_and_hold_benchmark(insample_data, "In-Sample")
        
        # 2. Out-of-Sample Backtesting with optimized parameters
        print("\n📊 Out-of-Sample Backtesting (2023-2024)")
        results['oos_strategy'] = self.run_strategy_backtest(oos_data, "Out-of-Sample", optimal_params)
        results['oos_benchmark'] = self.calculate_buy_and_hold_benchmark(oos_data, "Out-of-Sample")
        
        # 3. Combined Analysis
        print("\n📊 Combined Period Analysis")
        
        # Combine returns for overall assessment
        combined_strategy_returns = pd.concat([
            results['insample_strategy']['returns'],
            results['oos_strategy']['returns']
        ])
        
        combined_benchmark_returns = pd.concat([
            results['insample_benchmark']['returns'],
            results['oos_benchmark']['returns']
        ])
        
        # Calculate combined metrics
        combined_strategy_metrics = self._calculate_combined_metrics(combined_strategy_returns, "Strategy")
        combined_benchmark_metrics = self._calculate_combined_metrics(combined_benchmark_returns, "Benchmark")
        
        results['combined_strategy'] = {
            'returns': combined_strategy_returns,
            'metrics': combined_strategy_metrics
        }
        results['combined_benchmark'] = {
            'returns': combined_benchmark_returns,
            'metrics': combined_benchmark_metrics
        }
        
        # 4. Performance Summary
        self._print_performance_summary(results)
        
        return results
    
    def _calculate_combined_metrics(self, returns: pd.Series, name: str) -> Dict[str, float]:
        """Calculate performance metrics for a returns series"""
        if len(returns) == 0:
            return {}
            
        try:
            # Basic calculations
            total_return = (1 + returns).prod() - 1
            annual_return = total_return / (len(returns) / 252)
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            
            # Drawdown calculation
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdowns.min()
            
            # Additional metrics
            win_rate = (returns > 0).mean()
            profit_factor = returns[returns > 0].sum() / abs(returns[returns < 0].sum()) if (returns < 0).any() else np.inf
            
            # VaR calculations
            var_95 = returns.quantile(0.05)
            cvar_95 = returns[returns <= var_95].mean()
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'best_day': returns.max(),
                'worst_day': returns.min(),
                'positive_days': (returns > 0).sum(),
                'negative_days': (returns < 0).sum(),
                'total_days': len(returns)
            }
            
            print(f"✅ {name} Combined Metrics:")
            print(f"   • Total return: {total_return:.2%}")
            print(f"   • Annual return: {annual_return:.2%}")
            print(f"   • Sharpe ratio: {sharpe_ratio:.3f}")
            print(f"   • Max drawdown: {max_drawdown:.2%}")
            print(f"   • Win rate: {win_rate:.2%}")
            print(f"   • Profit factor: {profit_factor:.2f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error calculating metrics for {name}: {str(e)}")
            return {}
    
    def _print_performance_summary(self, results: Dict[str, Any]):
        """Print comprehensive performance summary"""
        print("\n" + "=" * 80)
        print("📊 BACKTESTING PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Extract metrics
        is_strat = results['insample_strategy']['metrics']
        oos_strat = results['oos_strategy']['metrics']
        is_bench = results['insample_benchmark']['metrics']
        oos_bench = results['oos_benchmark']['metrics']
        combined_strat = results['combined_strategy']['metrics']
        combined_bench = results['combined_benchmark']['metrics']
        
        # Create comparison table
        print("\n📋 PERFORMANCE COMPARISON TABLE")
        print("-" * 80)
        print(f"{'Metric':<20} {'IS_Strat':<12} {'OOS_Strat':<12} {'IS_Bench':<12} {'OOS_Bench':<12}")
        print("-" * 80)
        
        metrics_to_compare = [
            ('Total Return', 'total_return', '.2%'),
            ('Sharpe Ratio', 'sharpe_ratio', '.3f'),
            ('Max Drawdown', 'max_drawdown', '.2%'),
            ('Win Rate', 'win_rate', '.2%'),
            ('Volatility', 'volatility', '.2%')
        ]
        
        for metric_name, metric_key, fmt in metrics_to_compare:
            is_s = f"{is_strat.get(metric_key, 0):{fmt}}"
            oos_s = f"{oos_strat.get(metric_key, 0):{fmt}}"
            is_b = f"{is_bench.get(metric_key, 0):{fmt}}"
            oos_b = f"{oos_bench.get(metric_key, 0):{fmt}}"
            print(f"{metric_name:<20} {is_s:<12} {oos_s:<12} {is_b:<12} {oos_b:<12}")
        
        print("-" * 80)
        
        # Strategy vs Benchmark Analysis
        print("\n🎯 STRATEGY vs BENCHMARK ANALYSIS")
        print("-" * 50)
        
        # In-Sample comparison
        is_outperformance = is_strat.get('total_return', 0) - is_bench.get('total_return', 0)
        oos_outperformance = oos_strat.get('total_return', 0) - oos_bench.get('total_return', 0)
        
        print(f"In-Sample Outperformance: {is_outperformance:.2%}")
        print(f"Out-of-Sample Outperformance: {oos_outperformance:.2%}")
        
        # Risk-adjusted comparison
        is_sharpe_diff = is_strat.get('sharpe_ratio', 0) - is_bench.get('sharpe_ratio', 0)
        oos_sharpe_diff = oos_strat.get('sharpe_ratio', 0) - oos_bench.get('sharpe_ratio', 0)
        
        print(f"In-Sample Sharpe Improvement: {is_sharpe_diff:.3f}")
        print(f"Out-of-Sample Sharpe Improvement: {oos_sharpe_diff:.3f}")
        
        # Consistency analysis
        print(f"\n📈 CONSISTENCY ANALYSIS")
        print("-" * 30)
        consistent_outperformance = (is_outperformance > 0) and (oos_outperformance > 0)
        consistent_risk_adj = (is_sharpe_diff > 0) and (oos_sharpe_diff > 0)
        
        print(f"Consistent Outperformance: {'✅ YES' if consistent_outperformance else '❌ NO'}")
        print(f"Consistent Risk-Adj Performance: {'✅ YES' if consistent_risk_adj else '❌ NO'}")
        
        # Store results for later use
        self.results['backtesting'] = results
        
        print("\n✅ Backtesting phase completed successfully!")
        return results

    def execute_statistical_validation(self, backtesting_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute comprehensive statistical validation using advanced methods
        """
        print("\n🧪 PHASE 3: ADVANCED STATISTICAL VALIDATION")
        print("-" * 50)
        
        statistical_results = {}
        
        # Extract strategy returns for analysis
        insample_returns = backtesting_results['insample_strategy']['returns']
        oos_returns = backtesting_results['oos_strategy']['returns']
        combined_returns = backtesting_results['combined_strategy']['returns']
        
        # Initialize statistical testing framework
        stat_tests = StatisticalTests(ret_series=combined_returns)
        bootstrap = AdvancedBootstrapping(ret_series=combined_returns)
        
        print(f"🔍 Analyzing {len(combined_returns)} return observations...")
        
        # 1. PERMUTATION TESTS
        print("\n🎲 Executing Permutation Tests...")
        statistical_results['permutation_tests'] = self._run_permutation_tests(
            stat_tests, combined_returns, insample_returns, oos_returns
        )
        
        # 2. BOOTSTRAP VALIDATION
        print("\n🔄 Executing Bootstrap Validation...")
        statistical_results['bootstrap_tests'] = self._run_bootstrap_validation(
            bootstrap, combined_returns, insample_returns, oos_returns
        )
        
        # 3. WHITE'S REALITY CHECK & ROMANO-WOLF
        print("\n⚡ Executing Multiple Testing Corrections...")
        statistical_results['multiple_testing'] = self._run_multiple_testing_corrections(
            stat_tests, bootstrap, combined_returns, insample_returns, oos_returns
        )
        
        # 4. DISTRIBUTIONAL TESTS
        print("\n📊 Executing Distributional Tests...")
        statistical_results['distributional_tests'] = self._run_distributional_tests(
            bootstrap, combined_returns, insample_returns, oos_returns
        )
        
        # 5. STATISTICAL EDGE ANALYSIS
        print("\n🎯 Analyzing Statistical Edge...")
        edge_analyzer = StatisticalEdgeAnalyzer()
        
        # Combine all permutation results
        all_permutation_results = statistical_results['permutation_tests']
        
        # Traditional metrics from backtesting
        traditional_metrics = backtesting_results['combined_strategy']['metrics']
        
        # Bootstrap results
        bootstrap_results = statistical_results['bootstrap_tests']
        
        edge_analysis = edge_analyzer.analyze_edge(
            all_permutation_results, bootstrap_results, traditional_metrics
        )
        
        statistical_results['edge_analysis'] = edge_analysis
        
        # 6. PRINT STATISTICAL SUMMARY
        self._print_statistical_summary(statistical_results)
        
        # Store results
        self.results['statistical_validation'] = statistical_results
        
        print("\n✅ Statistical validation phase completed successfully!")
        return statistical_results
    
    def _run_permutation_tests(self, stat_tests: StatisticalTests, 
                              combined_returns: pd.Series,
                              insample_returns: pd.Series,
                              oos_returns: pd.Series) -> Dict[str, Any]:
        """Execute comprehensive permutation tests"""
        
        permutation_results = {}
        n_permutations = 10000
        
        print(f"   • Running {n_permutations} permutations per test...")
        
        # 1. Sign Test on Combined Returns
        print("   • Permutation sign test on combined returns...")
        try:
            sign_test = stat_tests.permutation_test_on_signs(
                n_permutations=n_permutations
            )
            permutation_results['permutation_sign_test'] = sign_test
            print(f"     ✓ P-value: {sign_test['p_value']:.4f}")
        except Exception as e:
            print(f"     ❌ Sign test failed: {str(e)}")
        
        # 2. Sharpe Ratio Permutation Test
        print("   • Permutation test on Sharpe ratio...")
        try:
            def sharpe_calculator(returns):
                if len(returns) == 0 or np.std(returns) == 0:
                    return 0
                return np.mean(returns) / np.std(returns) * np.sqrt(252)
                
            sharpe_test = stat_tests.permutation_test_on_metric(
                metric_func=sharpe_calculator,
                n_permutations=n_permutations
            )
            permutation_results['permutation_sharpe_test'] = sharpe_test
            print(f"     ✓ P-value: {sharpe_test['p_value']:.4f}")
        except Exception as e:
            print(f"     ❌ Sharpe test failed: {str(e)}")
        
        # 3. Profit Factor Permutation Test
        print("   • Permutation test on Profit Factor...")
        try:
            def profit_factor_calculator(returns):
                if len(returns) == 0:
                    return 1.0
                profits = returns[returns > 0]
                losses = returns[returns < 0]
                if len(profits) == 0:
                    return 0.0
                if len(losses) == 0:
                    return np.inf
                return profits.sum() / abs(losses.sum())
                
            pf_test = stat_tests.permutation_test_on_metric(
                metric_func=profit_factor_calculator,
                n_permutations=n_permutations
            )
            permutation_results['permutation_pf_test'] = pf_test
            print(f"     ✓ P-value: {pf_test['p_value']:.4f}")
        except Exception as e:
            print(f"     ❌ Profit Factor test failed: {str(e)}")
        
        # 4. Maximum Drawdown Permutation Test
        print("   • Permutation test on Maximum Drawdown...")
        try:
            def max_drawdown_calculator(returns):
                if len(returns) == 0:
                    return 0
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                return abs(np.min(drawdowns))
                
            dd_test = stat_tests.permutation_test_on_metric(
                metric_func=max_drawdown_calculator,
                n_permutations=n_permutations
            )
            permutation_results['permutation_dd_test'] = dd_test
            print(f"     ✓ P-value: {dd_test['p_value']:.4f}")
        except Exception as e:
            print(f"     ❌ Max Drawdown test failed: {str(e)}")
        
        return permutation_results
    
    def _run_bootstrap_validation(self, bootstrap: AdvancedBootstrapping,
                                 combined_returns: pd.Series,
                                 insample_returns: pd.Series,
                                 oos_returns: pd.Series) -> Dict[str, Any]:
        """Execute comprehensive bootstrap validation"""
        
        bootstrap_results = {}
        n_bootstrap = 5000
        
        print(f"   • Running {n_bootstrap} bootstrap simulations...")
        
        # 1. Stationary Bootstrap on Combined Returns
        print("   • Stationary bootstrap simulation...")
        try:
            bootstrap_sim = bootstrap.run_bootstrap_simulation(
                combined_returns.values,
                n_simulations=n_bootstrap,
                block_size=20
            )
            
            # Calculate statistics for each simulation
            simulated_stats = []
            for sim in bootstrap_sim:
                try:
                    total_return = np.prod(1 + sim) - 1
                    annual_return = total_return / (len(sim) / 252)
                    volatility = np.std(sim) * np.sqrt(252)
                    sharpe = annual_return / volatility if volatility > 0 else 0
                    
                    # Max drawdown
                    cumulative = np.cumprod(1 + sim)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdowns = (cumulative - running_max) / running_max
                    max_dd = np.min(drawdowns)
                    
                    simulated_stats.append({
                        'Total_Return': total_return,
                        'Annual_Return': annual_return,
                        'Volatility': volatility,
                        'Sharpe': sharpe,
                        'Max_Drawdown': max_dd
                    })
                except:
                    continue
            
            bootstrap_results['simulated_stats'] = simulated_stats
            
            # Calculate confidence intervals
            if simulated_stats:
                metrics = ['Total_Return', 'Annual_Return', 'Sharpe', 'Max_Drawdown']
                confidence_intervals = {}
                
                for metric in metrics:
                    values = [s[metric] for s in simulated_stats if not np.isnan(s[metric])]
                    if values:
                        confidence_intervals[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'ci_lower': np.percentile(values, 2.5),
                            'ci_upper': np.percentile(values, 97.5),
                            'p_positive': np.mean(np.array(values) > 0)
                        }
                
                bootstrap_results['confidence_intervals'] = confidence_intervals
                print(f"     ✓ Generated {len(simulated_stats)} valid bootstrap samples")
            
        except Exception as e:
            print(f"     ❌ Bootstrap simulation failed: {str(e)}")
        
        # 2. Block Bootstrap for Robustness
        print("   • Block bootstrap simulation...")
        try:
            block_bootstrap_sim = bootstrap.run_bootstrap_simulation(
                combined_returns.values,
                n_simulations=1000,  # Fewer for computational efficiency
                block_size=30
            )
            
            # Calculate Sharpe ratios for block bootstrap
            block_sharpes = []
            for sim in block_bootstrap_sim:
                try:
                    annual_return = np.mean(sim) * 252
                    volatility = np.std(sim) * np.sqrt(252)
                    sharpe = annual_return / volatility if volatility > 0 else 0
                    block_sharpes.append(sharpe)
                except:
                    continue
            
            bootstrap_results['block_bootstrap_sharpes'] = block_sharpes
            print(f"     ✓ Generated {len(block_sharpes)} block bootstrap Sharpe ratios")
            
        except Exception as e:
            print(f"     ❌ Block bootstrap failed: {str(e)}")
        
        return bootstrap_results
    
    def _run_multiple_testing_corrections(self, stat_tests: StatisticalTests,
                                        bootstrap: AdvancedBootstrapping,
                                        combined_returns: pd.Series,
                                        insample_returns: pd.Series,
                                        oos_returns: pd.Series) -> Dict[str, Any]:
        """Execute multiple testing corrections"""
        
        multiple_testing_results = {}
        
        # Prepare benchmark returns (random walk)
        benchmark_returns = np.random.normal(0, combined_returns.std(), len(combined_returns))
        
        print("   • White's Reality Check...")
        try:
            # Prepare data for White's Reality Check
            # We'll test if our strategy is better than random
            strategy_performance = np.mean(combined_returns.values) * 252  # Annualized return
            
            # Generate bootstrap samples for our strategy
            bootstrap_results = bootstrap.run_bootstrap_simulation(
                combined_returns.values, n_simulations=1000, block_size=20
            )
            
            strategy_bootstrap_perfs = []
            for sim in bootstrap_results:
                annual_ret = np.mean(sim) * 252
                strategy_bootstrap_perfs.append(annual_ret)
            
            # For simplified version, we'll use one strategy vs benchmark
            whites_test = stat_tests.whites_reality_check(
                observed_performances=[strategy_performance],
                bootstrapped_performances=[strategy_bootstrap_perfs],
                benchmark_performance=0.0  # Risk-free rate
            )
            multiple_testing_results['whites_reality_check'] = whites_test
            print(f"     ✓ P-value: {whites_test['p_value']:.4f}")
        except Exception as e:
            print(f"     ❌ White's test failed: {str(e)}")
        
        print("   • Romano-Wolf StepM procedure...")
        try:
            # Create strategy variants for testing
            is_performance = np.mean(insample_returns.values) * 252
            oos_performance = np.mean(oos_returns.values) * 252
            combined_performance = np.mean(combined_returns.values) * 252
            
            # Generate bootstrap samples for each variant
            is_bootstrap = bootstrap.run_bootstrap_simulation(
                insample_returns.values, n_simulations=500, block_size=15
            )
            oos_bootstrap = bootstrap.run_bootstrap_simulation(
                oos_returns.values, n_simulations=500, block_size=15
            )
            combined_bootstrap = bootstrap.run_bootstrap_simulation(
                combined_returns.values, n_simulations=500, block_size=20
            )
            
            is_bootstrap_perfs = [np.mean(sim) * 252 for sim in is_bootstrap]
            oos_bootstrap_perfs = [np.mean(sim) * 252 for sim in oos_bootstrap]
            combined_bootstrap_perfs = [np.mean(sim) * 252 for sim in combined_bootstrap]
            
            rw_test = stat_tests.romano_wolf_stepm(
                observed_performances=[is_performance, oos_performance, combined_performance],
                bootstrapped_performances=[is_bootstrap_perfs, oos_bootstrap_perfs, combined_bootstrap_perfs],
                benchmark_performance=0.0
            )
            multiple_testing_results['romano_wolf'] = rw_test
            print(f"     ✓ Rejected hypotheses: {len(rw_test['rejected_hypotheses_indices'])}")
        except Exception as e:
            print(f"     ❌ Romano-Wolf test failed: {str(e)}")
        
        return multiple_testing_results
    
    def _run_distributional_tests(self, bootstrap: AdvancedBootstrapping,
                                 combined_returns: pd.Series,
                                 insample_returns: pd.Series,
                                 oos_returns: pd.Series) -> Dict[str, Any]:
        """Execute distributional comparison tests"""
        
        distributional_results = {}
        
        print("   • Kolmogorov-Smirnov test (in-sample vs out-of-sample)...")
        try:
            from scipy.stats import ks_2samp
            # Compare in-sample vs out-of-sample distributions
            ks_stat, ks_pvalue = ks_2samp(insample_returns.values, oos_returns.values)
            distributional_results['ks_test'] = {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'interpretation': 'Distributions are significantly different' if ks_pvalue < 0.05 else 'Distributions are similar'
            }
            print(f"     ✓ KS statistic: {ks_stat:.6f}, P-value: {ks_pvalue:.4f}")
        except Exception as e:
            print(f"     ❌ KS test failed: {str(e)}")
        
        print("   • Mann-Whitney U test...")
        try:
            from scipy.stats import mannwhitneyu
            # Non-parametric test for distribution differences
            mw_stat, mw_pvalue = mannwhitneyu(
                insample_returns.values, 
                oos_returns.values, 
                alternative='two-sided'
            )
            distributional_results['mannwhitney_test'] = {
                'statistic': mw_stat,
                'p_value': mw_pvalue,
                'interpretation': 'Distributions have different medians' if mw_pvalue < 0.05 else 'Distributions have similar medians'
            }
            print(f"     ✓ MW statistic: {mw_stat:.6f}, P-value: {mw_pvalue:.4f}")
        except Exception as e:
            print(f"     ❌ Mann-Whitney test failed: {str(e)}")
        except Exception as e:
            print(f"     ❌ MMD test failed: {str(e)}")
        
        return distributional_results
    
    def _print_statistical_summary(self, statistical_results: Dict[str, Any]):
        """Print comprehensive statistical validation summary"""
        print("\n" + "=" * 80)
        print("🧪 STATISTICAL VALIDATION SUMMARY")
        print("=" * 80)
        
        # Permutation Tests Summary
        if 'permutation_tests' in statistical_results:
            print("\n🎲 PERMUTATION TESTS")
            print("-" * 30)
            perm_tests = statistical_results['permutation_tests']
            
            for test_name, test_result in perm_tests.items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    significance = "✅ SIGNIFICANT" if test_result['p_value'] < 0.05 else "❌ NOT SIGNIFICANT"
                    print(f"{test_name}: P-value = {test_result['p_value']:.4f} ({significance})")
        
        # Bootstrap Summary
        if 'bootstrap_tests' in statistical_results:
            print("\n🔄 BOOTSTRAP VALIDATION")
            print("-" * 30)
            bootstrap_tests = statistical_results['bootstrap_tests']
            
            if 'confidence_intervals' in bootstrap_tests:
                ci = bootstrap_tests['confidence_intervals']
                for metric, stats in ci.items():
                    print(f"{metric}:")
                    print(f"  Mean: {stats['mean']:.4f}, CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
                    print(f"  P(Positive): {stats['p_positive']:.2%}")
        
        # Edge Analysis Summary
        if 'edge_analysis' in statistical_results:
            print("\n🎯 STATISTICAL EDGE ANALYSIS")
            print("-" * 30)
            edge = statistical_results['edge_analysis']
            
            print(f"Edge Score: {edge['edge_score']}/{edge['max_score']} ({edge['edge_percentage']:.1f}%)")
            print(f"Assessment: {edge['edge_assessment']}")
            print(f"Recommendation: {edge['recommendation']}")
            
            print("\nDetailed Indicators:")
            for indicator, value in edge['indicators'].items():
                status = "✅" if value else "❌"
                print(f"  {indicator}: {status}")
        
        print("\n" + "=" * 80)
        
    def generate_comprehensive_visualizations(self, backtesting_results: Dict[str, Any], 
                                           statistical_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate comprehensive visualizations for all analyses
        """
        print("\n📊 PHASE 4: EXTENSIVE VISUALIZATIONS")
        print("-" * 50)
        
        plot_files = {}
        
        # Set up matplotlib parameters
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })
        
        # 1. EQUITY CURVES COMPARISON
        print("   • Generating equity curves comparison...")
        try:
            plot_files['equity_curves'] = self._plot_equity_curves(backtesting_results)
            print("     ✓ Equity curves saved")
        except Exception as e:
            print(f"     ❌ Equity curves failed: {str(e)}")
        
        # 2. RETURNS ANALYSIS
        print("   • Generating returns analysis...")
        try:
            plot_files['returns_analysis'] = self._plot_returns_analysis(backtesting_results)
            print("     ✓ Returns analysis saved")
        except Exception as e:
            print(f"     ❌ Returns analysis failed: {str(e)}")
        
        # 3. STATISTICAL VALIDATION PLOTS
        print("   • Generating statistical validation plots...")
        try:
            plot_files['statistical_validation'] = self._plot_statistical_validation(statistical_results)
            print("     ✓ Statistical validation plots saved")
        except Exception as e:
            print(f"     ❌ Statistical validation plots failed: {str(e)}")
        
        # 4. BOOTSTRAP DISTRIBUTIONS
        print("   • Generating bootstrap distribution plots...")
        try:
            plot_files['bootstrap_distributions'] = self._plot_bootstrap_distributions(statistical_results)
            print("     ✓ Bootstrap distributions saved")
        except Exception as e:
            print(f"     ❌ Bootstrap distributions failed: {str(e)}")
        
        # 5. RISK ANALYSIS PLOTS
        print("   • Generating risk analysis plots...")
        try:
            plot_files['risk_analysis'] = self._plot_risk_analysis(backtesting_results)
            print("     ✓ Risk analysis plots saved")
        except Exception as e:
            print(f"     ❌ Risk analysis plots failed: {str(e)}")
        
        # 6. PERFORMANCE HEATMAP
        print("   • Generating performance heatmap...")
        try:
            plot_files['performance_heatmap'] = self._plot_performance_heatmap(backtesting_results, statistical_results)
            print("     ✓ Performance heatmap saved")
        except Exception as e:
            print(f"     ❌ Performance heatmap failed: {str(e)}")
        
        # Store plot files
        self.results['plot_files'] = plot_files
        
        print(f"\n✅ Generated {len(plot_files)} visualization sets!")
        return plot_files
    
    def _plot_equity_curves(self, backtesting_results: Dict[str, Any]) -> str:
        """Generate comprehensive equity curves comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Equity Curves Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        is_strat_returns = backtesting_results['insample_strategy']['returns']
        oos_strat_returns = backtesting_results['oos_strategy']['returns']
        is_bench_returns = backtesting_results['insample_benchmark']['returns']
        oos_bench_returns = backtesting_results['oos_benchmark']['returns']
        
        # Calculate cumulative returns
        is_strat_cum = (1 + is_strat_returns).cumprod()
        oos_strat_cum = (1 + oos_strat_returns).cumprod()
        is_bench_cum = (1 + is_bench_returns).cumprod()
        oos_bench_cum = (1 + oos_bench_returns).cumprod()
        
        # 1. In-Sample Comparison (Top Left)
        axes[0, 0].plot(is_strat_cum.index, is_strat_cum.values, label='Strategy', linewidth=2, color='blue')
        axes[0, 0].plot(is_bench_cum.index, is_bench_cum.values, label='Buy & Hold', linewidth=2, color='red')
        axes[0, 0].set_title('In-Sample Period (2017-2022)', fontweight='bold')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Out-of-Sample Comparison (Top Right)
        axes[0, 1].plot(oos_strat_cum.index, oos_strat_cum.values, label='Strategy', linewidth=2, color='blue')
        axes[0, 1].plot(oos_bench_cum.index, oos_bench_cum.values, label='Buy & Hold', linewidth=2, color='red')
        axes[0, 1].set_title('Out-of-Sample Period (2023-2024)', fontweight='bold')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Combined Period (Bottom Left)
        combined_strat_returns = pd.concat([is_strat_returns, oos_strat_returns])
        combined_bench_returns = pd.concat([is_bench_returns, oos_bench_returns])
        combined_strat_cum = (1 + combined_strat_returns).cumprod()
        combined_bench_cum = (1 + combined_bench_returns).cumprod()
        
        axes[1, 0].plot(combined_strat_cum.index, combined_strat_cum.values, label='Strategy', linewidth=2, color='blue')
        axes[1, 0].plot(combined_bench_cum.index, combined_bench_cum.values, label='Buy & Hold', linewidth=2, color='red')
        axes[1, 0].axvline(x=pd.Timestamp('2023-01-01'), color='green', linestyle='--', alpha=0.7, label='OOS Start')
        axes[1, 0].set_title('Combined Period (2017-2024)', fontweight='bold')
        axes[1, 0].set_ylabel('Cumulative Return')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Outperformance (Bottom Right)
        is_outperf = is_strat_cum / is_bench_cum - 1
        oos_outperf = oos_strat_cum / oos_bench_cum - 1
        combined_outperf = combined_strat_cum / combined_bench_cum - 1
        
        axes[1, 1].plot(combined_outperf.index, combined_outperf.values * 100, linewidth=2, color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].axvline(x=pd.Timestamp('2023-01-01'), color='green', linestyle='--', alpha=0.7, label='OOS Start')
        axes[1, 1].set_title('Strategy Outperformance vs Buy & Hold', fontweight='bold')
        axes[1, 1].set_ylabel('Outperformance (%)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.plots_dir / "equity_curves_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_returns_analysis(self, backtesting_results: Dict[str, Any]) -> str:
        """Generate comprehensive returns analysis plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Returns Analysis', fontsize=16, fontweight='bold')
        
        # Extract returns
        combined_returns = backtesting_results['combined_strategy']['returns']
        is_returns = backtesting_results['insample_strategy']['returns']
        oos_returns = backtesting_results['oos_strategy']['returns']
        
        # 1. Returns Distribution (Top Left)
        axes[0, 0].hist(combined_returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(x=combined_returns.mean() * 100, color='red', linestyle='--', label='Mean')
        axes[0, 0].set_title('Returns Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Daily Return (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. QQ Plot (Top Middle)
        from scipy import stats
        stats.probplot(combined_returns, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rolling Volatility (Top Right)
        rolling_vol = combined_returns.rolling(window=30).std() * np.sqrt(252) * 100
        axes[0, 2].plot(rolling_vol.index, rolling_vol.values, linewidth=1, color='orange')
        axes[0, 2].axvline(x=pd.Timestamp('2023-01-01'), color='green', linestyle='--', alpha=0.7, label='OOS Start')
        axes[0, 2].set_title('Rolling 30-Day Volatility', fontweight='bold')
        axes[0, 2].set_ylabel('Annualized Volatility (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Monthly Returns Heatmap (Bottom Left)
        monthly_returns = combined_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_matrix = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()
        
        if not monthly_returns_matrix.empty:
            im = axes[1, 0].imshow(monthly_returns_matrix.values * 100, cmap='RdYlGn', aspect='auto')
            axes[1, 0].set_title('Monthly Returns Heatmap (%)', fontweight='bold')
            axes[1, 0].set_ylabel('Year')
            axes[1, 0].set_xlabel('Month')
            if len(monthly_returns_matrix.index) > 0:
                axes[1, 0].set_yticks(range(len(monthly_returns_matrix.index)))
                axes[1, 0].set_yticklabels(monthly_returns_matrix.index)
            axes[1, 0].set_xticks(range(12))
            axes[1, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Autocorrelation (Bottom Middle)
        from statsmodels.tsa.stattools import acf
        lags = range(1, min(50, len(combined_returns)//4))
        autocorr = acf(combined_returns, nlags=max(lags), fft=True)[1:]
        axes[1, 1].bar(lags, autocorr[:len(lags)], alpha=0.7, color='purple')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('Returns Autocorrelation', fontweight='bold')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. In-Sample vs Out-of-Sample Comparison (Bottom Right)
        comparison_data = {
            'In-Sample': [is_returns.mean() * 252 * 100, is_returns.std() * np.sqrt(252) * 100],
            'Out-of-Sample': [oos_returns.mean() * 252 * 100, oos_returns.std() * np.sqrt(252) * 100]
        }
        
        x = np.arange(2)
        width = 0.35
        
        axes[1, 2].bar(x - width/2, comparison_data['In-Sample'], width, label='In-Sample', alpha=0.7)
        axes[1, 2].bar(x + width/2, comparison_data['Out-of-Sample'], width, label='Out-of-Sample', alpha=0.7)
        axes[1, 2].set_title('IS vs OOS Comparison', fontweight='bold')
        axes[1, 2].set_ylabel('Percentage (%)')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(['Annual Return', 'Volatility'])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.plots_dir / "returns_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_statistical_validation(self, statistical_results: Dict[str, Any]) -> str:
        """Generate statistical validation visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Statistical Validation Results', fontsize=16, fontweight='bold')
        
        # 1. P-Values Summary (Top Left)
        if 'permutation_tests' in statistical_results:
            perm_tests = statistical_results['permutation_tests']
            test_names = []
            p_values = []
            
            for test_name, test_result in perm_tests.items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    test_names.append(test_name.replace('permutation_', '').replace('_test', ''))
                    p_values.append(test_result['p_value'])
            
            if test_names and p_values:
                colors = ['green' if p < 0.05 else 'red' for p in p_values]
                bars = axes[0, 0].bar(test_names, p_values, color=colors, alpha=0.7)
                axes[0, 0].axhline(y=0.05, color='black', linestyle='--', label='α = 0.05')
                axes[0, 0].set_title('Permutation Test P-Values', fontweight='bold')
                axes[0, 0].set_ylabel('P-Value')
                axes[0, 0].set_ylim(0, max(0.1, max(p_values) * 1.1))
                axes[0, 0].legend()
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Add p-value labels on bars
                for bar, p_val in zip(bars, p_values):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                   f'{p_val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Bootstrap Confidence Intervals (Top Right)
        if 'bootstrap_tests' in statistical_results and 'confidence_intervals' in statistical_results['bootstrap_tests']:
            ci_data = statistical_results['bootstrap_tests']['confidence_intervals']
            metrics = list(ci_data.keys())
            means = [ci_data[m]['mean'] for m in metrics]
            ci_lower = [ci_data[m]['ci_lower'] for m in metrics]
            ci_upper = [ci_data[m]['ci_upper'] for m in metrics]
            
            x_pos = np.arange(len(metrics))
            axes[0, 1].errorbar(x_pos, means, yerr=[np.array(means) - np.array(ci_lower), 
                                                   np.array(ci_upper) - np.array(means)], 
                               fmt='o', capsize=5, capthick=2, linewidth=2)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0, 1].set_title('Bootstrap 95% Confidence Intervals', fontweight='bold')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels([m.replace('_', ' ') for m in metrics], rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Edge Analysis Radar Chart (Bottom Left)
        if 'edge_analysis' in statistical_results:
            edge_data = statistical_results['edge_analysis']['indicators']
            categories = list(edge_data.keys())
            values = [1 if edge_data[cat] else 0 for cat in categories]
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]
            
            axes[1, 0].plot(angles, values, 'o-', linewidth=2, color='blue')
            axes[1, 0].fill(angles, values, alpha=0.25, color='blue')
            axes[1, 0].set_xticks(angles[:-1])
            axes[1, 0].set_xticklabels([cat.replace('_', '\n') for cat in categories], fontsize=8)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_yticks([0, 0.5, 1])
            axes[1, 0].set_yticklabels(['Fail', 'Partial', 'Pass'])
            axes[1, 0].set_title('Statistical Edge Indicators', fontweight='bold')
            axes[1, 0].grid(True)
        
        # 4. Overall Assessment (Bottom Right)
        if 'edge_analysis' in statistical_results:
            edge_analysis = statistical_results['edge_analysis']
            
            # Create a summary text plot
            axes[1, 1].text(0.5, 0.8, f"EDGE ASSESSMENT", ha='center', va='center', 
                           fontsize=14, fontweight='bold', transform=axes[1, 1].transAxes)
            
            axes[1, 1].text(0.5, 0.6, f"{edge_analysis['edge_assessment']}", ha='center', va='center',
                           fontsize=12, fontweight='bold', color='blue', transform=axes[1, 1].transAxes)
            
            axes[1, 1].text(0.5, 0.4, f"Score: {edge_analysis['edge_score']}/{edge_analysis['max_score']} ({edge_analysis['edge_percentage']:.1f}%)", 
                           ha='center', va='center', fontsize=11, transform=axes[1, 1].transAxes)
            
            axes[1, 1].text(0.5, 0.2, f"Recommendation:", ha='center', va='center',
                           fontsize=10, fontweight='bold', transform=axes[1, 1].transAxes)
            
            axes[1, 1].text(0.5, 0.1, f"{edge_analysis['recommendation']}", ha='center', va='center',
                           fontsize=9, wrap=True, transform=axes[1, 1].transAxes)
            
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        filename = self.plots_dir / "statistical_validation.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_bootstrap_distributions(self, statistical_results: Dict[str, Any]) -> str:
        """Generate bootstrap distribution plots"""
        
        if 'bootstrap_tests' not in statistical_results or 'simulated_stats' not in statistical_results['bootstrap_tests']:
            # Create empty plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.text(0.5, 0.5, 'No Bootstrap Data Available', ha='center', va='center', 
                   fontsize=16, transform=ax.transAxes)
            ax.set_title('Bootstrap Distributions')
            filename = self.plots_dir / "bootstrap_distributions.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            return str(filename)
        
        simulated_stats = statistical_results['bootstrap_tests']['simulated_stats']
        
        # Extract metrics
        metrics = ['Sharpe', 'Total_Return', 'Max_Drawdown']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Bootstrap Distributions', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            values = [s[metric] for s in simulated_stats if not np.isnan(s[metric])]
            
            if values:
                axes[i].hist(values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].axvline(x=np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
                axes[i].axvline(x=np.percentile(values, 2.5), color='orange', linestyle=':', label='95% CI')
                axes[i].axvline(x=np.percentile(values, 97.5), color='orange', linestyle=':')
                axes[i].set_title(f'{metric} Distribution', fontweight='bold')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = self.plots_dir / "bootstrap_distributions.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_risk_analysis(self, backtesting_results: Dict[str, Any]) -> str:
        """Generate risk analysis plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
        
        combined_returns = backtesting_results['combined_strategy']['returns']
        
        # 1. Drawdown Analysis (Top Left)
        cumulative_returns = (1 + combined_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        axes[0, 0].fill_between(drawdowns.index, drawdowns.values * 100, 0, alpha=0.7, color='red')
        axes[0, 0].set_title('Drawdown Analysis', fontweight='bold')
        axes[0, 0].set_ylabel('Drawdown (%)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. VaR Analysis (Top Right)
        var_levels = [0.01, 0.05, 0.10]
        var_values = [combined_returns.quantile(level) * 100 for level in var_levels]
        
        axes[0, 1].bar(range(len(var_levels)), var_values, alpha=0.7, color='orange')
        axes[0, 1].set_title('Value at Risk (VaR)', fontweight='bold')
        axes[0, 1].set_ylabel('VaR (%)')
        axes[0, 1].set_xticks(range(len(var_levels)))
        axes[0, 1].set_xticklabels([f'{level*100:.0f}%' for level in var_levels])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio (Bottom Left)
        rolling_sharpe = combined_returns.rolling(window=252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        
        axes[1, 0].plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='green')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 0].axhline(y=1, color='blue', linestyle='--', alpha=0.7, label='Sharpe = 1')
        axes[1, 0].set_title('Rolling 252-Day Sharpe Ratio', fontweight='bold')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Risk-Return Scatter (Bottom Right)
        # Calculate rolling metrics for scatter plot
        window = 63  # Quarterly windows
        rolling_returns = combined_returns.rolling(window).mean() * 252 * 100
        rolling_vol = combined_returns.rolling(window).std() * np.sqrt(252) * 100
        
        valid_data = ~(rolling_returns.isna() | rolling_vol.isna())
        if valid_data.any():
            scatter = axes[1, 1].scatter(rolling_vol[valid_data], rolling_returns[valid_data], 
                                       alpha=0.6, c=range(len(rolling_vol[valid_data])), cmap='viridis')
            axes[1, 1].set_title('Risk-Return Profile (Rolling Quarterly)', fontweight='bold')
            axes[1, 1].set_xlabel('Volatility (%)')
            axes[1, 1].set_ylabel('Annualized Return (%)')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Time Period')
        
        plt.tight_layout()
        filename = self.plots_dir / "risk_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def _plot_performance_heatmap(self, backtesting_results: Dict[str, Any], 
                                statistical_results: Dict[str, Any]) -> str:
        """Generate performance heatmap summary"""
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle('Performance Summary Heatmap', fontsize=16, fontweight='bold')
        
        # Prepare data for heatmap
        metrics_data = []
        
        # Extract metrics from different sources
        combined_metrics = backtesting_results['combined_strategy']['metrics']
        is_metrics = backtesting_results['insample_strategy']['metrics']
        oos_metrics = backtesting_results['oos_strategy']['metrics']
        
        # Define metrics to include
        metric_names = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'volatility']
        periods = ['Combined', 'In-Sample', 'Out-of-Sample']
        
        # Create data matrix
        for period in periods:
            if period == 'Combined':
                source_metrics = combined_metrics
            elif period == 'In-Sample':
                source_metrics = is_metrics
            else:
                source_metrics = oos_metrics
            
            row = []
            for metric in metric_names:
                value = source_metrics.get(metric, 0)
                # Normalize values for heatmap
                if metric == 'total_return':
                    normalized_value = value * 100  # Convert to percentage
                elif metric == 'sharpe_ratio':
                    normalized_value = value
                elif metric == 'max_drawdown':
                    normalized_value = abs(value) * 100  # Make positive and percentage
                elif metric == 'win_rate':
                    normalized_value = value * 100  # Convert to percentage
                elif metric == 'volatility':
                    normalized_value = value * 100  # Convert to percentage
                else:
                    normalized_value = value
                    
                row.append(normalized_value)
            metrics_data.append(row)
        
        # Create heatmap
        im = ax.imshow(metrics_data, cmap='RdYlGn', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(metric_names)))
        ax.set_yticks(range(len(periods)))
        ax.set_xticklabels([name.replace('_', ' ').title() for name in metric_names])
        ax.set_yticklabels(periods)
        
        # Add text annotations
        for i in range(len(periods)):
            for j in range(len(metric_names)):
                text = f'{metrics_data[i][j]:.2f}'
                ax.text(j, i, text, ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Performance Value')
        
        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.tight_layout()
        filename = self.plots_dir / "performance_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(filename)
    
    def generate_comprehensive_report(self, backtesting_results: Dict[str, Any], 
                                    statistical_results: Dict[str, Any],
                                    plot_files: Dict[str, str]) -> str:
        """
        Generate comprehensive final report
        """
        print("\n📋 PHASE 5: RESULTS CONSOLIDATION AND REPORTING")
        print("-" * 50)
        
        report_content = self._create_report_content(backtesting_results, statistical_results, plot_files)
        
        # Save report
        report_filename = self.reports_dir / f"comprehensive_backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_filename, 'w') as f:
            f.write(report_content)
        
        # Also save as JSON for programmatic access
        json_filename = self.reports_dir / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_summary = {
            'backtesting_results': self._serialize_results(backtesting_results),
            'statistical_results': self._serialize_results(statistical_results),
            'plot_files': plot_files,
            'timestamp': datetime.now().isoformat(),
            'summary': self._create_executive_summary(backtesting_results, statistical_results)
        }
        
        import json
        with open(json_filename, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print(f"✅ Comprehensive report saved: {report_filename}")
        print(f"✅ JSON results saved: {json_filename}")
        
        return str(report_filename)
    
    def _create_report_content(self, backtesting_results: Dict[str, Any], 
                              statistical_results: Dict[str, Any],
                              plot_files: Dict[str, str]) -> str:
        """Create comprehensive markdown report"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# Comprehensive Statistical Arbitrage Backtesting Report

**Generated:** {timestamp}
**Strategy:** Statistical Arbitrage (BTC-ETH Pairs)
**Analysis Period:** 2017-2024 (In-Sample: 2017-2022, Out-of-Sample: 2023-2024)

---

## Executive Summary

{self._create_executive_summary(backtesting_results, statistical_results)}

---

## 1. Backtesting Performance

### 1.1 Overall Performance Metrics

"""
        
        # Add performance table
        combined_metrics = backtesting_results['combined_strategy']['metrics']
        is_metrics = backtesting_results['insample_strategy']['metrics']
        oos_metrics = backtesting_results['oos_strategy']['metrics']
        
        report += """
| Metric | Combined | In-Sample | Out-of-Sample |
|--------|----------|-----------|---------------|
"""
        
        metrics_to_report = [
            ('Total Return', 'total_return', '.2%'),
            ('Annual Return', 'annual_return', '.2%'),
            ('Sharpe Ratio', 'sharpe_ratio', '.3f'),
            ('Max Drawdown', 'max_drawdown', '.2%'),
            ('Win Rate', 'win_rate', '.2%'),
            ('Volatility', 'volatility', '.2%'),
            ('Profit Factor', 'profit_factor', '.2f')
        ]
        
        for metric_name, metric_key, fmt in metrics_to_report:
            combined_val = f"{combined_metrics.get(metric_key, 0):{fmt}}"
            is_val = f"{is_metrics.get(metric_key, 0):{fmt}}"
            oos_val = f"{oos_metrics.get(metric_key, 0):{fmt}}"
            report += f"| {metric_name} | {combined_val} | {is_val} | {oos_val} |\n"
        
        report += f"""
### 1.2 Risk Metrics

| Risk Metric | Value |
|-------------|-------|
| Value at Risk (5%) | {combined_metrics.get('var_95', 0):.2%} |
| Conditional VaR (5%) | {combined_metrics.get('cvar_95', 0):.2%} |
| Skewness | {combined_metrics.get('skewness', 0):.3f} |
| Kurtosis | {combined_metrics.get('kurtosis', 0):.3f} |
| Best Day | {combined_metrics.get('best_day', 0):.2%} |
| Worst Day | {combined_metrics.get('worst_day', 0):.2%} |

---

## 2. Statistical Validation Results

### 2.1 Permutation Tests
"""
        
        # Add permutation test results
        if 'permutation_tests' in statistical_results:
            perm_tests = statistical_results['permutation_tests']
            for test_name, test_result in perm_tests.items():
                if isinstance(test_result, dict) and 'p_value' in test_result:
                    significance = "✅ SIGNIFICANT" if test_result['p_value'] < 0.05 else "❌ NOT SIGNIFICANT"
                    report += f"- **{test_name.replace('_', ' ').title()}**: P-value = {test_result['p_value']:.4f} ({significance})\n"
        
        report += "\n### 2.2 Bootstrap Validation\n"
        
        # Add bootstrap results
        if 'bootstrap_tests' in statistical_results and 'confidence_intervals' in statistical_results['bootstrap_tests']:
            ci_data = statistical_results['bootstrap_tests']['confidence_intervals']
            for metric, stats in ci_data.items():
                report += f"- **{metric.replace('_', ' ').title()}**: {stats['mean']:.4f} (95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])\n"
                report += f"  - Probability of Positive: {stats['p_positive']:.2%}\n"
        
        report += "\n### 2.3 Multiple Testing Corrections\n"
        
        # Add multiple testing results
        if 'multiple_testing' in statistical_results:
            mt_results = statistical_results['multiple_testing']
            if 'whites_reality_check' in mt_results:
                whites = mt_results['whites_reality_check']
                significance = "✅ SIGNIFICANT" if whites['p_value'] < 0.05 else "❌ NOT SIGNIFICANT"
                report += f"- **White's Reality Check**: P-value = {whites['p_value']:.4f} ({significance})\n"
            
            if 'romano_wolf' in mt_results:
                rw = mt_results['romano_wolf']
                report += f"- **Romano-Wolf StepM**: {len(rw['significant_strategies'])} significant strategies\n"
        
        report += "\n### 2.4 Distributional Tests\n"
        
        # Add distributional test results
        if 'distributional_tests' in statistical_results:
            dist_tests = statistical_results['distributional_tests']
            if 'wasserstein_test' in dist_tests:
                wass = dist_tests['wasserstein_test']
                report += f"- **Wasserstein Distance**: {wass['distance']:.6f} (P-value: {wass['p_value']:.4f})\n"
            
            if 'mmd_test' in dist_tests:
                mmd = dist_tests['mmd_test']
                report += f"- **Maximum Mean Discrepancy**: {mmd['mmd']:.6f} (P-value: {mmd['p_value']:.4f})\n"
        
        report += f"""
---

## 3. Statistical Edge Analysis

{self._format_edge_analysis(statistical_results.get('edge_analysis', {}))}

---

## 4. Visualizations

The following plots have been generated and saved:

"""
        
        # Add plot file references
        for plot_name, plot_path in plot_files.items():
            report += f"- **{plot_name.replace('_', ' ').title()}**: `{plot_path}`\n"
        
        report += f"""
---

## 5. Conclusions and Recommendations

{self._create_conclusions(backtesting_results, statistical_results)}

---

## 6. Technical Details

- **Data Source**: BTC_USD and ETH_USD daily data
- **Strategy Type**: Statistical Arbitrage (Mean Reversion)
- **Bootstrap Simulations**: 5,000 iterations
- **Permutation Tests**: 10,000 iterations
- **Significance Level**: α = 0.05
- **Analysis Framework**: Advanced Statistical Validation Pipeline

---

*Report generated by Saucedo Quantitative Trading Engine*
*Comprehensive Statistical Arbitrage Backtesting Pipeline*
"""
        
        return report
    
    def _create_executive_summary(self, backtesting_results: Dict[str, Any], 
                                 statistical_results: Dict[str, Any]) -> str:
        """Create executive summary"""
        
        combined_metrics = backtesting_results['combined_strategy']['metrics']
        edge_analysis = statistical_results.get('edge_analysis', {})
        
        total_return = combined_metrics.get('total_return', 0)
        sharpe_ratio = combined_metrics.get('sharpe_ratio', 0)
        max_drawdown = combined_metrics.get('max_drawdown', 0)
        
        edge_assessment = edge_analysis.get('edge_assessment', 'UNKNOWN')
        edge_percentage = edge_analysis.get('edge_percentage', 0)
        recommendation = edge_analysis.get('recommendation', 'Unable to determine')
        
        summary = f"""
The Statistical Arbitrage strategy applied to BTC-ETH pairs from 2017-2024 achieved a total return of {total_return:.2%} 
with a Sharpe ratio of {sharpe_ratio:.3f} and maximum drawdown of {max_drawdown:.2%}.

**Statistical Edge Assessment:** {edge_assessment} ({edge_percentage:.1f}% confidence)

**Key Findings:**
- The strategy demonstrates {'consistent' if total_return > 0 and sharpe_ratio > 0 else 'inconsistent'} performance across the test period
- Statistical validation {'supports' if edge_percentage >= 60 else 'does not support'} the presence of a significant edge
- Risk-adjusted returns are {'favorable' if sharpe_ratio > 1 else 'unfavorable'} compared to typical trading strategies

**Recommendation:** {recommendation}
"""
        
        return summary.strip()
    
    def _format_edge_analysis(self, edge_analysis: Dict[str, Any]) -> str:
        """Format edge analysis section"""
        
        if not edge_analysis:
            return "Edge analysis not available."
        
        content = f"""
### Overall Assessment: {edge_analysis.get('edge_assessment', 'Unknown')}

**Edge Score:** {edge_analysis.get('edge_score', 0)}/{edge_analysis.get('max_score', 10)} ({edge_analysis.get('edge_percentage', 0):.1f}%)

**Recommendation:** {edge_analysis.get('recommendation', 'No recommendation available')}

### Detailed Edge Indicators:

"""
        
        indicators = edge_analysis.get('indicators', {})
        for indicator, passed in indicators.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            content += f"- **{indicator.replace('_', ' ').title()}**: {status}\n"
        
        return content
    
    def _create_conclusions(self, backtesting_results: Dict[str, Any], 
                           statistical_results: Dict[str, Any]) -> str:
        """Create conclusions section"""
        
        combined_metrics = backtesting_results['combined_strategy']['metrics']
        edge_analysis = statistical_results.get('edge_analysis', {})
        
        total_return = combined_metrics.get('total_return', 0)
        sharpe_ratio = combined_metrics.get('sharpe_ratio', 0)
        edge_percentage = edge_analysis.get('edge_percentage', 0)
        
        conclusions = f"""
### Performance Assessment

1. **Return Generation**: The strategy {'successfully generated positive returns' if total_return > 0 else 'failed to generate positive returns'} over the full test period.

2. **Risk-Adjusted Performance**: With a Sharpe ratio of {sharpe_ratio:.3f}, the strategy {'demonstrates strong' if sharpe_ratio > 1 else 'shows weak'} risk-adjusted performance.

3. **Statistical Significance**: The comprehensive statistical validation shows {edge_percentage:.1f}% confidence in the strategy's edge, indicating {'strong' if edge_percentage >= 80 else 'moderate' if edge_percentage >= 60 else 'weak'} statistical support.

### Risk Considerations

- Maximum drawdown of {combined_metrics.get('max_drawdown', 0):.2%} suggests {'acceptable' if abs(combined_metrics.get('max_drawdown', 0)) < 0.2 else 'elevated'} risk levels
- Volatility of {combined_metrics.get('volatility', 0):.2%} indicates {'moderate' if combined_metrics.get('volatility', 0) < 0.3 else 'high'} return variability

### Final Recommendation

Based on the comprehensive analysis, this strategy is {'RECOMMENDED for deployment' if edge_percentage >= 70 and total_return > 0 else 'NOT RECOMMENDED for deployment without further optimization'}.
"""
        
        return conclusions.strip()
    
    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize results for JSON storage"""
        
        def convert_value(obj):
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                # Convert to dict and ensure keys are strings
                data_dict = obj.to_dict()
                return convert_dict_keys_to_strings(data_dict)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return convert_dict_keys_to_strings({k: convert_value(v) for k, v in obj.items()})
            elif isinstance(obj, list):
                return [convert_value(item) for item in obj]
            elif isinstance(obj, pd.Timestamp):
                return str(obj)
            else:
                return obj
        
        def convert_dict_keys_to_strings(d):
            """Convert dictionary keys to strings if they're Timestamps or other non-JSON types"""
            if not isinstance(d, dict):
                return d
            
            new_dict = {}
            for k, v in d.items():
                # Convert key to string if it's a Timestamp or other non-JSON type
                if isinstance(k, pd.Timestamp):
                    new_key = str(k)
                elif isinstance(k, (np.integer, np.floating)):
                    new_key = str(k)
                else:
                    new_key = str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k
                
                # Recursively convert nested dictionaries
                if isinstance(v, dict):
                    new_dict[new_key] = convert_dict_keys_to_strings(v)
                else:
                    new_dict[new_key] = v
            
            return new_dict
        
        return convert_value(results)
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Execute the complete backtesting and analysis pipeline
        """
        print("\n🚀 EXECUTING COMPLETE COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        
        try:
            # PHASE 1: Data Loading
            insample_data, oos_data = self.load_and_prepare_data()
            
            # PHASE 2: Backtesting
            backtesting_results = self.execute_backtesting_phase(insample_data, oos_data)
            
            # PHASE 3: Statistical Validation
            statistical_results = self.execute_statistical_validation(backtesting_results)
            
            # PHASE 4: Visualizations
            plot_files = self.generate_comprehensive_visualizations(backtesting_results, statistical_results)
            
            # PHASE 5: Reporting
            report_file = self.generate_comprehensive_report(backtesting_results, statistical_results, plot_files)
            
            # Store all results
            self.results['complete_analysis'] = {
                'backtesting_results': backtesting_results,
                'statistical_results': statistical_results,
                'plot_files': plot_files,
                'report_file': report_file,
                'execution_timestamp': datetime.now().isoformat()
            }
            
            print("\n" + "=" * 80)
            print("🎉 COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📊 Report saved: {report_file}")
            print(f"📈 Plots generated: {len(plot_files)}")
            print(f"📋 Report saved: {report_file}")
            print(f"🧪 Statistical validation: Complete")
            print("=" * 80)
            
            # Display final edge assessment
            edge_analysis = statistical_results.get('edge_analysis', {})
            if edge_analysis:
                print(f"\n🎯 FINAL EDGE ASSESSMENT")
                print(f"Assessment: {edge_analysis.get('edge_assessment', 'Unknown')}")
                print(f"Score: {edge_analysis.get('edge_score', 0)}/{edge_analysis.get('max_score', 10)} ({edge_analysis.get('edge_percentage', 0):.1f}%)")
                print(f"Recommendation: {edge_analysis.get('recommendation', 'No recommendation')}")
            
            return self.results['complete_analysis']
        
        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            exit(1)

# MAIN EXECUTION
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Comprehensive Statistical Arbitrage Backtesting Pipeline')
    parser.add_argument('--strategy', type=str, default='composite', 
                        choices=['composite', 'entropy'], 
                        help='Strategy type to test (composite or entropy)')
    args = parser.parse_args()
    
    print("🚀 Starting Comprehensive Statistical Arbitrage Backtesting Pipeline")
    print("=" * 80)
    print(f"🎯 Strategy Selected: {args.strategy.upper()}")
    print("=" * 80)
    
    try:
        # Initialize the comprehensive backtester with strategy selection
        backtester = ComprehensiveBacktester(strategy_type=args.strategy)
        
        # Execute the complete analysis pipeline
        results = backtester.run_complete_analysis()
        
        print("\n🎯 PIPELINE EXECUTION SUMMARY")
        print("=" * 50)
        print("✅ All phases completed successfully!")
        print(f"📊 Strategy: {args.strategy.upper()}")
        print(f"📊 Analysis period: 2017-2024")
        print(f"📈 Plots generated: {len(results['plot_files'])}")
        print(f"📋 Report saved: {results['report_file']}")
        print(f"🧪 Statistical validation: Complete")
        print("=" * 50)
        
        # Display final edge assessment
        edge_analysis = results['statistical_results'].get('edge_analysis', {})
        if edge_analysis:
            print(f"\n🎯 FINAL EDGE ASSESSMENT")
            print(f"Assessment: {edge_analysis.get('edge_assessment', 'Unknown')}")
            print(f"Score: {edge_analysis.get('edge_score', 0)}/{edge_analysis.get('max_score', 10)} ({edge_analysis.get('edge_percentage', 0):.1f}%)")
            print(f"Recommendation: {edge_analysis.get('recommendation', 'No recommendation')}")
        
        print("\n✨ Analysis pipeline completed successfully! ✨")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)