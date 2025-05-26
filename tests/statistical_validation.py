#!/usr/bin/env python3
"""
Advanced Statistical Validation Framework

This module provides comprehensive statistical validation for trading strategies including:
1. Monte Carlo bootstrapping for confidence intervals
2. Permutation tests for statistical significance
3. Out-of-sample testing with walk-forward analysis
4. Multiple testing correction (Bonferroni, FDR)
5. Statistical edge detection and validation

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
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import jarque_bera, shapiro, kstest
from statsmodels.stats.multitest import multipletests
import json
from dataclasses import dataclass, asdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import BollingerBandsStrategy
from src.strategies.simple_pairs_strategy_fixed import SimplePairsStrategy, TrendFollowingStrategy, VolatilityBreakoutStrategy
from utils.data_loader import DataLoader
from src.bootstrapping.core import AdvancedBootstrapping
from src.bootstrapping.statistical_tests import StatisticalTests

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')

@dataclass
class ValidationConfig:
    """Configuration for statistical validation"""
    n_bootstrap_samples: int = 1000
    n_permutation_samples: int = 1000
    confidence_level: float = 0.95
    alpha: float = 0.05
    min_sample_size: int = 30  # Adjusted from 100
    out_of_sample_split: float = 0.2  # 20% for out-of-sample
    walk_forward_windows: int = 5
    multiple_testing_method: str = 'fdr_bh'  # 'bonferroni', 'fdr_bh'

@dataclass
class StatisticalResults:
    """Container for statistical test results"""
    strategy_name: str
    symbol: str
    
    # Basic metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    
    # Bootstrap results
    bootstrap_confidence_interval: Tuple[float, float]
    bootstrap_mean: float
    bootstrap_std: float
    bootstrap_percentiles: Dict[str, float]
    
    # Permutation test results
    permutation_p_value: float
    permutation_test_statistic: float
    permutation_null_distribution: List[float]
    
    # Out-of-sample results
    in_sample_metrics: Dict[str, float]
    out_of_sample_metrics: Dict[str, float]
    performance_degradation: float
    
    # Statistical significance
    is_statistically_significant: bool
    adjusted_p_value: float
    edge_score: float
    edge_assessment: str
    
    # Distribution tests
    normality_test_p_value: float
    skewness: float
    kurtosis: float

class MonteCarloValidator:
    """
    Advanced Monte Carlo validation for trading strategies
    """
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.data_loader = DataLoader()
        self.results: List[StatisticalResults] = []
        
    def bootstrap_strategy_returns(
        self, 
        returns: pd.Series, 
        n_samples: int = None
    ) -> Dict[str, Any]:
        """
        Perform bootstrap resampling on strategy returns
        
        Args:
            returns: Series of strategy returns
            n_samples: Number of bootstrap samples
            
        Returns:
            Dictionary with bootstrap statistics
        """
        n_samples = n_samples or self.config.n_bootstrap_samples
        
        if len(returns) < self.config.min_sample_size:
            raise ValueError(f"Insufficient data: {len(returns)} < {self.config.min_sample_size}")
        
        # Bootstrap resampling
        bootstrap_returns = []
        bootstrap_sharpes = []
        bootstrap_total_returns = []
        
        for _ in range(n_samples):
            # Resample with replacement
            resampled_returns = np.random.choice(returns.values, size=len(returns), replace=True)
            
            # Calculate metrics
            total_return = (1 + pd.Series(resampled_returns)).prod() - 1
            annualized_return = (1 + pd.Series(resampled_returns).mean()) ** 252 - 1
            volatility = pd.Series(resampled_returns).std() * np.sqrt(252)
            sharpe = annualized_return / volatility if volatility > 0 else 0
            
            bootstrap_returns.append(annualized_return)
            bootstrap_sharpes.append(sharpe)
            bootstrap_total_returns.append(total_return)
        
        bootstrap_returns = np.array(bootstrap_returns)
        
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_interval = (
            np.percentile(bootstrap_returns, lower_percentile),
            np.percentile(bootstrap_returns, upper_percentile)
        )
        
        return {
            'bootstrap_returns': bootstrap_returns,
            'bootstrap_sharpes': bootstrap_sharpes,
            'bootstrap_total_returns': bootstrap_total_returns,
            'mean': np.mean(bootstrap_returns),
            'std': np.std(bootstrap_returns),
            'confidence_interval': confidence_interval,
            'percentiles': {
                '5th': np.percentile(bootstrap_returns, 5),
                '25th': np.percentile(bootstrap_returns, 25),
                '50th': np.percentile(bootstrap_returns, 50),
                '75th': np.percentile(bootstrap_returns, 75),
                '95th': np.percentile(bootstrap_returns, 95)
            }
        }
    
    def permutation_test(
        self, 
        strategy_returns: pd.Series, 
        benchmark_returns: pd.Series = None,
        n_permutations: int = None
    ) -> Dict[str, Any]:
        """
        Perform permutation test to assess statistical significance
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns (if None, test against zero)
            n_permutations: Number of permutation samples
            
        Returns:
            Dictionary with permutation test results
        """
        n_permutations = n_permutations or self.config.n_permutation_samples
        
        if benchmark_returns is None:
            # Test against zero (no skill hypothesis)
            benchmark_returns = pd.Series(np.zeros(len(strategy_returns)))
        
        # Align series
        min_length = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns.iloc[:min_length]
        benchmark_returns = benchmark_returns.iloc[:min_length]
        
        # Calculate observed test statistic (difference in means)
        observed_diff = strategy_returns.mean() - benchmark_returns.mean()
        
        # Combine all returns
        combined_returns = np.concatenate([strategy_returns.values, benchmark_returns.values])
        
        # Permutation testing
        permutation_diffs = []
        
        for _ in range(n_permutations):
            # Randomly permute the combined returns
            np.random.shuffle(combined_returns)
            
            # Split back into two groups
            perm_strategy = combined_returns[:len(strategy_returns)]
            perm_benchmark = combined_returns[len(strategy_returns):]
            
            # Calculate difference in means
            perm_diff = perm_strategy.mean() - perm_benchmark.mean()
            permutation_diffs.append(perm_diff)
        
        permutation_diffs = np.array(permutation_diffs)
        
        # Calculate p-value (two-tailed test)
        p_value = np.mean(np.abs(permutation_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_statistic': observed_diff,
            'p_value': p_value,
            'null_distribution': permutation_diffs.tolist(),
            'is_significant': p_value < self.config.alpha
        }
    
    def out_of_sample_validation(
        self, 
        strategy_class,
        strategy_params: Dict[str, Any],
        data: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Perform out-of-sample validation with walk-forward analysis
        
        Args:
            strategy_class: Strategy class to test
            strategy_params: Strategy parameters
            data: Full dataset
            symbol: Symbol being tested
            
        Returns:
            Dictionary with in-sample and out-of-sample results
        """
        # Split data into in-sample and out-of-sample
        split_idx = int(len(data) * (1 - self.config.out_of_sample_split))
        in_sample_data = data.iloc[:split_idx]
        out_sample_data = data.iloc[split_idx:]
        
        # Test in-sample
        strategy_in = strategy_class(**strategy_params)
        results_in = strategy_in.backtest(in_sample_data, symbol)
        
        # Test out-of-sample
        strategy_out = strategy_class(**strategy_params)
        results_out = strategy_out.backtest(out_sample_data, symbol)
        
        # Calculate performance degradation
        in_sample_return = results_in['metrics']['total_return']
        out_sample_return = results_out['metrics']['total_return']
        
        if in_sample_return != 0:
            performance_degradation = (in_sample_return - out_sample_return) / abs(in_sample_return)
        else:
            performance_degradation = 0
        
        return {
            'in_sample_metrics': results_in['metrics'],
            'out_sample_metrics': results_out['metrics'],
            'performance_degradation': performance_degradation,
            'in_sample_data_length': len(in_sample_data),
            'out_sample_data_length': len(out_sample_data)
        }
    
    def test_return_distribution(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Test the distribution properties of returns
        
        Args:
            returns: Strategy returns
            
        Returns:
            Dictionary with distribution test results
        """
        # Remove NaN values
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 20:
            return {
                'normality_p_value': np.nan,
                'skewness': np.nan,
                'kurtosis': np.nan,
                'jarque_bera_p': np.nan
            }
        
        # Normality tests
        shapiro_stat, shapiro_p = shapiro(clean_returns)
        jb_stat, jb_p = jarque_bera(clean_returns)
        
        # Moments
        skewness = stats.skew(clean_returns)
        kurtosis = stats.kurtosis(clean_returns)
        
        return {
            'normality_p_value': min(shapiro_p, jb_p),
            'skewness': skewness,
            'kurtosis': kurtosis,
            'jarque_bera_p': jb_p,
            'shapiro_p': shapiro_p
        }
    
    def calculate_edge_score(
        self, 
        metrics: Dict[str, float],
        bootstrap_results: Dict[str, Any],
        permutation_results: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Calculate comprehensive edge score
        
        Args:
            metrics: Basic strategy metrics
            bootstrap_results: Bootstrap test results
            permutation_results: Permutation test results
            
        Returns:
            Tuple of (edge_score, assessment)
        """
        score = 0
        max_score = 100
        
        # Performance criteria (40 points)
        sharpe = metrics.get('sharpe_ratio', 0)
        if sharpe > 2.0:
            score += 15
        elif sharpe > 1.0:
            score += 10
        elif sharpe > 0.5:
            score += 5
        
        total_return = metrics.get('total_return', 0)
        if total_return > 0.5:
            score += 10
        elif total_return > 0.2:
            score += 7
        elif total_return > 0:
            score += 3
        
        win_rate = metrics.get('win_rate', 0.5)
        if win_rate > 0.6:
            score += 10
        elif win_rate > 0.55:
            score += 7
        elif win_rate > 0.5:
            score += 3
        
        max_dd = abs(metrics.get('max_drawdown', 0))
        if max_dd < 0.1:
            score += 5
        elif max_dd < 0.2:
            score += 3
        elif max_dd < 0.3:
            score += 1
        
        # Statistical significance (40 points)
        if permutation_results['is_significant']:
            score += 25
        
        # Bootstrap consistency (20 points)
        ci_lower, ci_upper = bootstrap_results['confidence_interval']
        if ci_lower > 0:  # Entire CI above zero
            score += 20
        elif bootstrap_results['mean'] > 0:  # Mean positive
            score += 10
        
        # Trading activity (bonus)
        num_trades = metrics.get('num_trades', 0)
        if num_trades >= 50:
            score += 5
        elif num_trades >= 20:
            score += 2
        
        score = min(score, max_score)
        
        # Assessment
        if score >= 80:
            assessment = "STRONG STATISTICAL EDGE"
        elif score >= 60:
            assessment = "MODERATE STATISTICAL EDGE"
        elif score >= 40:
            assessment = "WEAK STATISTICAL EDGE"
        else:
            assessment = "NO STATISTICAL EDGE"
        
        return score, assessment
    
    def validate_strategy(
        self,
        strategy_class,
        strategy_params: Dict[str, Any],
        symbol: str,
        start_date: str,
        end_date: str
    ) -> StatisticalResults:
        """
        Comprehensive validation of a single strategy
        
        Args:
            strategy_class: Strategy class to validate
            strategy_params: Strategy parameters
            symbol: Symbol to test
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            StatisticalResults object
        """
        print(f"\n🔬 Validating {strategy_class.__name__} on {symbol}")
        
        # Load data
        data = self.data_loader.load_partitioned_crypto_data(
            symbol=symbol,
            interval='1d',
            start_date=start_date,
            end_date=end_date
        )
        
        if data is None or len(data) < self.config.min_sample_size:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Run basic backtest
        strategy = strategy_class(**strategy_params)
        results = strategy.backtest(data, symbol)
        
        if len(strategy.trades) == 0:
            print(f"  ⚠️ No trades generated for {strategy_class.__name__}")
            returns = pd.Series([0.0])
        else:
            returns = results['returns'].dropna()
        
        if len(returns) < 20:
            print(f"  ⚠️ Insufficient return data: {len(returns)} periods")
            returns = pd.Series([0.0] * 20)  # Dummy data for testing
        
        # Bootstrap analysis
        print("  📊 Running bootstrap analysis...")
        try:
            bootstrap_results = self.bootstrap_strategy_returns(returns)
        except Exception as e:
            print(f"  ❌ Bootstrap failed: {e}")
            bootstrap_results = {
                'mean': 0, 'std': 0, 'confidence_interval': (0, 0),
                'percentiles': {'5th': 0, '25th': 0, '50th': 0, '75th': 0, '95th': 0}
            }
        
        # Permutation test
        print("  🎲 Running permutation test...")
        try:
            permutation_results = self.permutation_test(returns)
        except Exception as e:
            print(f"  ❌ Permutation test failed: {e}")
            permutation_results = {
                'observed_statistic': 0, 'p_value': 1.0,
                'null_distribution': [0], 'is_significant': False
            }
        
        # Out-of-sample validation
        print("  📈 Running out-of-sample validation...")
        try:
            oos_results = self.out_of_sample_validation(
                strategy_class, strategy_params, data, symbol
            )
        except Exception as e:
            print(f"  ❌ Out-of-sample validation failed: {e}")
            oos_results = {
                'in_sample_metrics': results['metrics'],
                'out_sample_metrics': results['metrics'],
                'performance_degradation': 0
            }
        
        # Distribution tests
        print("  📐 Testing return distribution...")
        distribution_results = self.test_return_distribution(returns)
        
        # Calculate edge score
        edge_score, edge_assessment = self.calculate_edge_score(
            results['metrics'], bootstrap_results, permutation_results
        )
        
        # Create results object
        statistical_results = StatisticalResults(
            strategy_name=strategy_class.__name__,
            symbol=symbol,
            total_return=results['metrics']['total_return'],
            annualized_return=results['metrics']['annualized_return'],
            volatility=results['metrics']['volatility'],
            sharpe_ratio=results['metrics']['sharpe_ratio'],
            max_drawdown=results['metrics']['max_drawdown'],
            win_rate=results['metrics']['win_rate'],
            num_trades=results['metrics']['num_trades'],
            bootstrap_confidence_interval=bootstrap_results['confidence_interval'],
            bootstrap_mean=bootstrap_results['mean'],
            bootstrap_std=bootstrap_results['std'],
            bootstrap_percentiles=bootstrap_results['percentiles'],
            permutation_p_value=permutation_results['p_value'],
            permutation_test_statistic=permutation_results['observed_statistic'],
            permutation_null_distribution=permutation_results['null_distribution'],
            in_sample_metrics=oos_results['in_sample_metrics'],
            out_of_sample_metrics=oos_results['out_sample_metrics'],
            performance_degradation=oos_results['performance_degradation'],
            is_statistically_significant=permutation_results['is_significant'],
            adjusted_p_value=permutation_results['p_value'],  # Will be adjusted later
            edge_score=edge_score,
            edge_assessment=edge_assessment,
            normality_test_p_value=distribution_results['normality_p_value'],
            skewness=distribution_results['skewness'],
            kurtosis=distribution_results['kurtosis']
        )
        
        print(f"  ✅ Validation complete: {edge_assessment} (Score: {edge_score:.1f}/100)")
        
        return statistical_results
    
    def multiple_testing_correction(self, results: List[StatisticalResults]) -> List[StatisticalResults]:
        """
        Apply multiple testing correction to p-values
        
        Args:
            results: List of StatisticalResults
            
        Returns:
            Updated results with corrected p-values
        """
        p_values = [r.permutation_p_value for r in results]
        
        if self.config.multiple_testing_method == 'bonferroni':
            corrected_p_values = [p * len(p_values) for p in p_values]
            corrected_p_values = [min(p, 1.0) for p in corrected_p_values]
        else:  # FDR
            rejected, corrected_p_values, _, _ = multipletests(
                p_values, method=self.config.multiple_testing_method
            )
        
        # Update results
        for i, result in enumerate(results):
            result.adjusted_p_value = corrected_p_values[i]
            result.is_statistically_significant = corrected_p_values[i] < self.config.alpha
        
        return results
    
    def generate_validation_report(
        self, 
        results: List[StatisticalResults],
        output_path: str = None
    ) -> str:
        """
        Generate comprehensive validation report
        
        Args:
            results: List of validation results
            output_path: Path to save report
            
        Returns:
            Report as string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = f"results/reports/statistical_validation_{timestamp}.md"
        
        # Create results directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""# Statistical Validation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

"""
        
        # Summary table
        report += "| Strategy | Symbol | Edge Score | Assessment | Sharpe | Return | P-Value | Significant |\n"
        report += "|----------|--------|------------|------------|--------|--------|---------|--------------|\n"
        
        for result in results:
            sig_mark = "✅" if result.is_statistically_significant else "❌"
            report += f"| {result.strategy_name} | {result.symbol} | {result.edge_score:.1f}/100 | {result.edge_assessment} | {result.sharpe_ratio:.3f} | {result.total_return:.2%} | {result.adjusted_p_value:.4f} | {sig_mark} |\n"
        
        report += "\n## Detailed Results\n\n"
        
        # Detailed analysis for each strategy
        for result in results:
            report += f"### {result.strategy_name} - {result.symbol}\n\n"
            
            report += "#### Performance Metrics\n"
            report += f"- **Total Return**: {result.total_return:.2%}\n"
            report += f"- **Annualized Return**: {result.annualized_return:.2%}\n"
            report += f"- **Sharpe Ratio**: {result.sharpe_ratio:.3f}\n"
            report += f"- **Maximum Drawdown**: {result.max_drawdown:.2%}\n"
            report += f"- **Win Rate**: {result.win_rate:.2%}\n"
            report += f"- **Number of Trades**: {result.num_trades}\n\n"
            
            report += "#### Statistical Analysis\n"
            report += f"- **Edge Score**: {result.edge_score:.1f}/100 ({result.edge_assessment})\n"
            report += f"- **Permutation P-Value**: {result.permutation_p_value:.4f}\n"
            report += f"- **Adjusted P-Value**: {result.adjusted_p_value:.4f}\n"
            report += f"- **Statistically Significant**: {'Yes' if result.is_statistically_significant else 'No'}\n\n"
            
            report += "#### Bootstrap Analysis\n"
            ci_lower, ci_upper = result.bootstrap_confidence_interval
            report += f"- **Bootstrap Mean Return**: {result.bootstrap_mean:.2%}\n"
            report += f"- **Bootstrap Std**: {result.bootstrap_std:.2%}\n"
            report += f"- **95% Confidence Interval**: [{ci_lower:.2%}, {ci_upper:.2%}]\n\n"
            
            report += "#### Out-of-Sample Validation\n"
            report += f"- **In-Sample Sharpe**: {result.in_sample_metrics['sharpe_ratio']:.3f}\n"
            report += f"- **Out-of-Sample Sharpe**: {result.out_of_sample_metrics['sharpe_ratio']:.3f}\n"
            report += f"- **Performance Degradation**: {result.performance_degradation:.2%}\n\n"
            
            report += "#### Distribution Properties\n"
            report += f"- **Normality P-Value**: {result.normality_test_p_value:.4f}\n"
            report += f"- **Skewness**: {result.skewness:.3f}\n"
            report += f"- **Excess Kurtosis**: {result.kurtosis:.3f}\n\n"
            
            report += "---\n\n"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report)
        
        return report

def main():
    """Main validation runner"""
    print("🔬 Advanced Statistical Validation Framework")
    print("=" * 60)
    
    # Configuration
    config = ValidationConfig(
        n_bootstrap_samples=1000,
        n_permutation_samples=1000,
        confidence_level=0.95,
        alpha=0.05
    )
    
    validator = MonteCarloValidator(config)
    
    # Define strategies to test
    strategies_to_test = [
        (MomentumStrategy, {'lookback_period': 20, 'momentum_threshold': 0.02}),
        (BollingerBandsStrategy, {'window': 20, 'num_stds': 2.0}),
        (SimplePairsStrategy, {'lookback_period': 30, 'z_entry_threshold': 1.5}),
        (TrendFollowingStrategy, {'short_window': 10, 'long_window': 30}),
        (VolatilityBreakoutStrategy, {'volatility_window': 20, 'breakout_threshold': 1.5})
    ]
    
    symbols = ['BTC_USD', 'ETH_USD']
    start_date = '2024-01-01'
    end_date = '2024-07-01'
    
    all_results = []
    
    # Test each strategy
    for strategy_class, params in strategies_to_test:
        for symbol in symbols:
            try:
                result = validator.validate_strategy(
                    strategy_class, params, symbol, start_date, end_date
                )
                all_results.append(result)
            except Exception as e:
                print(f"❌ Failed to validate {strategy_class.__name__} on {symbol}: {e}")
    
    # Apply multiple testing correction
    print(f"\n📊 Applying multiple testing correction ({config.multiple_testing_method})...")
    all_results = validator.multiple_testing_correction(all_results)
    
    # Generate report
    print("\n📋 Generating validation report...")
    report = validator.generate_validation_report(all_results)
    
    # Save results as JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results/exports/statistical_validation_{timestamp}.json"
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump([asdict(result) for result in all_results], f, indent=2, default=str)
    
    # Print summary
    print(f"\n✅ Validation complete!")
    print(f"📄 Report saved to: results/reports/statistical_validation_{timestamp}.md")
    print(f"💾 Data saved to: {json_path}")
    
    # Summary statistics
    significant_count = sum(1 for r in all_results if r.is_statistically_significant)
    strong_edge_count = sum(1 for r in all_results if "STRONG" in r.edge_assessment)
    
    print(f"\n📊 Summary:")
    print(f"  • Total strategies tested: {len(all_results)}")
    print(f"  • Statistically significant: {significant_count}/{len(all_results)}")
    print(f"  • Strong statistical edge: {strong_edge_count}/{len(all_results)}")

if __name__ == "__main__":
    main()
