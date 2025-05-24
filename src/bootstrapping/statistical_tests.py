"""
Statistical Tests Module

Comprehensive statistical testing suite for bootstrap validation,
including significance tests, distribution comparisons, and multiple
hypothesis testing corrections.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from scipy.stats import kstest, anderson, jarque_bera
import warnings


class StatisticalTests:
    """
    Comprehensive statistical testing for bootstrap validation.
    
    Implements various statistical tests to validate bootstrap results
    and compare strategies against benchmarks with proper multiple
    comparison corrections.
    """
    
    def __init__(self, ret_series: pd.Series, benchmark_series: Optional[pd.Series] = None):
        """
        Initialize statistical tests.
        
        Args:
            ret_series: Strategy return series
            benchmark_series: Optional benchmark return series
        """
        self.ret_series = ret_series
        self.benchmark_series = benchmark_series
        
    def empirical_p_values(self, simulated_stats: List[Dict[str, float]], 
                          benchmark_stats: Optional[Dict[str, float]] = None,
                          metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Calculate empirical p-values for strategy vs benchmark comparison.
        
        Args:
            simulated_stats: List of simulated statistics
            benchmark_stats: Benchmark statistics (if None, calculated from benchmark_series)
            metrics: Metrics to test (if None, uses all available)
            
        Returns:
            Dictionary of p-values and test statistics
        """
        if benchmark_stats is None and self.benchmark_series is not None:
            benchmark_stats = self._calculate_benchmark_stats()
        elif benchmark_stats is None:
            warnings.warn("No benchmark provided for empirical p-value calculation")
            return {}
            
        # Convert to DataFrame for easier manipulation
        sim_df = pd.DataFrame(simulated_stats)
        
        if metrics is None:
            metrics = sim_df.columns.tolist()
            
        results = {}
        
        for metric in metrics:
            if metric not in sim_df.columns or metric not in benchmark_stats:
                continue
                
            sim_values = sim_df[metric].dropna().values
            bench_value = benchmark_stats[metric]
            
            if len(sim_values) == 0 or np.isnan(bench_value):
                continue
                
            # Calculate ranks and p-values
            rank = np.sum(sim_values <= bench_value)
            n = len(sim_values)
            
            # Empirical p-values with continuity correction
            p_left = (rank + 1) / (n + 1)
            p_right = 1 - p_left
            p_two_sided = 2 * min(p_left, p_right)
            
            # Mid-p adjustment for better small sample properties
            p_left_mid = rank / n + 0.5 / n
            p_right_mid = 1 - p_left_mid
            p_two_sided_mid = 2 * min(p_left_mid, p_right_mid)
            
            results[metric] = {
                'benchmark_value': bench_value,
                'sim_mean': sim_values.mean(),
                'sim_std': sim_values.std(),
                'rank': rank,
                'p_left': p_left,
                'p_right': p_right,
                'p_two_sided': p_two_sided,
                'p_two_sided_mid': p_two_sided_mid,
                'effect_size': (sim_values.mean() - bench_value) / sim_values.std() if sim_values.std() > 0 else np.nan
            }
            
        return results
    
    def multiple_comparison_correction(self, p_values: Dict[str, float], 
                                     method: str = 'bonferroni') -> Dict[str, float]:
        """
        Apply multiple comparison corrections to p-values.
        
        Args:
            p_values: Dictionary of p-values
            method: Correction method ('bonferroni', 'fdr_bh', 'fdr_by')
            
        Returns:
            Corrected p-values
        """
        from statsmodels.stats.multitest import multipletests
        
        p_vals = list(p_values.values())
        metrics = list(p_values.keys())
        
        if method == 'bonferroni':
            corrected = [min(p * len(p_vals), 1.0) for p in p_vals]
        else:
            _, corrected, _, _ = multipletests(p_vals, method=method)
            
        return dict(zip(metrics, corrected))
    
    def kolmogorov_smirnov_test(self, simulated_stats: List[Dict[str, float]], 
                               original_stats: Dict[str, float],
                               metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Kolmogorov-Smirnov test for distribution comparison.
        
        Args:
            simulated_stats: List of simulated statistics
            original_stats: Original strategy statistics
            metrics: Metrics to test
            
        Returns:
            KS test results
        """
        sim_df = pd.DataFrame(simulated_stats)
        
        if metrics is None:
            metrics = sim_df.columns.tolist()
            
        results = {}
        
        for metric in metrics:
            if metric not in sim_df.columns or metric not in original_stats:
                continue
                
            sim_values = sim_df[metric].dropna().values
            
            if len(sim_values) < 2:
                continue
                
            # Test against normal distribution
            ks_stat_normal, p_val_normal = kstest(sim_values, 'norm', 
                                                 args=(sim_values.mean(), sim_values.std()))
            
            # Test against uniform distribution
            ks_stat_uniform, p_val_uniform = kstest(sim_values, 'uniform',
                                                   args=(sim_values.min(), sim_values.max() - sim_values.min()))
            
            results[metric] = {
                'ks_stat_normal': ks_stat_normal,
                'p_val_normal': p_val_normal,
                'ks_stat_uniform': ks_stat_uniform,
                'p_val_uniform': p_val_uniform,
                'is_normal': p_val_normal > 0.05,
                'is_uniform': p_val_uniform > 0.05
            }
            
        return results
    
    def anderson_darling_test(self, simulated_stats: List[Dict[str, float]],
                             metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Anderson-Darling test for normality.
        
        Args:
            simulated_stats: List of simulated statistics
            metrics: Metrics to test
            
        Returns:
            Anderson-Darling test results
        """
        sim_df = pd.DataFrame(simulated_stats)
        
        if metrics is None:
            metrics = sim_df.columns.tolist()
            
        results = {}
        
        for metric in metrics:
            if metric not in sim_df.columns:
                continue
                
            sim_values = sim_df[metric].dropna().values
            
            if len(sim_values) < 8:  # Minimum sample size for AD test
                continue
                
            try:
                ad_stat, critical_vals, significance_level = anderson(sim_values, dist='norm')
                
                # Determine if normal at 5% level
                is_normal = ad_stat < critical_vals[2]  # 5% critical value is at index 2
                
                results[metric] = {
                    'ad_statistic': ad_stat,
                    'critical_values': critical_vals.tolist(),
                    'significance_levels': [15, 10, 5, 2.5, 1],
                    'is_normal_5pct': is_normal,
                    'p_value_approx': 1 - significance_level if ad_stat > critical_vals[2] else None
                }
            except Exception as e:
                warnings.warn(f"Anderson-Darling test failed for {metric}: {e}")
                continue
                
        return results
    
    def jarque_bera_test(self, simulated_stats: List[Dict[str, float]],
                        metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Jarque-Bera test for normality.
        
        Args:
            simulated_stats: List of simulated statistics
            metrics: Metrics to test
            
        Returns:
            Jarque-Bera test results
        """
        sim_df = pd.DataFrame(simulated_stats)
        
        if metrics is None:
            metrics = sim_df.columns.tolist()
            
        results = {}
        
        for metric in metrics:
            if metric not in sim_df.columns:
                continue
                
            sim_values = sim_df[metric].dropna().values
            
            if len(sim_values) < 2:
                continue
                
            try:
                jb_stat, p_val = jarque_bera(sim_values)
                
                results[metric] = {
                    'jb_statistic': jb_stat,
                    'p_value': p_val,
                    'is_normal': p_val > 0.05,
                    'skewness': stats.skew(sim_values),
                    'kurtosis': stats.kurtosis(sim_values)
                }
            except Exception as e:
                warnings.warn(f"Jarque-Bera test failed for {metric}: {e}")
                continue
                
        return results
    
    def mann_whitney_test(self, simulated_stats: List[Dict[str, float]],
                         benchmark_stats: Optional[Dict[str, float]] = None,
                         metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Mann-Whitney U test for non-parametric comparison.
        
        Args:
            simulated_stats: List of simulated statistics
            benchmark_stats: Benchmark statistics
            metrics: Metrics to test
            
        Returns:
            Mann-Whitney test results
        """
        if benchmark_stats is None and self.benchmark_series is not None:
            benchmark_stats = self._calculate_benchmark_stats()
        elif benchmark_stats is None:
            return {}
            
        sim_df = pd.DataFrame(simulated_stats)
        
        if metrics is None:
            metrics = sim_df.columns.tolist()
            
        results = {}
        
        for metric in metrics:
            if metric not in sim_df.columns or metric not in benchmark_stats:
                continue
                
            sim_values = sim_df[metric].dropna().values
            bench_value = benchmark_stats[metric]
            
            if len(sim_values) < 2 or np.isnan(bench_value):
                continue
                
            # Create benchmark "sample" by repeating the value
            # This is a simplification - ideally we'd have multiple benchmark observations
            bench_sample = np.array([bench_value])
            
            try:
                statistic, p_value = stats.mannwhitneyu(
                    sim_values, bench_sample, 
                    alternative='two-sided'
                )
                
                results[metric] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'median_sim': np.median(sim_values),
                    'median_bench': bench_value
                }
            except Exception as e:
                warnings.warn(f"Mann-Whitney test failed for {metric}: {e}")
                continue
                
        return results
    
    def wilcoxon_signed_rank_test(self, paired_differences: np.ndarray) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test for paired samples.
        
        Args:
            paired_differences: Array of paired differences
            
        Returns:
            Wilcoxon test results
        """
        if len(paired_differences) < 6:  # Minimum for Wilcoxon test
            return {}
            
        try:
            statistic, p_value = stats.wilcoxon(paired_differences, alternative='two-sided')
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'median_difference': np.median(paired_differences),
                'mean_difference': np.mean(paired_differences)
            }
        except Exception as e:
            warnings.warn(f"Wilcoxon test failed: {e}")
            return {}
    
    def autocorrelation_tests(self, returns: np.ndarray, lags: int = 10) -> Dict[str, Any]:
        """
        Test for autocorrelation in returns.
        
        Args:
            returns: Return array
            lags: Number of lags to test
            
        Returns:
            Autocorrelation test results
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.tsa.stattools import acf
        
        try:
            # Ljung-Box test
            ljung_box = acorr_ljungbox(returns, lags=lags, return_df=True)
            
            # Autocorrelation function
            acf_values = acf(returns, nlags=lags, fft=True)
            
            return {
                'ljung_box_stats': ljung_box['lb_stat'].tolist(),
                'ljung_box_pvalues': ljung_box['lb_pvalue'].tolist(),
                'autocorrelations': acf_values.tolist(),
                'significant_autocorr': np.any(ljung_box['lb_pvalue'] < 0.05)
            }
        except Exception as e:
            warnings.warn(f"Autocorrelation tests failed: {e}")
            return {}
    
    def arch_test(self, returns: np.ndarray, lags: int = 5) -> Dict[str, float]:
        """
        ARCH test for heteroscedasticity.
        
        Args:
            returns: Return array
            lags: Number of lags for ARCH test
            
        Returns:
            ARCH test results
        """
        try:
            from arch.univariate import arch_model
            
            # Fit ARCH model
            model = arch_model(returns, vol='ARCH', p=lags)
            results = model.fit(disp='off')
            
            # Extract test statistics
            arch_stat = results.arch_lm_test(lags)
            
            return {
                'arch_statistic': arch_stat['statistic'],
                'p_value': arch_stat['pvalue'],
                'has_arch_effects': arch_stat['pvalue'] < 0.05
            }
        except Exception as e:
            warnings.warn(f"ARCH test failed: {e}")
            return {}
    
    def _calculate_benchmark_stats(self) -> Dict[str, float]:
        """Calculate statistics for benchmark series."""
        if self.benchmark_series is None:
            return {}
            
        # This would use the same metric calculation as in core.py
        # For now, return basic metrics
        returns = self.benchmark_series.values
        
        return {
            'CumulativeReturn': (1 + returns).prod() - 1,
            'Volatility': np.std(returns) * np.sqrt(252),  # Assuming daily data
            'Sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else np.nan,
            'MaxDrawdown': self._calculate_max_drawdown(returns),
            'Skewness': stats.skew(returns),
            'Kurtosis': stats.kurtosis(returns)
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown for returns array."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def run_all_tests(self, bootstrap_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive statistical test suite.
        
        Args:
            bootstrap_results: Results from bootstrap simulation
            
        Returns:
            Complete test results
        """
        simulated_stats = bootstrap_results['simulated_stats']
        original_stats = bootstrap_results['original_stats']
        
        results = {
            'empirical_p_values': self.empirical_p_values(simulated_stats),
            'kolmogorov_smirnov': self.kolmogorov_smirnov_test(simulated_stats, original_stats),
            'anderson_darling': self.anderson_darling_test(simulated_stats),
            'jarque_bera': self.jarque_bera_test(simulated_stats),
            'mann_whitney': self.mann_whitney_test(simulated_stats),
            'autocorrelation': self.autocorrelation_tests(self.ret_series.values),
            'arch_test': self.arch_test(self.ret_series.values)
        }
        
        # Apply multiple comparison corrections
        if results['empirical_p_values']:
            p_vals = {metric: data['p_two_sided'] for metric, data in results['empirical_p_values'].items()}
            results['bonferroni_corrected'] = self.multiple_comparison_correction(p_vals, 'bonferroni')
            results['fdr_corrected'] = self.multiple_comparison_correction(p_vals, 'fdr_bh')
        
        return results
