"""
Statistical Tests Module

Comprehensive statistical testing suite for bootstrap validation,
including significance tests, distribution comparisons, and multiple
hypothesis testing corrections.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable # Added Callable
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

    def permutation_test_on_signs(
        self,
        n_permutations: int = 10000,
        alternative: str = 'two-sided',
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Performs a permutation test on signs (Monte Carlo sign test).

        Tests the null hypothesis that the median of the differences (or returns) is zero.
        If benchmark_series is provided via the class constructor, tests strategy_returns - benchmark_returns.
        Otherwise, tests strategy_returns (provided via constructor) against zero.

        Args:
            n_permutations: Number of permutations to generate for the null distribution.
            alternative: Defines the alternative hypothesis.
                         'two-sided': median is not zero.
                         'greater': median is greater than zero.
                         'less': median is less than zero.
            random_seed: Optional seed for the random number generator for reproducibility.

        Returns:
            Dictionary with test results:
                'observed_statistic': Number of positive signs in the original data.
                'n_nonzero': Number of non-zero differences/returns used.
                'p_value': The calculated p-value.
                'alternative': The alternative hypothesis tested.
                'n_permutations': Number of permutations used.
                'series_tested': Description of the series that was tested.
                'info': Additional information, e.g., if no non-zero values were found.
        """
        if self.benchmark_series is not None:
            if len(self.ret_series) != len(self.benchmark_series):
                # This check should ideally be in __init__ or when series are set
                warnings.warn("Return series and benchmark series have different lengths. Test might be misleading.")
                # Proceeding by aligning, though this is not ideal.
                min_len = min(len(self.ret_series), len(self.benchmark_series))
                diff_series = self.ret_series.values[:min_len] - self.benchmark_series.values[:min_len]
            else:
                diff_series = self.ret_series.values - self.benchmark_series.values
            series_name = "Differences (Strategy - Benchmark)"
        else:
            diff_series = self.ret_series.values
            series_name = "Strategy Returns"

        # Remove zeros
        non_zero_diffs = diff_series[diff_series != 0]
        m = len(non_zero_diffs)

        if m == 0:
            warnings.warn(f"No non-zero values in {series_name} for permutation sign test. Returning NaN p-value.")
            return {
                'observed_statistic': 0,
                'n_nonzero': 0,
                'p_value': np.nan,
                'alternative': alternative,
                'n_permutations': n_permutations,
                'series_tested': series_name,
                'info': f"No non-zero values in {series_name}."
            }

        # Observed statistic: number of positive signs
        observed_statistic = np.sum(non_zero_diffs > 0)

        # Initialize RNG for permutations
        rng = np.random.default_rng(seed=random_seed)
        
        perm_positive_counts = np.zeros(n_permutations, dtype=int)

        for i in range(n_permutations):
            # For each of the m non-zero values, randomly assign a sign (+1 or -1)
            # This simulates drawing from Binomial(m, 0.5) for the count of positive signs
            perm_positive_counts[i] = rng.binomial(m, 0.5)
            # Alternative (explicit sign permutation, less direct for this specific statistic):
            # random_signs = rng.choice([-1, 1], size=m, replace=True)
            # perm_positive_counts[i] = np.sum(random_signs > 0)


        # Calculate p-value using the (count_extreme + 1) / (n_permutations + 1) formula
        if alternative == 'greater':
            count_extreme = np.sum(perm_positive_counts >= observed_statistic)
        elif alternative == 'less':
            count_extreme = np.sum(perm_positive_counts <= observed_statistic)
        elif alternative == 'two-sided':
            expected_value_H0 = m / 2.0
            observed_deviation_from_mean = abs(observed_statistic - expected_value_H0)
            perm_deviations_from_mean = np.abs(perm_positive_counts - expected_value_H0)
            # Count how many permuted deviations are >= observed deviation
            # Adding a small epsilon for float comparisons can sometimes be useful, but generally not needed for counts.
            count_extreme = np.sum(perm_deviations_from_mean >= observed_deviation_from_mean)
        else:
            raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'.")

        p_value = (count_extreme + 1) / (n_permutations + 1)
        p_value = min(p_value, 1.0) # Ensure p-value does not exceed 1.0

        return {
            'observed_statistic': observed_statistic,
            'n_nonzero': m,
            'p_value': p_value,
            'alternative': alternative,
            'n_permutations': n_permutations,
            'series_tested': series_name,
            'info': None
        }

    def whites_reality_check(
        self,
        observed_performances: List[float], 
        bootstrapped_performances: List[List[float]],
        benchmark_performance: float = 0.0,
        higher_is_better: bool = True
    ) -> Dict[str, Any]:
        """
        Performs White's Reality Check (RC).

        Tests the null hypothesis H0: max_k E[performance_k - benchmark] <= 0.
        This means the best rule is no better than the benchmark.

        Args:
            observed_performances: A list of observed performance metrics for L rules.
                                   Example: [rule1_perf, rule2_perf, ...]
            bootstrapped_performances: A list of L lists, where each inner list
                                       contains B bootstrapped performance metrics
                                       for the corresponding rule.
                                       Shape: (L_rules, B_samples)
                                       Example: [[rule1_boot1, rule1_boot2,...],
                                                 [rule2_boot1, rule2_boot2,...]]
            benchmark_performance: The performance of the benchmark. Default is 0.0.
            higher_is_better: If True, higher performance values are considered better.
                              If False (e.g., for risk metrics), lower values are better.

        Returns:
            A dictionary containing:
                'p_value': White's Reality Check p-value.
                'observed_max_performance_delta': The maximum observed performance
                                                      difference from the benchmark.
                                                      (obs_perf - benchmark_perf if higher_is_better,
                                                       benchmark_perf - obs_perf if lower_is_better).
                'num_rules': Number of rules tested.
                'num_bootstrap_samples': Number of bootstrap samples per rule.
                'info': Any additional information or warnings.
        """
        num_rules = len(observed_performances)
        if num_rules == 0:
            return {
                'p_value': np.nan, 
                'observed_max_performance_delta': np.nan,
                'num_rules': 0,
                'num_bootstrap_samples': 0,
                'info': "No rules provided."
            }

        if not bootstrapped_performances or len(bootstrapped_performances) != num_rules:
            raise ValueError(
                "Mismatch in length between observed_performances and bootstrapped_performances."
            )

        try:
            num_bootstrap_samples = len(bootstrapped_performances[0])
        except TypeError: # bootstrapped_performances[0] might not be indexable if not list of lists
            raise ValueError("bootstrapped_performances should be a list of lists.")
        
        if num_bootstrap_samples == 0:
             return {
                'p_value': np.nan, 
                'observed_max_performance_delta': np.nan,
                'num_rules': num_rules,
                'num_bootstrap_samples': 0,
                'info': "No bootstrap samples provided for rules."
            }

        if any(len(b_perf) != num_bootstrap_samples for b_perf in bootstrapped_performances):
            raise ValueError("All rules must have the same number of bootstrap samples.")

        obs_perf_arr = np.array(observed_performances, dtype=float)
        boot_perf_arr = np.array(bootstrapped_performances, dtype=float) # Shape (L, B)

        # 1. Calculate observed performance differences from benchmark
        # f_k_obs_delta = f_k_obs - benchmark (if higher is better)
        # f_k_obs_delta = benchmark - f_k_obs (if lower is better)
        if higher_is_better:
            obs_perf_delta = obs_perf_arr - benchmark_performance
        else:
            obs_perf_delta = benchmark_performance - obs_perf_arr

        # 2. Observed Reality Check statistic V
        # V = max_k (f_k_obs_delta)
        V_observed = np.max(obs_perf_delta)

        # 3. Construct bootstrap statistics for V
        # For each bootstrap sample b, calculate V_b = max_k ( (f_k_b - f_k_obs) or (f_k_obs - f_k_b) )
        # These are the demeaned bootstrap performances.
        simulated_V_b_values = np.zeros(num_bootstrap_samples, dtype=float)

        for b_idx in range(num_bootstrap_samples):
            # demeaned_perf_kb = f_k_b - f_k_obs
            demeaned_boot_perf_for_sample_b = boot_perf_arr[:, b_idx] - obs_perf_arr
            
            if not higher_is_better:
                # If lower is better, we are interested in max_k (f_k_obs - f_k_b)
                demeaned_boot_perf_for_sample_b = -demeaned_boot_perf_for_sample_b
            
            simulated_V_b_values[b_idx] = np.max(demeaned_boot_perf_for_sample_b)

        # 4. Calculate p-value
        # p_RC = (Count(simulated_V_b >= V_observed) + 1) / (num_bootstrap_samples + 1)
        count_extreme = np.sum(simulated_V_b_values >= V_observed)
        p_value = (count_extreme + 1) / (num_bootstrap_samples + 1)
        p_value = min(p_value, 1.0) # Ensure p-value does not exceed 1.0

        return {
            'p_value': p_value,
            'observed_max_performance_delta': V_observed,
            'num_rules': num_rules,
            'num_bootstrap_samples': num_bootstrap_samples,
            'info': None
        }

    def romano_wolf_stepm(
        self,
        observed_performances: List[float],
        bootstrapped_performances: List[List[float]],
        benchmark_performance: float = 0.0,
        higher_is_better: bool = True,
        alpha: float = 0.05,
        iterations: int = 1 # Number of "steps" in the step-down procedure
    ) -> Dict[str, Any]:
        """
        Performs Romano-Wolf Stepwise Multiple Testing (StepM).

        This is an improvement over White's Reality Check for controlling FWER.
        It iteratively removes hypotheses that are deemed insignificant.

        Args:
            observed_performances: List of observed performance metrics for L rules.
            bootstrapped_performances: List of L lists, each with B bootstrapped metrics.
                                       Shape: (L_rules, B_samples)
            benchmark_performance: Performance of the benchmark.
            higher_is_better: True if higher performance is better.
            alpha: Significance level for hypothesis rejection.
            iterations: Number of step-down iterations. More iterations can improve power
                        but increase computation. Typically 1-3.

        Returns:
            A dictionary containing:
                'rejected_hypotheses_indices': Indices of rules whose null hypothesis
                                                 (performance <= benchmark) was rejected.
                'p_values_stepwise': Adjusted p-values for each rule after the procedure.
                'num_rules': Initial number of rules.
                'num_bootstrap_samples': Number of bootstrap samples.
                'alpha': Significance level used.
                'info': Additional information or warnings.
        """
        num_rules = len(observed_performances)
        if num_rules == 0:
            return {
                'rejected_hypotheses_indices': [],
                'p_values_stepwise': [],
                'num_rules': 0,
                'num_bootstrap_samples': 0,
                'alpha': alpha,
                'info': "No rules provided."
            }

        if not bootstrapped_performances or len(bootstrapped_performances) != num_rules:
            raise ValueError("Mismatch: observed_performances and bootstrapped_performances lengths.")
        
        try:
            num_bootstrap_samples = len(bootstrapped_performances[0])
        except TypeError:
            raise ValueError("bootstrapped_performances should be a list of lists.")

        if num_bootstrap_samples == 0:
            return {
                'rejected_hypotheses_indices': [],
                'p_values_stepwise': [np.nan] * num_rules,
                'num_rules': num_rules,
                'num_bootstrap_samples': 0,
                'alpha': alpha,
                'info': "No bootstrap samples provided."
            }
        
        if any(len(b_perf) != num_bootstrap_samples for b_perf in bootstrapped_performances):
            raise ValueError("All rules must have the same number of bootstrap samples.")

        obs_perf_arr = np.array(observed_performances, dtype=float)
        boot_perf_arr = np.array(bootstrapped_performances, dtype=float) # Shape (L, B)

        # Calculate performance deltas (observed and bootstrapped)
        if higher_is_better:
            obs_deltas = obs_perf_arr - benchmark_performance
            # boot_deltas_kb = f_k_b - benchmark. We need f_k_b - f_k_obs for studentization later.
            # For now, let's use (f_k_b - f_k_obs) as the core of the bootstrap distribution.
            # The Romano-Wolf paper uses (l_i,b - l_i_hat) where l is loss.
            # So for performance, it's (p_i_hat - p_i,b) if higher is better.
            # Or, more directly, center the bootstrapped deltas around zero.
            # demeaned_boot_perf_kb = boot_perf_arr_kb - mean(boot_perf_arr_k*)
            # centered_boot_deltas = boot_perf_arr - np.mean(boot_perf_arr, axis=1, keepdims=True)
            # For StepM, we test H0: mu_k <= 0. The test statistics are t_k = obs_deltas_k.
            # The bootstrapped test statistics are t_k_b = boot_deltas_kb - obs_deltas_k
            # where boot_deltas_kb = (boot_perf_arr_kb - benchmark_performance)
            
            # Let's use the formulation: statistic is (perf_k - benchmark)
            # Bootstrap distribution of (perf_k_b - perf_k_obs)
            boot_centered_diffs = boot_perf_arr - obs_perf_arr[:, np.newaxis]

        else: # lower is better (e.g. risk)
            obs_deltas = benchmark_performance - obs_perf_arr
            # boot_centered_diffs = (benchmark_performance - boot_perf_arr) - (benchmark_performance - obs_perf_arr[:, np.newaxis])
            # boot_centered_diffs = obs_perf_arr[:, np.newaxis] - boot_perf_arr
            boot_centered_diffs = obs_perf_arr[:, np.newaxis] - boot_perf_arr


        # Initialize
        active_indices = list(range(num_rules))
        rejected_indices = []
        p_values_stepwise = np.full(num_rules, np.nan)

        for _iter in range(min(iterations, num_rules)): # Iterate at most num_rules times
            if not active_indices:
                break

            # Current set of active hypotheses
            current_obs_deltas = obs_deltas[active_indices]
            current_boot_centered_diffs = boot_centered_diffs[active_indices, :] # Shape (L_active, B)
            
            # Max statistic over active hypotheses for each bootstrap sample
            # V_b = max_{k in active_indices} (boot_centered_diffs_kb)
            max_boot_stats_b = np.max(current_boot_centered_diffs, axis=0) # Shape (B,)

            # Calculate p-values for currently active hypotheses
            # p_k = P(max_boot_stats_b >= obs_deltas_k)
            current_p_values = np.zeros(len(active_indices))
            for i, k_idx in enumerate(active_indices):
                obs_delta_k = obs_deltas[k_idx] # This is t_k
                count_extreme = np.sum(max_boot_stats_b >= obs_delta_k)
                current_p_values[i] = (count_extreme + 1) / (num_bootstrap_samples + 1)
                current_p_values[i] = min(current_p_values[i], 1.0)

            # Find minimum p-value among active hypotheses
            min_p_val_current_step = np.min(current_p_values)
            
            # Store this p-value for all hypotheses that were active at this step
            # and haven't been assigned a p-value yet.
            # This is a simplification; true RW p-values are more complex to assign post-hoc.
            # For now, this p-value is the critical one for this step.
            for i, k_idx in enumerate(active_indices):
                if np.isnan(p_values_stepwise[k_idx]):
                     p_values_stepwise[k_idx] = min_p_val_current_step


            # If min_p_val > alpha, stop. No more rejections.
            if min_p_val_current_step > alpha:
                break
            
            # Else, reject hypotheses whose individual p-value (using max_boot_stats_b) <= alpha
            # This is slightly different from pure step-down where only one (the max) is rejected.
            # Romano-Wolf allows rejecting all hypotheses for which p_k <= alpha at this stage.
            newly_rejected_this_step_local_indices = []
            for i, k_idx in enumerate(active_indices):
                # Re-calculate p-value for *this specific* hypothesis k against the max_boot_stats_b distribution
                # This is essentially what current_p_values[i] is.
                if current_p_values[i] <= alpha:
                    original_idx = active_indices[i]
                    if original_idx not in rejected_indices: # Ensure not already rejected
                        rejected_indices.append(original_idx)
                        # The p-value for a rejected hypothesis is the one that led to its rejection.
                        p_values_stepwise[original_idx] = current_p_values[i] 
                        newly_rejected_this_step_local_indices.append(i)

            if not newly_rejected_this_step_local_indices:
                # Should not happen if min_p_val_current_step <= alpha, but as a safeguard
                break

            # Update active_indices: remove newly rejected hypotheses
            active_indices = [idx for i, idx in enumerate(active_indices) if i not in newly_rejected_this_step_local_indices]
            
        # For hypotheses not rejected, their p-value is the last min_p_val_current_step that was > alpha,
        # or the p-value calculated at the last step if iterations ran out.
        # The current p_values_stepwise assignment handles this reasonably.
        # Any remaining NaNs mean they were never tested for rejection (e.g. if iterations were too few)
        # or were part of a later step that didn't lead to rejection.
        # A more rigorous assignment for non-rejected p-values would be the smallest alpha for which they *would* be rejected.

        # Sort rejected_indices for consistency
        rejected_indices.sort()

        return {
            'rejected_hypotheses_indices': rejected_indices,
            'p_values_stepwise': p_values_stepwise.tolist(), # Convert numpy array to list
            'num_rules': num_rules,
            'num_bootstrap_samples': num_bootstrap_samples,
            'alpha': alpha,
            'info': f"Completed {min(iterations, num_rules)} iterations." if active_indices else "All active hypotheses processed or rejected."
        }

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
        
        # Default parameters for permutation sign test, can be made configurable if needed
        permutation_sign_test_params = {
            "n_permutations": 10000,
            "alternative": "two-sided",
            "random_seed": 42 # For reproducibility of this specific test run
        }

        results = {
            'empirical_p_values': self.empirical_p_values(simulated_stats),
            'kolmogorov_smirnov': self.kolmogorov_smirnov_test(simulated_stats, original_stats),
            'anderson_darling': self.anderson_darling_test(simulated_stats),
            'jarque_bera': self.jarque_bera_test(simulated_stats),
            'mann_whitney': self.mann_whitney_test(simulated_stats),
            'autocorrelation': self.autocorrelation_tests(self.ret_series.values),
            'arch_test': self.arch_test(self.ret_series.values),
            'permutation_sign_test': self.permutation_test_on_signs(**permutation_sign_test_params)
        }
        
        # Apply multiple comparison corrections
        if results['empirical_p_values']:
            p_vals = {metric: data['p_two_sided'] for metric, data in results['empirical_p_values'].items()}
            results['bonferroni_corrected'] = self.multiple_comparison_correction(p_vals, 'bonferroni')
            results['fdr_corrected'] = self.multiple_comparison_correction(p_vals, 'fdr_bh')
        
        return results
    
    def _circular_block_permutation(self, data: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
        """
        Performs a circular block permutation on the data.
        Blocks are non-overlapping. If n is not a multiple of block_length,
        the last block will be shorter.
        A more advanced version could use overlapping blocks.
        """
        n = len(data)
        if block_length <= 0 or block_length >= n: # block_length == n is just one block
            # Fallback to IID permutation if block_length is invalid or trivial
            return rng.permutation(data)

        num_complete_blocks = n // block_length
        
        blocks = []
        for i in range(num_complete_blocks):
            blocks.append(data[i * block_length : (i + 1) * block_length])
        
        remainder_len = n % block_length
        if remainder_len > 0:
            # Add the remainder as the last block
            blocks.append(data[num_complete_blocks * block_length:])

        permuted_block_indices = rng.permutation(len(blocks))
        
        permuted_data_list = [blocks[i] for i in permuted_block_indices]
        
        return np.concatenate(permuted_data_list)

    def permutation_test_on_metric(
        self,
        metric_calculator: Callable[[pd.Series], float],
        n_permutations: int = 1000,
        permutation_method: str = 'iid', # 'iid' or 'circular_block'
        block_length: Optional[int] = None,
        alternative: str = 'greater', # 'greater', 'less', 'two-sided'
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Performs a permutation test for a given performance metric.

        Tests the null hypothesis that the observed metric on the original series
        is not significantly different from what would be expected by chance,
        as determined by permuting the series.

        Args:
            metric_calculator: A function that takes a pd.Series of returns
                               and returns a single float metric.
            n_permutations: Number of permutations to generate.
            permutation_method: 'iid' for independent shuffle, 
                                'circular_block' for circular block permutation.
            block_length: Required if permutation_method is 'circular_block'.
                          Specifies the length of blocks.
            alternative: Defines the alternative hypothesis.
                         'greater': observed metric is greater than random.
                         'less': observed metric is less than random.
                         'two-sided': observed metric is different from random.
            random_seed: Optional seed for reproducibility.

        Returns:
            Dictionary with test results:
                'observed_metric': The metric calculated on the original series.
                'permuted_metrics': List of metrics from permuted series.
                'p_value': The calculated p-value.
                'alternative': The alternative hypothesis tested.
                'n_permutations': Number of permutations used.
                'permutation_method': Method used for permutation.
        """
        if self.ret_series is None or self.ret_series.empty:
            warnings.warn("Return series is empty. Cannot perform permutation test on metric.")
            return {
                'observed_metric': np.nan,
                'permuted_metrics': [],
                'p_value': np.nan,
                'alternative': alternative,
                'n_permutations': n_permutations,
                'permutation_method': permutation_method,
                'info': "Return series is empty."
            }

        original_data = self.ret_series.values
        observed_metric = metric_calculator(self.ret_series.copy()) # Pass a copy

        if np.isnan(observed_metric):
             warnings.warn("Observed metric is NaN. Check metric_calculator or input data.")
        
        rng = np.random.default_rng(seed=random_seed)
        permuted_metrics = np.zeros(n_permutations, dtype=float)

        if permutation_method == 'circular_block' and (block_length is None or block_length <= 0):
            raise ValueError("block_length must be a positive integer for circular_block permutation.")

        for i in range(n_permutations):
            if permutation_method == 'iid':
                permuted_data = rng.permutation(original_data)
            elif permutation_method == 'circular_block':
                permuted_data = self._circular_block_permutation(original_data, block_length, rng)
            else:
                raise ValueError(f"Unknown permutation_method: {permutation_method}")
            
            permuted_series = pd.Series(permuted_data, index=self.ret_series.index) # Keep original index for context
            permuted_metrics[i] = metric_calculator(permuted_series)

        # Calculate p-value
        if alternative == 'greater':
            count_extreme = np.sum(permuted_metrics >= observed_metric)
        elif alternative == 'less':
            count_extreme = np.sum(permuted_metrics <= observed_metric)
        elif alternative == 'two-sided':
            # Deviation from the mean of permuted metrics
            mean_permuted = np.nanmean(permuted_metrics)
            observed_deviation = abs(observed_metric - mean_permuted)
            permuted_deviations = np.abs(permuted_metrics - mean_permuted)
            count_extreme = np.sum(permuted_deviations >= observed_deviation)
        else:
            raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'.")

        p_value = (count_extreme + 1) / (n_permutations + 1)
        p_value = min(p_value, 1.0)

        return {
            'observed_metric': observed_metric,
            'permuted_metrics': permuted_metrics.tolist(),
            'p_value': p_value,
            'alternative': alternative,
            'n_permutations': n_permutations,
            'permutation_method': permutation_method,
            'block_length_used': block_length if permutation_method == 'circular_block' else None,
            'info': None
        }
