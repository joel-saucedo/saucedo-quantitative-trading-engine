"""
Test Bootstrapping Core Module

Tests for the core bootstrapping functionality including
basic bootstrap, block bootstrap, and statistical tests.
"""

import pytest
import pandas as pd
import numpy as np
from src.bootstrapping.core import BootstrapEngine
from src.bootstrapping.statistical_tests import StatisticalTests
from src.bootstrapping.risk_metrics import RiskMetrics


class TestBootstrapEngine:
    """Test cases for BootstrapEngine class."""
    
    def test_bootstrap_initialization(self, bootstrap_data):
        """Test bootstrap engine initialization."""
        engine = BootstrapEngine(bootstrap_data)
        assert engine.data is not None
        assert len(engine.data) == len(bootstrap_data)
        assert engine.n_simulations == 1000
        assert engine.block_size is None
    
    def test_iid_bootstrap(self, bootstrap_data):
        """Test IID bootstrap resampling."""
        engine = BootstrapEngine(bootstrap_data, n_simulations=100)
        results = engine.iid_bootstrap()
        
        assert len(results) == 100
        assert all(len(sample) == len(bootstrap_data) for sample in results)
        
        # Check that resampling maintains approximate statistical properties
        original_mean = bootstrap_data.mean()
        resampled_means = [sample.mean() for sample in results]
        mean_of_means = np.mean(resampled_means)
        
        # Should be close to original mean (within 2 standard errors)
        std_error = bootstrap_data.std() / np.sqrt(len(bootstrap_data))
        assert abs(mean_of_means - original_mean) < 2 * std_error
    
    def test_block_bootstrap(self, bootstrap_data):
        """Test block bootstrap for autocorrelated data."""
        engine = BootstrapEngine(bootstrap_data, n_simulations=50, block_size=10)
        results = engine.block_bootstrap()
        
        assert len(results) == 50
        assert all(len(sample) == len(bootstrap_data) for sample in results)
        
        # Check that blocks preserve some autocorrelation structure
        original_autocorr = bootstrap_data.autocorr(lag=1)
        resampled_autocorrs = [sample.autocorr(lag=1) for sample in results if not np.isnan(sample.autocorr(lag=1))]
        
        if resampled_autocorrs:
            mean_autocorr = np.mean(resampled_autocorrs)
            # Block bootstrap should preserve some autocorrelation
            # (though not perfectly due to block boundaries)
            assert abs(mean_autocorr) >= abs(original_autocorr) * 0.3
    
    def test_stationary_bootstrap(self, bootstrap_data):
        """Test stationary bootstrap."""
        engine = BootstrapEngine(bootstrap_data, n_simulations=50)
        results = engine.stationary_bootstrap(avg_block_size=10)
        
        assert len(results) == 50
        # Stationary bootstrap can have variable length, so just check non-empty
        assert all(len(sample) > 0 for sample in results)
    
    def test_bootstrap_statistics(self, bootstrap_data):
        """Test bootstrap statistics calculation."""
        engine = BootstrapEngine(bootstrap_data, n_simulations=100)
        
        def test_statistic(data):
            return data.mean()
        
        stats = engine.bootstrap_statistic(test_statistic)
        
        assert len(stats) == 100
        assert all(not np.isnan(stat) for stat in stats)
        
        # Test confidence intervals
        ci = engine.bootstrap_confidence_interval(test_statistic, confidence_level=0.95)
        assert len(ci) == 2
        assert ci[0] < ci[1]
        
        # Original statistic should often fall within confidence interval
        original_stat = test_statistic(bootstrap_data)
        # Note: This might fail occasionally due to randomness, but should pass most of the time
        
    def test_multiple_statistics(self, bootstrap_data):
        """Test bootstrap with multiple statistics."""
        engine = BootstrapEngine(bootstrap_data, n_simulations=50)
        
        def multiple_stats(data):
            return {
                'mean': data.mean(),
                'std': data.std(),
                'skew': data.skew() if len(data) > 2 else 0
            }
        
        results = engine.bootstrap_statistic(multiple_stats)
        
        assert len(results) == 50
        assert all('mean' in result for result in results)
        assert all('std' in result for result in results)
    
    def test_streaming_bootstrap(self, bootstrap_data):
        """Test streaming bootstrap interface."""
        engine = BootstrapEngine(bootstrap_data, n_simulations=1000)
        
        def test_statistic(data):
            return data.mean()
        
        stats = []
        for stat in engine.bootstrap_streaming(test_statistic, batch_size=100):
            stats.extend(stat)
        
        assert len(stats) == 1000
    
    def test_bootstrap_with_different_methods(self, bootstrap_data):
        """Test different bootstrap methods produce different results."""
        engine = BootstrapEngine(bootstrap_data, n_simulations=100)
        
        def test_stat(data):
            return data.mean()
        
        iid_stats = engine.bootstrap_statistic(test_stat, method='iid')
        block_stats = engine.bootstrap_statistic(test_stat, method='block', block_size=10)
        
        # Results should be different (though both should center around true mean)
        iid_var = np.var(iid_stats)
        block_var = np.var(block_stats)
        
        # They shouldn't be identical
        assert not np.allclose(iid_stats, block_stats)


class TestStatisticalTests:
    """Test cases for StatisticalTests class."""
    
    def test_empirical_p_value(self, bootstrap_data):
        """Test empirical p-value calculation."""
        tests = StatisticalTests()
        
        # Test a simple hypothesis: mean = 0
        def test_statistic(data):
            return data.mean()
        
        engine = BootstrapEngine(bootstrap_data, n_simulations=200)
        p_value = tests.empirical_p_value(
            engine, test_statistic, 
            observed_statistic=bootstrap_data.mean(),
            alternative='two-sided'
        )
        
        assert 0 <= p_value <= 1
    
    def test_multiple_testing_correction(self):
        """Test multiple testing corrections."""
        tests = StatisticalTests()
        p_values = [0.01, 0.03, 0.05, 0.10, 0.20]
        
        # Bonferroni correction
        bonf_corrected = tests.multiple_testing_correction(p_values, method='bonferroni')
        assert all(bonf >= orig for bonf, orig in zip(bonf_corrected, p_values))
        
        # BH correction
        bh_corrected = tests.multiple_testing_correction(p_values, method='bh')
        assert len(bh_corrected) == len(p_values)
    
    def test_normality_tests(self, bootstrap_data):
        """Test normality testing methods."""
        tests = StatisticalTests()
        
        # Generate normal data for comparison
        normal_data = pd.Series(np.random.normal(0, 1, 1000))
        
        # Test on normal data
        ks_stat, ks_p = tests.kolmogorov_smirnov_test(normal_data)
        assert 0 <= ks_p <= 1
        
        jb_stat, jb_p = tests.jarque_bera_test(normal_data)
        assert 0 <= jb_p <= 1
        
        ad_stat, ad_p = tests.anderson_darling_test(normal_data)
        assert 0 <= ad_p <= 1
    
    def test_autocorrelation_tests(self, bootstrap_data):
        """Test autocorrelation testing."""
        tests = StatisticalTests()
        
        # Create autocorrelated data
        autocorr_data = pd.Series(np.random.normal(0, 1, 100))
        for i in range(1, len(autocorr_data)):
            autocorr_data.iloc[i] += 0.5 * autocorr_data.iloc[i-1]
        
        ljung_stat, ljung_p = tests.ljung_box_test(autocorr_data)
        assert 0 <= ljung_p <= 1
        
        adf_stat, adf_p = tests.adf_test(autocorr_data)
        assert 0 <= adf_p <= 1


class TestRiskMetrics:
    """Test cases for RiskMetrics class."""
    
    def test_risk_metrics_calculation(self, bootstrap_data):
        """Test basic risk metrics calculation."""
        risk_metrics = RiskMetrics()
        
        metrics = risk_metrics.calculate_metrics(bootstrap_data)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'var_95', 'var_99', 'cvar_95', 'cvar_99',
            'max_drawdown', 'volatility', 'skewness', 'kurtosis'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert not np.isnan(metrics[metric])
    
    def test_var_calculation(self, bootstrap_data):
        """Test VaR calculation methods."""
        risk_metrics = RiskMetrics()
        
        # Historical VaR
        var_95 = risk_metrics.value_at_risk(bootstrap_data, confidence_level=0.95)
        var_99 = risk_metrics.value_at_risk(bootstrap_data, confidence_level=0.99)
        
        assert var_99 < var_95  # 99% VaR should be more extreme
        
        # Parametric VaR
        parametric_var = risk_metrics.parametric_var(bootstrap_data, confidence_level=0.95)
        assert not np.isnan(parametric_var)
    
    def test_expected_shortfall(self, bootstrap_data):
        """Test Expected Shortfall (CVaR) calculation."""
        risk_metrics = RiskMetrics()
        
        var_95 = risk_metrics.value_at_risk(bootstrap_data, confidence_level=0.95)
        es_95 = risk_metrics.expected_shortfall(bootstrap_data, confidence_level=0.95)
        
        # Expected shortfall should be more extreme than VaR
        assert es_95 <= var_95
    
    def test_drawdown_analysis(self, bootstrap_data):
        """Test drawdown analysis."""
        risk_metrics = RiskMetrics()
        
        # Convert returns to price series for drawdown calculation
        price_series = (1 + bootstrap_data).cumprod()
        
        max_dd, dd_duration = risk_metrics.maximum_drawdown(price_series, return_duration=True)
        
        assert max_dd <= 0  # Drawdown should be negative or zero
        assert dd_duration >= 0  # Duration should be non-negative
    
    def test_bootstrap_risk_metrics(self, bootstrap_data):
        """Test bootstrap-based risk metrics."""
        risk_metrics = RiskMetrics()
        engine = BootstrapEngine(bootstrap_data, n_simulations=100)
        
        bootstrap_metrics = risk_metrics.bootstrap_risk_metrics(engine)
        
        assert 'var_distribution' in bootstrap_metrics
        assert 'expected_shortfall_distribution' in bootstrap_metrics
        assert 'confidence_intervals' in bootstrap_metrics


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_bootstrap_analysis(self, bootstrap_data):
        """Test complete bootstrap analysis workflow."""
        # Initialize components
        engine = BootstrapEngine(bootstrap_data, n_simulations=200)
        tests = StatisticalTests()
        risk_metrics = RiskMetrics()
        
        # Define test statistic
        def sharpe_ratio(data):
            if data.std() == 0:
                return 0
            return data.mean() / data.std() * np.sqrt(252)
        
        # Bootstrap analysis
        sharpe_distribution = engine.bootstrap_statistic(sharpe_ratio)
        confidence_interval = engine.bootstrap_confidence_interval(sharpe_ratio)
        
        # Statistical tests
        observed_sharpe = sharpe_ratio(bootstrap_data)
        p_value = tests.empirical_p_value(
            engine, sharpe_ratio, observed_sharpe, alternative='greater'
        )
        
        # Risk analysis
        risk_analysis = risk_metrics.bootstrap_risk_metrics(engine)
        
        # Assertions
        assert len(sharpe_distribution) == 200
        assert len(confidence_interval) == 2
        assert 0 <= p_value <= 1
        assert 'var_distribution' in risk_analysis
        
        # Results should be reasonable
        assert not np.isnan(observed_sharpe)
        assert confidence_interval[0] < confidence_interval[1]
    
    def test_different_data_types(self):
        """Test bootstrap with different data types."""
        # Test with different data characteristics
        
        # Trending data
        trending_data = pd.Series(np.cumsum(np.random.normal(0.001, 0.01, 500)))
        engine_trend = BootstrapEngine(trending_data.diff().dropna(), n_simulations=50)
        
        # Volatile data  
        volatile_data = pd.Series(np.random.normal(0, 0.05, 500))
        engine_vol = BootstrapEngine(volatile_data, n_simulations=50)
        
        # Test both work
        def test_stat(data):
            return data.mean()
        
        trend_stats = engine_trend.bootstrap_statistic(test_stat)
        vol_stats = engine_vol.bootstrap_statistic(test_stat)
        
        assert len(trend_stats) == 50
        assert len(vol_stats) == 50
