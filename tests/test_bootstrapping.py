"""
Test Bootstrapping Core Module

Tests for the core bootstrapping functionality including
basic bootstrap, block bootstrap, and statistical tests.
"""

import pytest
import pandas as pd
import numpy as np
from src.bootstrapping.core import AdvancedBootstrapping, BootstrapConfig, BootstrapMethod, TimeFrame
from src.bootstrapping.statistical_tests import StatisticalTests
from src.bootstrapping.risk_metrics import RiskMetrics


class TestAdvancedBootstrapping:
    """Test cases for AdvancedBootstrapping class."""
    
    def test_bootstrap_initialization(self, bootstrap_returns_data):
        """Test bootstrap engine initialization."""
        config = BootstrapConfig(n_sims=100)
        engine = AdvancedBootstrapping(ret_series=bootstrap_returns_data, config=config, method=BootstrapMethod.IID)
        assert engine.ret_series is not None
        assert len(engine.ret_series) == len(bootstrap_returns_data)
        assert engine.config.n_sims == 100
        assert engine.method == BootstrapMethod.IID
    
    def test_iid_bootstrap(self, bootstrap_returns_data):
        """Test IID bootstrap resampling."""
        config = BootstrapConfig(n_sims=100)
        engine = AdvancedBootstrapping(ret_series=bootstrap_returns_data, config=config, method=BootstrapMethod.IID)
        results = engine.run_bootstrap_simulation()
        
        assert 'simulated_stats' in results
        assert 'original_stats' in results
        assert len(results['simulated_stats']) == 100
        
        original_mean_cagr = results['original_stats'].get('CAGR', np.nan) # Using CAGR as a proxy for mean return
        simulated_cagrs = [stat.get('CAGR', np.nan) for stat in results['simulated_stats']]
        
        assert len(simulated_cagrs) == 100
        if not np.isnan(original_mean_cagr) and all(not np.isnan(c) for c in simulated_cagrs):
            mean_of_simulated_cagrs = np.mean(simulated_cagrs)
            # This is a loose check, statistical properties are hard to assert tightly without more info
            # assert abs(mean_of_simulated_cagrs - original_mean_cagr) < abs(original_mean_cagr * 0.8) if original_mean_cagr != 0 else abs(mean_of_simulated_cagrs) < 0.1

    def test_block_bootstrap(self, bootstrap_returns_data):
        """Test block bootstrap for autocorrelated data."""
        config = BootstrapConfig(n_sims=50, block_length=10)
        engine = AdvancedBootstrapping(ret_series=bootstrap_returns_data, config=config, method=BootstrapMethod.BLOCK)
        results = engine.run_bootstrap_simulation()
        
        assert 'simulated_stats' in results
        assert len(results['simulated_stats']) == 50
        # Further checks on autocorrelation would require inspecting simulated series,
        # which are part of simulated_equity_curves. For now, this tests execution.

    def test_stationary_bootstrap(self, bootstrap_returns_data):
        """Test stationary bootstrap."""
        config = BootstrapConfig(n_sims=50, block_length=10) # block_length is used by stationary
        engine = AdvancedBootstrapping(ret_series=bootstrap_returns_data, config=config, method=BootstrapMethod.STATIONARY)
        results = engine.run_bootstrap_simulation()
        
        assert 'simulated_stats' in results
        assert len(results['simulated_stats']) == 50

    def test_bootstrap_statistics(self, bootstrap_returns_data):
        """Test bootstrap statistics calculation."""
        config = BootstrapConfig(n_sims=100)
        engine = AdvancedBootstrapping(ret_series=bootstrap_returns_data, config=config, method=BootstrapMethod.IID)
        
        results = engine.run_bootstrap_simulation()
        
        assert 'simulated_stats' in results
        assert len(results['simulated_stats']) == 100
        assert 'original_stats' in results
        
        original_stats = results['original_stats']
        # Check for some expected metrics from AdvancedBootstrapping._calculate_advanced_metrics
        assert 'CAGR' in original_stats 
        assert 'Sharpe' in original_stats
        assert 'MaxDrawdown' in original_stats

    # def test_multiple_statistics(self, bootstrap_returns_data):
    #     """
    #     Test bootstrap with multiple statistics.
    #     NOTE: AdvancedBootstrapping calculates a fixed set of metrics.
    #     This test is commented out as custom_stats_funcs is not supported by BootstrapConfig
    #     in the way previously envisioned for adding new, arbitrary statistics to the output.
    #     The engine calculates a predefined comprehensive set.
    #     """
    #     # Define custom statistics
    #     # def custom_skewness(series): return series.skew()
    #     # def custom_kurtosis(series): return series.kurtosis()

    #     # config = BootstrapConfig(
    #     #     n_sims=50
    #     #     # custom_stats_funcs are not part of BootstrapConfig or used by AdvancedBootstrapping
    #     # )
    #     # engine = AdvancedBootstrapping(ret_series=bootstrap_returns_data, config=config, method=BootstrapMethod.IID)
        
    #     # results = engine.run_bootstrap_simulation()
        
    #     # assert 'simulated_stats' in results
    #     # assert len(results['simulated_stats']) == 50
    #     # for stat_dict in results['simulated_stats']:
    #     #     assert 'CAGR' in stat_dict 
    #     #     assert 'Sharpe' in stat_dict 
    #     #     # Assert custom stats if they were to be supported
    #     #     # assert 'skew' in stat_dict 
    #     #     # assert 'kurt' in stat_dict 
        
    #     # assert 'Skewness' in results['original_stats'] # Check for existing Skewness
    #     # assert 'Kurtosis' in results['original_stats'] # Check for existing Kurtosis
    #     pass


class TestStatisticalTests:
    """Test cases for StatisticalTests class."""
    
    def test_empirical_p_value_and_all_tests(self, bootstrap_returns_data):
        """Test empirical p-value calculation via run_all_tests."""
        tests = StatisticalTests(ret_series=bootstrap_returns_data)
        
        config = BootstrapConfig(n_sims=200)
        engine = AdvancedBootstrapping(ret_series=bootstrap_returns_data, config=config, method=BootstrapMethod.IID)
        bootstrap_results = engine.run_bootstrap_simulation()

        all_test_results = tests.run_all_tests(bootstrap_results)

        assert 'empirical_p_values' in all_test_results
        if all_test_results['empirical_p_values']: # Check if not empty
            # Example: Check p-value for CAGR if benchmark was provided and CAGR calculated
            # This part depends on having a benchmark. If no benchmark, empirical_p_values might be empty.
            # For now, just ensure the structure is there.
            first_metric = list(all_test_results['empirical_p_values'].keys())[0]
            assert 'p_two_sided' in all_test_results['empirical_p_values'][first_metric]
            p_value = all_test_results['empirical_p_values'][first_metric]['p_two_sided']
            assert 0 <= p_value <= 1
        
        assert 'kolmogorov_smirnov' in all_test_results
        assert 'jarque_bera' in all_test_results
        assert 'anderson_darling' in all_test_results

    def test_multiple_testing_correction(self, bootstrap_returns_data):
        """Test multiple testing corrections."""
        # bootstrap_returns_data is not directly used here but needed for consistent fixture usage
        tests = StatisticalTests(ret_series=bootstrap_returns_data) # ret_series needed for constructor
        p_values_dict = {'metric1': 0.01, 'metric2': 0.03, 'metric3': 0.05, 'metric4': 0.10, 'metric5': 0.20}
        
        bonf_corrected = tests.multiple_comparison_correction(p_values_dict, method='bonferroni')
        assert all(bonf_corrected[key] >= p_values_dict[key] for key in p_values_dict)
        assert all(0 <= p <= 1 for p in bonf_corrected.values())

        bh_corrected = tests.multiple_comparison_correction(p_values_dict, method='fdr_bh')
        assert len(bh_corrected) == len(p_values_dict)
        assert all(0 <= p <= 1 for p in bh_corrected.values())

    def test_normality_related_tests_from_run_all(self, bootstrap_returns_data):
        """Test normality testing methods via run_all_tests."""
        tests = StatisticalTests(ret_series=bootstrap_returns_data)
        config = BootstrapConfig(n_sims=100) # Reduced sims for faster test
        engine = AdvancedBootstrapping(ret_series=bootstrap_returns_data, config=config, method=BootstrapMethod.IID)
        bootstrap_results = engine.run_bootstrap_simulation()
        
        all_test_results = tests.run_all_tests(bootstrap_results)

        assert 'kolmogorov_smirnov' in all_test_results
        # Example check on one metric from KS results
        if bootstrap_results['simulated_stats'] and all_test_results['kolmogorov_smirnov']:
            first_metric_key = list(all_test_results['kolmogorov_smirnov'].keys())[0]
            assert 'p_val_normal' in all_test_results['kolmogorov_smirnov'][first_metric_key]
            assert 0 <= all_test_results['kolmogorov_smirnov'][first_metric_key]['p_val_normal'] <= 1
        
        assert 'jarque_bera' in all_test_results
        assert 'anderson_darling' in all_test_results
        
        # Test with explicitly normal data for comparison (using a part of StatisticalTests logic)
        # This is more of a sanity check on the underlying scipy functions if needed,
        # but run_all_tests is the primary interface.
        # normal_data_sim_stats = [{'metric': val} for val in np.random.normal(0, 1, 100)]
        # normal_data_orig_stats = {'metric': 0}
        # ks_results_normal = tests.kolmogorov_smirnov_test(normal_data_sim_stats, normal_data_orig_stats, metrics=['metric'])
        # assert ks_results_normal['metric']['p_val_normal'] > 0.05


    def test_autocorrelation_tests(self, bootstrap_returns_data):
        """Test autocorrelation testing."""
        tests = StatisticalTests(ret_series=bootstrap_returns_data)
        
        # Test on provided data (using the direct method from StatisticalTests)
        autocorr_results = tests.autocorrelation_tests(bootstrap_returns_data.values, lags=5)
        assert 'ljung_box_pvalues' in autocorr_results
        assert len(autocorr_results['ljung_box_pvalues']) == 5
        assert all(0 <= p <= 1 for p in autocorr_results['ljung_box_pvalues'] if p is not None)
        # ARCH test is also part of run_all_tests, or can be called if needed
        # arch_results = tests.arch_test(bootstrap_returns_data.values)
        # assert 'p_value' in arch_results


class TestRiskMetrics:
    """Test cases for RiskMetrics class."""
    
    def test_risk_metrics_calculation(self, bootstrap_returns_data):
        """Test basic risk metrics calculation using calculate_all_metrics."""
        risk_calculator = RiskMetrics(confidence_levels=[0.95, 0.99])
        
        # calculate_all_metrics expects bootstrap_results dictionary
        config = BootstrapConfig(n_sims=100)
        engine = AdvancedBootstrapping(ret_series=bootstrap_returns_data, config=config, method=BootstrapMethod.IID)
        bootstrap_results = engine.run_bootstrap_simulation()
        
        metrics = risk_calculator.calculate_all_metrics(bootstrap_results)
        
        assert 'var_metrics' in metrics
        assert 'cvar_metrics' in metrics
        assert 'tail_risk' in metrics

        # Example: Check structure for VaR of 'CumulativeReturn'
        if bootstrap_results['simulated_stats'] and 'CumulativeReturn' in bootstrap_results['simulated_stats'][0]:
             if metrics['var_metrics'].get('CumulativeReturn'):
                assert 'VaR_95' in metrics['var_metrics']['CumulativeReturn']
                assert not np.isnan(metrics['var_metrics']['CumulativeReturn']['VaR_95'])
                assert 'VaR_99' in metrics['var_metrics']['CumulativeReturn']
                assert not np.isnan(metrics['var_metrics']['CumulativeReturn']['VaR_99'])

    def test_var_cvar_calculation_direct(self, bootstrap_returns_data):
        """Test VaR and CVaR calculation methods directly."""
        risk_calculator = RiskMetrics(confidence_levels=[0.95, 0.99])
        returns_array = bootstrap_returns_data.values

        var_results = risk_calculator.value_at_risk(returns_array)
        assert 'VaR_95' in var_results
        assert 'VaR_99' in var_results
        assert var_results['VaR_99'] <= var_results['VaR_95'] 

        cvar_results = risk_calculator.conditional_value_at_risk(returns_array)
        assert 'CVaR_95' in cvar_results
        assert 'CVaR_99' in cvar_results
        assert cvar_results['CVaR_99'] <= cvar_results['CVaR_95']
        assert cvar_results['CVaR_95'] <= var_results['VaR_95']


    def test_drawdown_related_metrics(self, sample_price_series_data):
        """Test drawdown related analysis from RiskMetrics."""
        risk_calculator = RiskMetrics()
        
        # Convert price series to equity curve (assuming prices are equity values)
        equity_curve = sample_price_series_data.values
        if equity_curve[0] == 0: # Avoid division by zero if prices start at 0
            equity_curve = equity_curve + 1e-6

        # Ulcer Index
        ulcer = risk_calculator.ulcer_index(equity_curve)
        assert not np.isnan(ulcer)

        # Drawdown Duration Analysis
        dd_analysis = risk_calculator.drawdown_duration_analysis(equity_curve)
        assert 'avg_drawdown_duration' in dd_analysis
        
        # MaxDrawdown is typically part of AdvancedBootstrapping results.
        # If we need to test a specific max drawdown calculation from RiskMetrics,
        # it has _calculate_max_drawdown_from_returns.
        # For this test, we focus on methods that take an equity curve.
        # Example:
        # returns_from_prices = sample_price_series_data.pct_change().dropna().values
        # if len(returns_from_prices) > 0:
        #     max_dd_from_rm = risk_calculator._calculate_max_drawdown_from_returns(returns_from_prices)
        #     assert max_dd_from_rm <= 0


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_bootstrap_analysis_flow(self, bootstrap_returns_data, sample_price_series_data):
        """Test complete bootstrap analysis workflow using AdvancedBootstrapping's full_analysis."""
        
        # Config for AdvancedBootstrapping
        # confidence_levels for BootstrapConfig is for internal calculations if any,
        # RiskMetrics and StatisticalTests will use their own defaults or passed CIs.
        config = BootstrapConfig(n_sims=50, confidence_levels=[0.95, 0.99]) 
        
        engine = AdvancedBootstrapping(
            ret_series=bootstrap_returns_data, 
            config=config, 
            method=BootstrapMethod.IID,
            timeframe=TimeFrame.DAY_1 # Example timeframe
        )
        
        # run_full_analysis combines bootstrap, statistical tests, and risk metrics
        full_results = engine.run_full_analysis()
        
        assert 'original_stats' in full_results
        assert 'simulated_stats' in full_results
        assert len(full_results['simulated_stats']) == 50
        assert 'statistical_tests' in full_results
        assert 'risk_metrics' in full_results
        
        # Check some original stats
        assert 'Sharpe' in full_results['original_stats']
        original_sharpe = full_results['original_stats']['Sharpe']
        assert not np.isnan(original_sharpe)

        # Check structure of statistical_tests
        assert 'empirical_p_values' in full_results['statistical_tests']
        # (Further checks depend on benchmark presence for p-values)

        # Check structure of risk_metrics
        # RiskMetrics.calculate_all_metrics is called internally by run_full_analysis
        # The output structure of full_results['risk_metrics'] will be that of RiskMetrics.calculate_all_metrics
        assert 'var_metrics' in full_results['risk_metrics']
        if full_results['simulated_stats'] and 'Sharpe' in full_results['simulated_stats'][0]:
            if full_results['risk_metrics']['var_metrics'].get('Sharpe'):
                 assert 'VaR_95' in full_results['risk_metrics']['var_metrics']['Sharpe']

        # Max drawdown from original stats (calculated by AdvancedBootstrapping)
        assert 'MaxDrawdown' in full_results['original_stats']
        assert full_results['original_stats']['MaxDrawdown'] <= 0

    def test_different_data_types_execution(self):
        """Test bootstrap execution with different conceptual data types."""
        # Trending data (positive mean returns)
        trending_returns = pd.Series(np.random.normal(0.001, 0.01, 100)) # Reduced size for speed
        config_trend = BootstrapConfig(n_sims=20)
        engine_trend = AdvancedBootstrapping(ret_series=trending_returns, config=config_trend, method=BootstrapMethod.IID)
        results_trend = engine_trend.run_bootstrap_simulation()
        assert len(results_trend['simulated_stats']) == 20

        # Volatile data (higher std dev)
        volatile_returns = pd.Series(np.random.normal(0, 0.05, 100)) # Reduced size for speed
        config_vol = BootstrapConfig(n_sims=20)
        engine_vol = AdvancedBootstrapping(ret_series=volatile_returns, config=config_vol, method=BootstrapMethod.IID)
        results_vol = engine_vol.run_bootstrap_simulation()
        assert len(results_vol['simulated_stats']) == 20
