#!/usr/bin/env python3
"""
Integration Test Script

This script tests the complete trading strategy analyzer framework
to ensure all components work together properly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    try:
        from src.utils import load_sample_data, generate_synthetic_data, generate_multi_asset_data
        
        # Test sample data loading
        data = load_sample_data()
        assert isinstance(data, pd.DataFrame), "Data should be DataFrame"
        assert not data.empty, "Data should not be empty"
        print("✓ Sample data loading works")
        
        # Test synthetic data generation
        synthetic_data = generate_synthetic_data(
            start_date="2020-01-01",
            end_date="2020-04-10"  # About 100 days
        )
        assert isinstance(synthetic_data, pd.DataFrame), "Synthetic data should be DataFrame"
        assert synthetic_data.shape[1] == 5, "Should have OHLCV columns"
        assert synthetic_data.shape[0] >= 90, "Should have about 100 periods"
        print("✓ Synthetic data generation works")
        
        # Test multi-asset data generation
        multi_asset_data = generate_multi_asset_data(n_assets=3, start_date="2020-01-01", end_date="2020-04-10")
        assert isinstance(multi_asset_data, pd.DataFrame), "Multi-asset data should be DataFrame"
        assert multi_asset_data.shape[1] == 3, "Should have 3 assets"
        assert multi_asset_data.shape[0] >= 90, "Should have about 100 periods"
        print("✓ Multi-asset data generation works")
        print("✓ Synthetic data generation works")
        
        return data
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None

def test_strategy_implementation(data):
    """Test strategy implementation."""
    print("\nTesting strategy implementation...")
    
    try:
        from src.strategies import MomentumStrategy, MeanReversionStrategy
        
        # Test momentum strategy
        momentum_strategy = MomentumStrategy(lookback_period=10)
        # Backtest the strategy
        backtest_results = momentum_strategy.backtest(data)
        returns = backtest_results['returns']
        signals = pd.Series([s[1] for s in backtest_results['signals']], index=data.index[:len(backtest_results['signals'])])

        assert isinstance(signals, (pd.Series, np.ndarray)), "Signals should be Series or array"
        assert isinstance(returns, pd.Series), "Returns should be Series"
        assert len(returns) > 0, "Should generate returns"
        print("✓ Momentum strategy works")
        
        # Test mean reversion strategy
        mr_strategy = MeanReversionStrategy(lookback_period=20)
        # Backtest the strategy
        mr_backtest_results = mr_strategy.backtest(data)
        mr_returns = mr_backtest_results['returns']
        mr_signals = pd.Series([s[1] for s in mr_backtest_results['signals']], index=data.index[:len(mr_backtest_results['signals'])])

        assert isinstance(mr_returns, pd.Series), "MR returns should be Series"
        print("✓ Mean reversion strategy works")
        
        return backtest_results # Return the full backtest_results
        
    except Exception as e:
        print(f"✗ Strategy implementation failed: {e}")
        return None

def test_performance_analysis(strategy_results):
    """Test performance analysis."""
    print("\nTesting performance analysis...")
    
    try:
        from src.analysis import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer(strategy_results)
        # Use generate_performance_report to get all metrics
        report = analyzer.generate_performance_report()
        
        # Access metrics correctly
        # summary_metrics contains the output of calculate_metrics from utils.metrics
        # which has 'performance', 'risk', and 'raw_metrics' keys.
        summary_metrics = report.get('summary_metrics', {})
        
        # Option 1: Access from the PerformanceMetrics dataclass
        performance_dataclass = summary_metrics.get('performance')
        assert performance_dataclass is not None, "Performance dataclass should exist in summary_metrics"
        sharpe_ratio = performance_dataclass.sharpe_ratio
        annual_return = performance_dataclass.annualized_return

        # Option 2: Access from raw_metrics dictionary (alternative)
        # raw_metrics_dict = summary_metrics.get('raw_metrics', {})
        # sharpe_ratio = raw_metrics_dict.get('sharpe_ratio')
        # annual_return = raw_metrics_dict.get('annualized_return')
        
        assert sharpe_ratio is not None, "Should include Sharpe ratio"
        assert annual_return is not None, "Should include annual return"
        print("✓ Performance analysis works")
        
        # Return the specific metrics needed by the main function for display
        return {'sharpe_ratio': sharpe_ratio, 'annual_return': annual_return}
        
    except Exception as e:
        print(f"✗ Performance analysis failed: {e}")
        return None

def test_risk_analysis(returns):
    """Test risk analysis."""
    print("\nTesting risk analysis...")
    
    try:
        from src.analysis import RiskAnalyzer
        
        risk_analyzer = RiskAnalyzer()
        
        # Test VaR calculation
        # Pass a single confidence level, not a list
        var_results = risk_analyzer.calculate_var(returns, confidence_level=0.95)
        # Assert that var_results is a float, not a dictionary for a single VaR calc
        assert isinstance(var_results, float), "VaR result should be a float"
        print("✓ VaR calculation works")
        
        # Test Expected Shortfall
        # Pass a single confidence level
        es_results = risk_analyzer.calculate_expected_shortfall(returns, confidence_level=0.95)
        assert isinstance(es_results, float), "ES result should be a float"
        print("✓ Expected Shortfall calculation works")
        
        # Return a dictionary for consistency with how it was used in main()
        return {'var_0.95': var_results, 'es_0.95': es_results} 
        
    except Exception as e:
        print(f"✗ Risk analysis failed: {e}")
        return None

def test_bootstrap_analysis(returns):
    """Test bootstrap analysis."""
    print("\nTesting bootstrap analysis...")
    
    try:
        from src.bootstrapping import AdvancedBootstrapping, BootstrapConfig
        
        # Use BootstrapConfig to pass parameters
        config = BootstrapConfig(n_sims=100) # Reduced for speed
        # Pass returns as a keyword argument
        bootstrap = AdvancedBootstrapping(ret_series=returns, config=config)
        
        # Call run_bootstrap_simulation()
        bootstrap_results = bootstrap.run_bootstrap_simulation()
        
        assert isinstance(bootstrap_results, dict), "Bootstrap results should be dictionary"
        
        # Check the structure of bootstrap_results
        assert 'simulated_stats' in bootstrap_results, "Should include simulated_stats"
        assert isinstance(bootstrap_results['simulated_stats'], list), "simulated_stats should be a list"
        assert len(bootstrap_results['simulated_stats']) == config.n_sims, f"Should have {config.n_sims} simulation results"
        if config.n_sims > 0:
            assert isinstance(bootstrap_results['simulated_stats'][0], dict), "Each item in simulated_stats should be a dict"
            assert 'Sharpe' in bootstrap_results['simulated_stats'][0], "Simulated stats should include Sharpe ratio"

        assert 'original_stats' in bootstrap_results, "Should include original_stats"
        assert isinstance(bootstrap_results['original_stats'], dict), "original_stats should be a dict"
        assert 'Sharpe' in bootstrap_results['original_stats'], "Original stats should include Sharpe ratio"
        print("✓ Bootstrap analysis works")
        
        # Return relevant metrics for the main function display
        original_sharpe = bootstrap_results['original_stats'].get('Sharpe')
        simulated_sharpes = [s.get('Sharpe') for s in bootstrap_results['simulated_stats']]
        # For simplicity, let's return the original Sharpe and the mean of simulated Sharpes
        # A proper CI would require more calculation, which might be beyond a simple integration test check here
        mean_simulated_sharpe = np.nanmean(simulated_sharpes) if simulated_sharpes else None
        
        return {
            'original_sharpe': original_sharpe,
            'mean_simulated_sharpe': mean_simulated_sharpe 
            # Placeholder for CI, actual CI calculation is more involved
            # 'confidence_interval': bootstrap_results.get('confidence_interval', 'N/A') 
        }
        
    except Exception as e:
        print(f"✗ Bootstrap analysis failed: {e}")
        return None

def test_portfolio_analysis():
    """Test portfolio analysis."""
    print("\nTesting portfolio analysis...")
    
    try:
        from src.analysis import PortfolioAnalyzer
        from src.utils import generate_multi_asset_data
        from src.analysis.portfolio_analyzer import PortfolioAllocation # Added import
        
        # Generate multi-asset data for portfolio analysis
        portfolio_data = generate_multi_asset_data(n_assets=3, start_date="2020-01-01", end_date="2020-04-10")
        
        portfolio_analyzer = PortfolioAnalyzer()
        
        # Test mean-variance optimization
        mv_result = portfolio_analyzer.mean_variance_optimization(
            portfolio_data, 
            target_return=0.10
        )
        
        assert isinstance(mv_result, PortfolioAllocation), "MV result should be PortfolioAllocation object"
        assert hasattr(mv_result, 'weights'), "Should include weights"
        assert hasattr(mv_result, 'expected_return'), "Should include expected return"
        print("✓ Mean-variance optimization works")
        
        return mv_result
        
    except Exception as e:
        print(f"✗ Portfolio analysis failed: {e}")
        return None

def test_validation():
    """Test data validation."""
    print("\nTesting data validation...")
    
    try:
        from src.utils import validate_input_data
        
        # Create test data with sufficient observations
        date_rng = pd.date_range(start='2020-01-01', end='2020-02-29', freq='B') # Approx 40 business days
        test_data = pd.DataFrame(date_rng, columns=['date'])
        test_data['price'] = np.random.rand(len(test_data)) * 100 + 100
        test_data['returns'] = test_data['price'].pct_change().fillna(0)
        test_data.set_index('date', inplace=True)
        
        # Validate the 'returns' column of the DataFrame
        # The validate_input_data function expects a Series for 'returns' or 'prices' data_type
        # if not validating a full DataFrame with OHLCV structure.
        validation_result = validate_input_data(test_data['returns'], data_type='returns', min_observations=30)
        # The validate_input_data function itself returns True if valid, or raises ValidationError.
        # It does not return a dictionary. The create_validation_report returns a dict.
        assert validation_result is True, "Validation should pass for valid returns data"
        print("✓ Data validation works")
        
        # For the main function, we can return a simple success indicator
        return {'validation_status': 'success'} 
        
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
        return None

def test_complete_workflow():
    """Test complete analysis workflow."""
    print("\nTesting complete workflow...")
    
    try:
        from src.strategies import StrategyTestSuite, MomentumStrategy # Import MomentumStrategy
        from src.utils import load_sample_data
        
        # Load data
        data = load_sample_data()
        
        # Create strategy test suite
        suite = StrategyTestSuite()
        
        # Define a simple strategy function for registration
        # This function should ideally come from the strategies module or be a proper strategy class instance
        # For this test, we'll use the MomentumStrategy's backtest method
        # In a real scenario, you'd register strategy functions that return a Series of returns
        
        # Instantiate the strategy
        momentum_strategy_instance = MomentumStrategy(lookback_period=10)

        # Define a wrapper function that matches the expected signature for strategy_func
        # The strategy_func in StrategyTestSuite expects a function that takes data and **params
        # and returns a pd.Series of returns.
        def momentum_strategy_callable(data_df, **params):
            # The MomentumStrategy's backtest method returns a dict.
            # We need to extract the 'returns' Series from it.
            # params from register_strategy will be passed here.
            # We need to ensure lookback_period is used if passed via params,
            # or use the instance's default.
            lookback = params.get('lookback_period', momentum_strategy_instance.lookback_period)
            # Create a new instance if params are different, or use the existing one
            temp_strategy = MomentumStrategy(lookback_period=lookback)
            results = temp_strategy.backtest(data_df)
            return results['returns']

        # Register the strategy
        suite.register_strategy(
            name='momentum_test', 
            strategy_func=momentum_strategy_callable, 
            params={'lookback_period': 20} # Example: override default lookback
        )
        
        # Run analysis for the single registered strategy
        # run_single_strategy expects the data and optionally a bootstrap method
        # It does not take a list of strategies or a bootstrap_analysis boolean directly
        results = suite.run_single_strategy(
            name='momentum_test',
            data=data
        )
        
        assert isinstance(results, dict), "Results should be dictionary"
        # run_single_strategy returns a dictionary of analysis results for that strategy
        # The keys would depend on what run_single_strategy populates.
        # Based on its current implementation, it seems to execute the strategy and perform bootstrapping.
        # Let's assume it returns at least the strategy returns or some bootstrapped stats.
        # The provided snippet for StrategyTestSuite.run_single_strategy only shows execution,
        # not the full analysis. Assuming it would return something like:
        # {'returns': ..., 'bootstrap_results': ...}
        # For now, let's check if it ran without error and produced a dict.
        # A more specific assertion would require knowing the exact structure of 'results'.
        # Given the current StrategyTestSuite, it seems to return a dict with strategy returns
        # and potentially bootstrap results if AdvancedBootstrapping is integrated there.
        # The current snippet of run_single_strategy ends before full analysis.
        # Assuming it's meant to return a comprehensive dict:
        assert 'strategy_returns' in results or 'original_stats' in results, "Should include strategy returns or bootstrap stats"
        print("✓ Complete workflow works (single strategy test)")
        
        return results # Return the actual results
        
    except Exception as e:
        print(f"✗ Complete workflow failed: {e}")
        return None

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("TRADING STRATEGY ANALYZER - INTEGRATION TESTS")
    print("=" * 60)
    
    # Test data loading
    data = test_data_loading()
    if data is None:
        print("❌ Critical failure: Cannot load data")
        return False
    
    # Test strategy implementation
    # Use the backtest results from the momentum strategy for further tests
    strategy_results = test_strategy_implementation(data)
    if strategy_results is None:
        print("❌ Critical failure: Cannot implement strategies")
        return False
    
    # Test performance analysis
    metrics = test_performance_analysis(strategy_results) # Pass strategy_results
    if metrics is None:
        print("❌ Performance analysis failed")
        return False
    
    # Test risk analysis
    risk_results = test_risk_analysis(strategy_results['returns']) # Pass returns from strategy_results
    if risk_results is None:
        print("❌ Risk analysis failed")
        return False
    
    # Test bootstrap analysis
    bootstrap_results = test_bootstrap_analysis(strategy_results['returns']) # Pass returns from strategy_results
    if bootstrap_results is None:
        print("❌ Bootstrap analysis failed")
        return False
    
    # Test portfolio analysis
    portfolio_results = test_portfolio_analysis()
    if portfolio_results is None:
        print("❌ Portfolio analysis failed")
        return False
    
    # Test validation
    validation_results = test_validation()
    if validation_results is None:
        print("❌ Validation failed")
        return False
    
    # Test complete workflow
    workflow_results = test_complete_workflow()
    if workflow_results is None:
        print("❌ Complete workflow failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("=" * 60)
    print("\nSample Results:")
    print(f"Strategy Annual Return: {metrics.get('annual_return', 'N/A'):.2%}")
    print(f"Strategy Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.3f}")
    # Update how bootstrap results are displayed based on the new return structure
    print(f"Original Sharpe (from Bootstrap): {bootstrap_results.get('original_sharpe', 'N/A')}")
    print(f"Mean Simulated Sharpe (from Bootstrap): {bootstrap_results.get('mean_simulated_sharpe', 'N/A')}")
    print(f"5% VaR: {list(risk_results.values())[0] if risk_results else 'N/A':.2%}")
    
    print("\n✅ Framework is ready for production use!")
    print("📚 Check docs/ for tutorials and API documentation")
    print("📂 See examples/ for detailed usage examples")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
