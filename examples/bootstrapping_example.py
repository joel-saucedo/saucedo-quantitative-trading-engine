"""
Bootstrapping analysis example.

This example demonstrates:
1. Bootstrap resampling of returns
2. Statistical significance testing
3. Risk metric estimation with confidence intervals
4. Bootstrap-based backtesting
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import framework modules
from src.bootstrapping.core import Bootstrap
from src.bootstrapping.statistical_tests import BootstrapTests
from src.bootstrapping.risk_metrics import BootstrapRiskMetrics
from src.bootstrapping.plotting import BootstrapPlotter
from src.strategies.momentum import MomentumStrategy
from src.utils.data_loader import generate_synthetic_data


def main():
    """Run bootstrapping example."""
    print("=" * 60)
    print("BOOTSTRAPPING ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n1. Generating sample data...")
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    data = generate_synthetic_data(
        start_date=start_date,
        end_date=end_date,
        assets=['SPY', 'QQQ', 'IWM'],
        initial_price=100.0,
        volatility=0.15,
        drift=0.08
    )
    
    # Calculate returns
    returns = data.pct_change().dropna()
    portfolio_returns = returns.mean(axis=1)  # Equal weight portfolio
    
    print(f"Data shape: {returns.shape}")
    print(f"Portfolio returns shape: {portfolio_returns.shape}")
    print(f"Sample period: {returns.index[0]} to {returns.index[-1]}")
    
    # 2. Basic bootstrap resampling
    print("\n2. Basic bootstrap resampling...")
    
    bootstrap = Bootstrap(
        method='iid',
        n_bootstrap=1000,
        block_size=None,
        random_state=42
    )
    
    # Bootstrap mean return
    bootstrap_means = bootstrap.bootstrap_statistic(
        portfolio_returns.values,
        statistic=np.mean
    )
    
    print(f"Original mean return: {portfolio_returns.mean():.4f}")
    print(f"Bootstrap mean (average): {bootstrap_means.mean():.4f}")
    print(f"Bootstrap std: {bootstrap_means.std():.4f}")
    
    # 3. Confidence intervals
    print("\n3. Computing confidence intervals...")
    
    # Mean return confidence interval
    mean_ci = bootstrap.confidence_interval(
        portfolio_returns.values,
        statistic=np.mean,
        confidence_level=0.95
    )
    
    print(f"95% CI for mean return: [{mean_ci[0]:.4f}, {mean_ci[1]:.4f}]")
    
    # Sharpe ratio confidence interval
    def sharpe_ratio(returns, risk_free_rate=0.02):
        excess_returns = returns - risk_free_rate / 252
        return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    sharpe_ci = bootstrap.confidence_interval(
        portfolio_returns.values,
        statistic=sharpe_ratio,
        confidence_level=0.95
    )
    
    original_sharpe = sharpe_ratio(portfolio_returns.values)
    print(f"Original Sharpe ratio: {original_sharpe:.3f}")
    print(f"95% CI for Sharpe ratio: [{sharpe_ci[0]:.3f}, {sharpe_ci[1]:.3f}]")
    
    # 4. Statistical tests
    print("\n4. Running statistical tests...")
    
    test_suite = BootstrapTests()
    
    # Test if mean return is significantly different from zero
    mean_test = test_suite.test_statistic(
        portfolio_returns.values,
        statistic=np.mean,
        null_value=0.0,
        alternative='two-sided'
    )
    
    print(f"Mean return test:")
    print(f"  - P-value: {mean_test['p_value']:.4f}")
    print(f"  - Significant at 5%: {mean_test['significant']}")
    
    # Test for autocorrelation
    autocorr_test = test_suite.test_autocorrelation(
        portfolio_returns.values,
        max_lag=5
    )
    
    print(f"Autocorrelation test (lag 1):")
    print(f"  - Autocorrelation: {autocorr_test['autocorrelations'][1]:.4f}")
    print(f"  - P-value: {autocorr_test['p_values'][1]:.4f}")
    
    # 5. Risk metrics with bootstrapping
    print("\n5. Bootstrap risk metrics...")
    
    risk_bootstrap = BootstrapRiskMetrics()
    
    # VaR confidence intervals
    var_results = risk_bootstrap.bootstrap_var(
        portfolio_returns.values,
        confidence_level=0.05,
        n_bootstrap=1000
    )
    
    print(f"VaR (5%) analysis:")
    print(f"  - Point estimate: {var_results['var']:.4f}")
    print(f"  - 95% CI: [{var_results['confidence_interval'][0]:.4f}, {var_results['confidence_interval'][1]:.4f}]")
    
    # Maximum drawdown confidence intervals
    dd_results = risk_bootstrap.bootstrap_max_drawdown(
        portfolio_returns.values,
        n_bootstrap=1000
    )
    
    print(f"Max Drawdown analysis:")
    print(f"  - Point estimate: {dd_results['max_drawdown']:.4f}")
    print(f"  - 95% CI: [{dd_results['confidence_interval'][0]:.4f}, {dd_results['confidence_interval'][1]:.4f}]")
    
    # 6. Block bootstrap for time series
    print("\n6. Block bootstrap analysis...")
    
    block_bootstrap = Bootstrap(
        method='block',
        n_bootstrap=1000,
        block_size=20,  # 20-day blocks
        random_state=42
    )
    
    # Compare IID vs Block bootstrap for autocorrelated data
    iid_means = bootstrap.bootstrap_statistic(
        portfolio_returns.values,
        statistic=np.mean
    )
    
    block_means = block_bootstrap.bootstrap_statistic(
        portfolio_returns.values,
        statistic=np.mean
    )
    
    print(f"Bootstrap standard errors:")
    print(f"  - IID bootstrap: {iid_means.std():.6f}")
    print(f"  - Block bootstrap: {block_means.std():.6f}")
    
    # 7. Strategy bootstrap
    print("\n7. Bootstrap strategy performance...")
    
    # Create strategy
    strategy = MomentumStrategy(
        lookback_period=10,
        holding_period=3,
        top_n=2
    )
    
    # Function to calculate strategy Sharpe ratio
    def strategy_sharpe(price_data):
        try:
            # Convert returns back to prices for strategy
            prices = (1 + pd.DataFrame(price_data.reshape(-1, len(data.columns)), 
                                     columns=data.columns)).cumprod() * 100
            
            # Run strategy
            results = strategy.backtest(prices)
            
            # Calculate Sharpe ratio
            if len(results.returns) > 0:
                excess_returns = results.returns - 0.02/252
                if excess_returns.std() > 0:
                    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return 0.0
            
        except:
            return 0.0
    
    # Bootstrap strategy performance (simplified for demonstration)
    print("  - Running bootstrap strategy analysis...")
    strategy_sharpes = []
    
    for i in range(100):  # Reduced number for speed
        # Bootstrap returns
        boot_returns = block_bootstrap.resample(returns.values)
        boot_returns_df = pd.DataFrame(boot_returns, columns=returns.columns)
        
        # Convert to prices
        boot_prices = (1 + boot_returns_df).cumprod() * 100
        boot_prices.index = pd.date_range(start='2020-01-01', periods=len(boot_prices), freq='D')
        
        try:
            results = strategy.backtest(boot_prices)
            if len(results.returns) > 10:  # Minimum number of observations
                sharpe = sharpe_ratio(results.returns.values)
                if np.isfinite(sharpe):
                    strategy_sharpes.append(sharpe)
        except:
            continue
    
    if strategy_sharpes:
        print(f"  - Bootstrap strategy Sharpe ratios:")
        print(f"    - Mean: {np.mean(strategy_sharpes):.3f}")
        print(f"    - Std: {np.std(strategy_sharpes):.3f}")
        print(f"    - 95% CI: [{np.percentile(strategy_sharpes, 2.5):.3f}, {np.percentile(strategy_sharpes, 97.5):.3f}]")
    
    # 8. Create visualizations
    print("\n8. Creating visualizations...")
    
    plotter = BootstrapPlotter()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot bootstrap distributions
    axes[0, 0].hist(bootstrap_means, bins=50, alpha=0.7, density=True)
    axes[0, 0].axvline(portfolio_returns.mean(), color='red', linestyle='--', label='Original')
    axes[0, 0].set_title('Bootstrap Distribution of Mean Returns')
    axes[0, 0].set_xlabel('Mean Return')
    axes[0, 0].legend()
    
    # Plot Sharpe ratio bootstrap
    sharpe_boots = bootstrap.bootstrap_statistic(
        portfolio_returns.values,
        statistic=sharpe_ratio
    )
    axes[0, 1].hist(sharpe_boots, bins=50, alpha=0.7, density=True)
    axes[0, 1].axvline(original_sharpe, color='red', linestyle='--', label='Original')
    axes[0, 1].set_title('Bootstrap Distribution of Sharpe Ratio')
    axes[0, 1].set_xlabel('Sharpe Ratio')
    axes[0, 1].legend()
    
    # Plot returns over time
    axes[1, 0].plot(portfolio_returns.index, portfolio_returns.cumsum())
    axes[1, 0].set_title('Cumulative Returns')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Cumulative Return')
    
    # Plot rolling Sharpe ratio
    rolling_sharpe = portfolio_returns.rolling(window=252).apply(
        lambda x: sharpe_ratio(x.values) if len(x) == 252 else np.nan
    )
    axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe.values)
    axes[1, 1].set_title('Rolling 1-Year Sharpe Ratio')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = '/home/joelasaucedo/Development/backtesting_engine/examples/bootstrapping_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to: {plot_path}")
    
    # 9. Save detailed results
    print("\n9. Saving results...")
    
    results_dir = '/home/joelasaucedo/Development/backtesting_engine/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save bootstrap results
    bootstrap_results = pd.DataFrame({
        'bootstrap_means': bootstrap_means,
        'bootstrap_sharpes': sharpe_boots
    })
    bootstrap_results.to_csv(f"{results_dir}/bootstrap_distributions.csv", index=False)
    
    # Save confidence intervals
    ci_results = pd.DataFrame({
        'metric': ['mean_return', 'sharpe_ratio', 'var_5pct', 'max_drawdown'],
        'point_estimate': [
            portfolio_returns.mean(),
            original_sharpe,
            var_results['var'],
            dd_results['max_drawdown']
        ],
        'ci_lower': [
            mean_ci[0],
            sharpe_ci[0],
            var_results['confidence_interval'][0],
            dd_results['confidence_interval'][0]
        ],
        'ci_upper': [
            mean_ci[1],
            sharpe_ci[1],
            var_results['confidence_interval'][1],
            dd_results['confidence_interval'][1]
        ]
    })
    ci_results.to_csv(f"{results_dir}/confidence_intervals.csv", index=False)
    
    print(f"Results saved to: {results_dir}")
    
    print("\n" + "=" * 60)
    print("BOOTSTRAPPING EXAMPLE COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
