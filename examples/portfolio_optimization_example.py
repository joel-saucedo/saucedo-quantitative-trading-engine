"""
Portfolio optimization example.

This example demonstrates:
1. Mean-variance optimization
2. Risk parity optimization
3. Black-Litterman model
4. Hierarchical risk parity
5. Portfolio rebalancing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Import framework modules
from src.analysis.portfolio_analyzer import PortfolioAnalyzer
from src.utils.data_loader import generate_synthetic_data
from src.utils.portfolio import (
    PortfolioUtils, rebalance_portfolio, risk_budget_portfolio,
    maximum_diversification_portfolio, calculate_optimal_portfolio_size
)


def main():
    """Run portfolio optimization example."""
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n1. Generating sample data...")
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Create data for different asset classes
    data = generate_synthetic_data(
        start_date=start_date,
        end_date=end_date,
        assets=['US_STOCKS', 'INTL_STOCKS', 'BONDS', 'COMMODITIES', 'REITS'],
        initial_price=100.0,
        volatility=0.18,
        drift=0.07
    )
    
    # Add some correlation structure
    returns = data.pct_change().dropna()
    
    print(f"Data shape: {returns.shape}")
    print(f"Assets: {list(returns.columns)}")
    print(f"Sample period: {returns.index[0]} to {returns.index[-1]}")
    
    # Display basic statistics
    print(f"\nAsset Statistics:")
    stats = pd.DataFrame({
        'Mean Return': returns.mean() * 252,
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    })
    print(stats.round(3))
    
    # 2. Initialize portfolio analyzer
    print("\n2. Setting up portfolio analyzer...")
    
    analyzer = PortfolioAnalyzer()
    
    # 3. Mean-variance optimization
    print("\n3. Mean-variance optimization...")
    
    # Target return optimization
    target_return = 0.08  # 8% target return
    mv_result = analyzer.mean_variance_optimization(
        returns=returns,
        target_return=target_return,
        risk_free_rate=0.02
    )
    
    print(f"Mean-Variance Optimization (Target Return: {target_return:.1%}):")
    print(f"  - Expected Return: {mv_result['expected_return']:.2%}")
    print(f"  - Expected Risk: {mv_result['expected_risk']:.2%}")
    print(f"  - Sharpe Ratio: {mv_result['sharpe_ratio']:.3f}")
    print(f"  - Weights:")
    for asset, weight in mv_result['weights'].items():
        print(f"    {asset}: {weight:.1%}")
    
    # Maximum Sharpe ratio portfolio
    max_sharpe_result = analyzer.mean_variance_optimization(
        returns=returns,
        objective='max_sharpe',
        risk_free_rate=0.02
    )
    
    print(f"\nMaximum Sharpe Portfolio:")
    print(f"  - Expected Return: {max_sharpe_result['expected_return']:.2%}")
    print(f"  - Expected Risk: {max_sharpe_result['expected_risk']:.2%}")
    print(f"  - Sharpe Ratio: {max_sharpe_result['sharpe_ratio']:.3f}")
    
    # 4. Risk parity optimization
    print("\n4. Risk parity optimization...")
    
    rp_result = analyzer.risk_parity_optimization(returns=returns)
    
    print(f"Risk Parity Portfolio:")
    print(f"  - Expected Return: {rp_result['expected_return']:.2%}")
    print(f"  - Expected Risk: {rp_result['expected_risk']:.2%}")
    print(f"  - Weights:")
    for asset, weight in rp_result['weights'].items():
        print(f"    {asset}: {weight:.1%}")
    
    # Calculate risk contributions
    weights_array = np.array([rp_result['weights'][asset] for asset in returns.columns])
    cov_matrix = returns.cov().values
    portfolio_var = np.dot(weights_array, np.dot(cov_matrix, weights_array))
    marginal_contrib = np.dot(cov_matrix, weights_array)
    risk_contrib = weights_array * marginal_contrib / portfolio_var
    
    print(f"  - Risk Contributions:")
    for i, asset in enumerate(returns.columns):
        print(f"    {asset}: {risk_contrib[i]:.1%}")
    
    # 5. Black-Litterman optimization
    print("\n5. Black-Litterman optimization...")
    
    # Create some views (example: US stocks will outperform by 2%)
    views = {
        'US_STOCKS': 0.02  # 2% excess return view
    }
    
    bl_result = analyzer.black_litterman_optimization(
        returns=returns,
        views=views,
        view_confidence=0.5,
        risk_free_rate=0.02
    )
    
    print(f"Black-Litterman Portfolio:")
    print(f"  - Expected Return: {bl_result['expected_return']:.2%}")
    print(f"  - Expected Risk: {bl_result['expected_risk']:.2%}")
    print(f"  - Views incorporated: {views}")
    print(f"  - Weights:")
    for asset, weight in bl_result['weights'].items():
        print(f"    {asset}: {weight:.1%}")
    
    # 6. Hierarchical risk parity
    print("\n6. Hierarchical risk parity...")
    
    hrp_result = analyzer.hierarchical_risk_parity(returns=returns)
    
    print(f"Hierarchical Risk Parity Portfolio:")
    print(f"  - Expected Return: {hrp_result['expected_return']:.2%}")
    print(f"  - Expected Risk: {hrp_result['expected_risk']:.2%}")
    print(f"  - Weights:")
    for asset, weight in hrp_result['weights'].items():
        print(f"    {asset}: {weight:.1%}")
    
    # 7. Additional portfolio optimization methods
    print("\n7. Additional optimization methods...")
    
    # Maximum diversification
    max_div_weights = maximum_diversification_portfolio(returns)
    print(f"Maximum Diversification Portfolio:")
    for asset, weight in max_div_weights.items():
        print(f"  {asset}: {weight:.1%}")
    
    # Kelly criterion
    kelly_weights = calculate_optimal_portfolio_size(
        returns=returns,
        risk_free_rate=0.02,
        max_leverage=1.0
    )
    print(f"\nKelly Criterion Portfolio:")
    for asset, weight in kelly_weights.items():
        print(f"  {asset}: {weight:.1%}")
    
    # 8. Portfolio comparison
    print("\n8. Portfolio comparison...")
    
    portfolios = {
        'Equal Weight': {asset: 1.0/len(returns.columns) for asset in returns.columns},
        'Mean-Variance': mv_result['weights'],
        'Risk Parity': rp_result['weights'],
        'Black-Litterman': bl_result['weights'],
        'HRP': hrp_result['weights'],
        'Max Diversification': max_div_weights
    }
    
    # Calculate performance for each portfolio
    portfolio_performance = {}
    
    for name, weights in portfolios.items():
        weights_array = np.array([weights.get(asset, 0) for asset in returns.columns])
        
        # Portfolio returns
        port_returns = (returns * weights_array).sum(axis=1)
        
        # Performance metrics
        total_return = (1 + port_returns).prod() - 1
        annual_return = port_returns.mean() * 252
        volatility = port_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_dd = (port_returns.cumsum() - port_returns.cumsum().expanding().max()).min()
        
        portfolio_performance[name] = {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_dd
        }
    
    # Display comparison
    comparison_df = pd.DataFrame(portfolio_performance).T
    print("\nPortfolio Performance Comparison:")
    print(comparison_df.round(3))
    
    # 9. Portfolio rebalancing example
    print("\n9. Portfolio rebalancing example...")
    
    # Simulate portfolio drift
    current_weights = {'US_STOCKS': 0.4, 'INTL_STOCKS': 0.3, 'BONDS': 0.2, 'COMMODITIES': 0.05, 'REITS': 0.05}
    target_weights = mv_result['weights']
    
    # Current prices (last day)
    current_prices = {asset: data[asset].iloc[-1] for asset in data.columns}
    portfolio_value = 100000  # $100k portfolio
    
    rebalance_result = rebalance_portfolio(
        current_weights=current_weights,
        target_weights=target_weights,
        prices=current_prices,
        portfolio_value=portfolio_value,
        transaction_cost=0.001  # 0.1% transaction cost
    )
    
    print(f"Rebalancing Analysis:")
    print(f"  - Rebalancing successful: {rebalance_result.success}")
    print(f"  - Turnover: {rebalance_result.turnover:.1%}")
    print(f"  - Transaction cost: ${rebalance_result.cost:.2f}")
    print(f"  - Transactions needed:")
    for asset, amount in rebalance_result.transactions.items():
        if abs(amount) > 0.01:  # Show only significant transactions
            action = "Buy" if amount > 0 else "Sell"
            print(f"    {action} {abs(amount):.2f} shares of {asset}")
    
    # 10. Create visualizations
    print("\n10. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Portfolio weights comparison
    weight_comparison = pd.DataFrame(portfolios).T
    weight_comparison.plot(kind='bar', ax=axes[0, 0], stacked=True)
    axes[0, 0].set_title('Portfolio Weights Comparison')
    axes[0, 0].set_ylabel('Weight')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Risk-return scatter
    for name, perf in portfolio_performance.items():
        axes[0, 1].scatter(perf['Volatility'], perf['Annual Return'], 
                          s=100, label=name, alpha=0.7)
    axes[0, 1].set_xlabel('Volatility')
    axes[0, 1].set_ylabel('Annual Return')
    axes[0, 1].set_title('Risk-Return Profile')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sharpe ratios
    sharpe_ratios = [perf['Sharpe Ratio'] for perf in portfolio_performance.values()]
    portfolio_names = list(portfolio_performance.keys())
    axes[0, 2].bar(portfolio_names, sharpe_ratios)
    axes[0, 2].set_title('Sharpe Ratio Comparison')
    axes[0, 2].set_ylabel('Sharpe Ratio')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Cumulative returns
    for name, weights in portfolios.items():
        weights_array = np.array([weights.get(asset, 0) for asset in returns.columns])
        port_returns = (returns * weights_array).sum(axis=1)
        cumulative_returns = (1 + port_returns).cumprod()
        axes[1, 0].plot(cumulative_returns.index, cumulative_returns.values, 
                       label=name, linewidth=2)
    
    axes[1, 0].set_title('Cumulative Returns')
    axes[1, 0].set_ylabel('Cumulative Return')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation matrix
    corr_matrix = returns.corr()
    im = axes[1, 1].imshow(corr_matrix.values, cmap='RdYlBu', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
    axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
    axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45)
    axes[1, 1].set_yticklabels(corr_matrix.columns)
    axes[1, 1].set_title('Asset Correlation Matrix')
    plt.colorbar(im, ax=axes[1, 1])
    
    # Efficient frontier
    target_returns = np.linspace(0.03, 0.15, 20)
    efficient_portfolios = []
    
    for target_ret in target_returns:
        try:
            result = analyzer.mean_variance_optimization(
                returns=returns,
                target_return=target_ret,
                risk_free_rate=0.02
            )
            efficient_portfolios.append({
                'return': result['expected_return'],
                'risk': result['expected_risk']
            })
        except:
            continue
    
    if efficient_portfolios:
        efficient_df = pd.DataFrame(efficient_portfolios)
        axes[1, 2].plot(efficient_df['risk'], efficient_df['return'], 
                       'b-', linewidth=2, label='Efficient Frontier')
        
        # Plot individual portfolios
        for name, perf in portfolio_performance.items():
            axes[1, 2].scatter(perf['Volatility'], perf['Annual Return'], 
                              s=100, label=name, alpha=0.7)
        
        axes[1, 2].set_xlabel('Risk (Volatility)')
        axes[1, 2].set_ylabel('Expected Return')
        axes[1, 2].set_title('Efficient Frontier')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = '/home/joelasaucedo/Development/backtesting_engine/examples/portfolio_optimization.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Visualizations saved to: {plot_path}")
    
    # 11. Save results
    print("\n11. Saving results...")
    
    results_dir = '/home/joelasaucedo/Development/backtesting_engine/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save portfolio weights
    weights_df = pd.DataFrame(portfolios)
    weights_df.to_csv(f"{results_dir}/portfolio_weights.csv")
    
    # Save performance comparison
    comparison_df.to_csv(f"{results_dir}/portfolio_performance_comparison.csv")
    
    # Save efficient frontier data
    if efficient_portfolios:
        efficient_df.to_csv(f"{results_dir}/efficient_frontier.csv", index=False)
    
    print(f"Results saved to: {results_dir}")
    
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION EXAMPLE COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
