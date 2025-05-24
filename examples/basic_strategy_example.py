"""
Basic strategy example demonstrating the backtesting framework.

This example shows how to:
1. Load data
2. Create and run a simple momentum strategy
3. Analyze performance and risk metrics
4. Generate plots and reports
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import framework modules
from src.strategies.momentum import MomentumStrategy
from src.analysis.performance_analyzer import PerformanceAnalyzer
from src.analysis.risk_analyzer import RiskAnalyzer
from src.utils.data_loader import generate_synthetic_data
from src.utils.metrics import PerformanceMetrics


def main():
    """Run basic strategy example."""
    print("=" * 60)
    print("BASIC STRATEGY EXAMPLE")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic data...")
    
    # Create 2 years of daily data
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    data = generate_synthetic_data(
        start_date=start_date,
        end_date=end_date,
        assets=['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        initial_price=100.0,
        volatility=0.2,
        drift=0.08
    )
    
    print(f"Generated data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Assets: {list(data.columns)}")
    
    # 2. Create and configure strategy
    print("\n2. Setting up momentum strategy...")
    
    strategy = MomentumStrategy(
        lookback_period=20,  # 20-day momentum
        holding_period=5,    # Hold for 5 days
        top_n=2,            # Select top 2 assets
        rebalance_frequency='weekly'
    )
    
    print(f"Strategy parameters:")
    print(f"  - Lookback period: {strategy.lookback_period} days")
    print(f"  - Holding period: {strategy.holding_period} days")
    print(f"  - Top N assets: {strategy.top_n}")
    print(f"  - Rebalance frequency: {strategy.rebalance_frequency}")
    
    # 3. Run backtest
    print("\n3. Running backtest...")
    
    results = strategy.backtest(data)
    
    print(f"Backtest completed:")
    print(f"  - Total trades: {len(results.trades)}")
    print(f"  - Total signals: {len(results.signals)}")
    print(f"  - Portfolio value: ${results.portfolio_values.iloc[-1]:.2f}")
    
    # 4. Calculate performance metrics
    print("\n4. Calculating performance metrics...")
    
    analyzer = PerformanceAnalyzer()
    performance_results = analyzer.analyze_performance(
        returns=results.returns,
        benchmark=data.pct_change().mean(axis=1).fillna(0),  # Equal weight benchmark
        portfolio_values=results.portfolio_values
    )
    
    print("\nPerformance Summary:")
    print(f"  - Total Return: {performance_results['total_return']:.2%}")
    print(f"  - Annualized Return: {performance_results['annual_return']:.2%}")
    print(f"  - Volatility: {performance_results['volatility']:.2%}")
    print(f"  - Sharpe Ratio: {performance_results['sharpe_ratio']:.3f}")
    print(f"  - Max Drawdown: {performance_results['max_drawdown']:.2%}")
    
    # 5. Risk analysis
    print("\n5. Performing risk analysis...")
    
    risk_analyzer = RiskAnalyzer()
    risk_results = risk_analyzer.calculate_var(
        returns=results.returns,
        confidence_level=0.05
    )
    
    print(f"\nRisk Metrics:")
    print(f"  - VaR (5%): {risk_results['var']:.2%}")
    print(f"  - Expected Shortfall: {risk_results['expected_shortfall']:.2%}")
    
    # 6. Generate simple plot
    print("\n6. Generating performance plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Portfolio value over time
    ax1.plot(results.portfolio_values.index, results.portfolio_values.values, 
             label='Strategy', linewidth=2)
    
    # Benchmark (equal weight)
    benchmark_values = (1 + data.pct_change().mean(axis=1).fillna(0)).cumprod() * 100
    ax1.plot(benchmark_values.index, benchmark_values.values, 
             label='Benchmark (Equal Weight)', linewidth=2, alpha=0.7)
    
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Daily returns
    ax2.plot(results.returns.index, results.returns.values, 
             alpha=0.7, linewidth=1)
    ax2.set_title('Daily Returns')
    ax2.set_ylabel('Return (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = '/home/joelasaucedo/Development/backtesting_engine/examples/basic_strategy_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # 7. Trade analysis
    print("\n7. Analyzing trades...")
    
    if len(results.trades) > 0:
        trades_df = pd.DataFrame([
            {
                'asset': trade.asset,
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'pnl': trade.pnl,
                'return': trade.return_pct
            }
            for trade in results.trades
        ])
        
        print(f"\nTrade Statistics:")
        print(f"  - Win Rate: {(trades_df['pnl'] > 0).mean():.1%}")
        print(f"  - Average PnL: ${trades_df['pnl'].mean():.2f}")
        print(f"  - Average Return: {trades_df['return'].mean():.2%}")
        print(f"  - Best Trade: ${trades_df['pnl'].max():.2f}")
        print(f"  - Worst Trade: ${trades_df['pnl'].min():.2f}")
        
        # Show first few trades
        print(f"\nFirst 5 trades:")
        print(trades_df.head().to_string(index=False))
    
    # 8. Save results
    print("\n8. Saving results...")
    
    results_dir = '/home/joelasaucedo/Development/backtesting_engine/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save portfolio values
    results.portfolio_values.to_csv(f"{results_dir}/basic_strategy_portfolio_values.csv")
    
    # Save returns
    results.returns.to_csv(f"{results_dir}/basic_strategy_returns.csv")
    
    # Save performance metrics
    performance_df = pd.DataFrame([performance_results])
    performance_df.to_csv(f"{results_dir}/basic_strategy_performance.csv", index=False)
    
    print(f"Results saved to: {results_dir}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
