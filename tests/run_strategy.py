#!/usr/bin/env python3
"""
Strategy Runner - Simple interface for testing and developing strategies

Quick commands for common tasks:
  python run_strategy.py momentum           # Test momentum strategy
  python run_strategy.py composite --plot   # Test composite with plots  
  python run_strategy.py --optimize composite --params lookback_window=20,30,40
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent.parent  # Go up one level from tests/ to project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.utils.data_loader import DataLoader
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.composite_pair_trading_strategy import CompositePairTradingStrategy


def load_default_data():
    """Load default BTC/ETH data for 2024."""
    data_loader = DataLoader()
    
    btc_data = data_loader.load_partitioned_crypto_data(
        symbol='BTC_USD', interval='1d',
        start_date='2024-01-01', end_date='2024-12-31'
    )
    
    eth_data = data_loader.load_partitioned_crypto_data(
        symbol='ETH_USD', interval='1d', 
        start_date='2024-01-01', end_date='2024-12-31'
    )
    
    return btc_data, eth_data


def run_strategy(strategy_name: str, plot: bool = False, **kwargs):
    """Run a single strategy."""
    
    print(f"🚀 Running {strategy_name.upper()} Strategy")
    
    # Load data
    btc_data, eth_data = load_default_data()
    if btc_data is None or eth_data is None:
        print("❌ Failed to load data")
        return None
    
    # Initialize strategy
    strategies = {
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'composite': CompositePairTradingStrategy
    }
    
    if strategy_name not in strategies:
        print(f"❌ Unknown strategy: {strategy_name}")
        return None
    
    strategy_class = strategies[strategy_name]
    strategy = strategy_class(**kwargs)
    
    # Run backtest
    if strategy_name == 'composite':
        pair_data = {'BTC_USD': btc_data, 'ETH_USD': eth_data}
        results = strategy.backtest(pair_data)
    else:
        results = strategy.backtest(btc_data)
    
    # Display results
    if results and 'returns' in results:
        returns = results['returns'].dropna()
        
        total_return = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        print(f"📊 Results:")
        print(f"  • Total Return: {total_return:.2%}")
        print(f"  • Sharpe Ratio: {sharpe:.3f}")
        print(f"  • Number of Trades: {len(returns)}")
        
        if plot and len(returns) > 0:
            # Simple plot
            cumulative = (1 + returns).cumprod()
            
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(cumulative.index, cumulative.values)
            plt.title(f'{strategy_name.upper()} Strategy - Cumulative Returns')
            plt.ylabel('Cumulative Return')
            
            plt.subplot(2, 1, 2)
            plt.plot(returns.index, returns.values)
            plt.title('Daily Returns')
            plt.ylabel('Daily Return')
            plt.xlabel('Date')
            
            plt.tight_layout()
            plt.show()
        
        return results
    else:
        print("❌ No results generated")
        return None


def optimize_strategy(strategy_name: str, param_ranges: dict):
    """Simple parameter optimization."""
    
    print(f"🎯 Optimizing {strategy_name.upper()}")
    
    # Load data
    btc_data, eth_data = load_default_data()
    if btc_data is None or eth_data is None:
        return None
    
    strategies = {
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'composite': CompositePairTradingStrategy
    }
    
    strategy_class = strategies[strategy_name]
    best_sharpe = -np.inf
    best_params = None
    
    # Simple grid search
    param_names = list(param_ranges.keys())
    
    for param_name in param_names:
        values = param_ranges[param_name]
        print(f"📈 Testing {param_name}: {values}")
        
        for value in values:
            params = {param_name: value}
            
            try:
                strategy = strategy_class(**params)
                
                if strategy_name == 'composite':
                    pair_data = {'BTC_USD': btc_data, 'ETH_USD': eth_data}
                    results = strategy.backtest(pair_data)
                else:
                    results = strategy.backtest(btc_data)
                
                if results and 'returns' in results:
                    returns = results['returns'].dropna()
                    if len(returns) > 0:
                        sharpe = returns.mean() / returns.std() * np.sqrt(252)
                        print(f"  {param_name}={value}: Sharpe={sharpe:.3f}")
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = params
            except Exception as e:
                print(f"  {param_name}={value}: Error - {e}")
    
    print(f"\n🏆 Best Parameters: {best_params}")
    print(f"🏆 Best Sharpe: {best_sharpe:.3f}")
    
    return best_params


def main():
    parser = argparse.ArgumentParser(description='Strategy Runner')
    parser.add_argument('strategy', nargs='?', choices=['momentum', 'mean_reversion', 'composite'],
                       help='Strategy to run')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    parser.add_argument('--optimize', help='Strategy to optimize')
    parser.add_argument('--params', help='Parameter ranges for optimization (e.g., lookback_window=20,30,40)')
    
    args = parser.parse_args()
    
    if args.optimize and args.params:
        # Optimization mode
        param_name, values_str = args.params.split('=')
        values = [int(v) if v.isdigit() else float(v) for v in values_str.split(',')]
        param_ranges = {param_name: values}
        
        optimize_strategy(args.optimize, param_ranges)
        
    elif args.strategy:
        # Single strategy mode
        run_strategy(args.strategy, plot=args.plot)
        
    else:
        # Interactive mode
        print("🎯 Strategy Runner - Interactive Mode")
        print("\nAvailable strategies:")
        print("  1. momentum")
        print("  2. mean_reversion") 
        print("  3. composite")
        
        choice = input("\nSelect strategy (1-3): ").strip()
        strategy_map = {'1': 'momentum', '2': 'mean_reversion', '3': 'composite'}
        
        if choice in strategy_map:
            strategy_name = strategy_map[choice]
            plot_choice = input("Show plots? (y/n): ").strip().lower() == 'y'
            run_strategy(strategy_name, plot=plot_choice)
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
