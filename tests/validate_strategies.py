#!/usr/bin/env python3
"""
Validation Test - Quick sanity check for strategy results

This script validates that strategy results are realistic and not overfitted.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Go up one level from tests/ to project root
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils.data_loader import DataLoader
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.composite_pair_trading_strategy import CompositePairTradingStrategy


def validate_results(returns, strategy_name):
    """Validate that results are realistic."""
    
    if len(returns) == 0:
        return {"valid": False, "reason": "No returns generated"}
    
    # Calculate metrics
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + returns.mean()) ** 252 - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = ann_return / volatility if volatility > 0 else 0
    
    # Validation checks
    issues = []
    
    # Check for unrealistic returns
    if total_return > 100:  # 10,000% return
        issues.append(f"Extremely high return: {total_return:.1%}")
    
    # Check for unrealistic Sharpe ratios
    if sharpe > 10:
        issues.append(f"Unrealistic Sharpe ratio: {sharpe:.1f}")
    
    # Check for too many winning trades (possible look-ahead bias)
    win_rate = (returns > 0).mean()
    if win_rate > 0.8:
        issues.append(f"Suspiciously high win rate: {win_rate:.1%}")
    
    # Check for constant returns (possible error)
    if returns.std() == 0:
        issues.append("All returns are identical")
    
    # Check for extreme daily returns
    max_daily_return = returns.max()
    if max_daily_return > 0.5:  # 50% daily return
        issues.append(f"Extreme daily return: {max_daily_return:.1%}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "metrics": {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "max_daily": max_daily_return,
            "num_trades": len(returns)
        }
    }


def main():
    print("🔍 Strategy Validation Test")
    print("=" * 50)
    
    # Load data
    data_loader = DataLoader()
    
    btc_data = data_loader.load_partitioned_crypto_data(
        symbol='BTC_USD', interval='1d',
        start_date='2024-01-01', end_date='2024-07-01'  # Limited period for validation
    )
    
    eth_data = data_loader.load_partitioned_crypto_data(
        symbol='ETH_USD', interval='1d',
        start_date='2024-01-01', end_date='2024-07-01'
    )
    
    if btc_data is None or eth_data is None:
        print("❌ Failed to load data")
        return
    
    print(f"📊 Data loaded: {len(btc_data)} periods")
    
    # Test strategies
    strategies = {
        'momentum': (MomentumStrategy(), btc_data),
        'mean_reversion': (MeanReversionStrategy(), btc_data), 
        'composite': (CompositePairTradingStrategy(
            name="CompositePairTrading",
            parameters={
                'z_entry_threshold': 0.8,  # Much lower threshold for more trades
                'z_exit_threshold': 0.1,   # Lower exit threshold  
                'lookback_window': 30,     # Shorter lookback for more responsive signals
            }
        ), (btc_data, eth_data))
    }
    
    all_valid = True
    
    for name, (strategy, data) in strategies.items():
        print(f"\n🧪 Testing {name.upper()}:")
        
        try:
            if name == 'composite':
                # Create data dictionary as expected by composite strategy
                composite_data = {
                    'BTC_USD': data[0],
                    'ETH_USD': data[1]
                }
                results = strategy.backtest(composite_data)
            else:
                results = strategy.backtest(data)
            
            if results and 'returns' in results:
                returns = results['returns'].dropna()
                validation = validate_results(returns, name)
                
                if validation['valid']:
                    print(f"  ✅ VALID - Strategy appears realistic")
                    metrics = validation['metrics']
                    print(f"     • Return: {metrics['total_return']:.2%}")
                    print(f"     • Sharpe: {metrics['sharpe_ratio']:.3f}")
                    print(f"     • Win Rate: {metrics['win_rate']:.1%}")
                    print(f"     • Trades: {metrics['num_trades']}")
                else:
                    print(f"  ⚠️  SUSPICIOUS - Potential issues detected:")
                    for issue in validation['issues']:
                        print(f"     • {issue}")
                    all_valid = False
            else:
                print(f"  ❌ FAILED - No results generated")
                all_valid = False
                
        except Exception as e:
            print(f"  ❌ ERROR - {str(e)}")
            all_valid = False
    
    print("\n" + "=" * 50)
    if all_valid:
        print("🎉 All strategies passed validation!")
    else:
        print("⚠️  Some strategies show suspicious results")
        print("💡 Consider:")
        print("   • Checking for look-ahead bias")
        print("   • Validating signal generation logic")
        print("   • Testing on out-of-sample data")
        print("   • Reviewing parameter settings")


if __name__ == "__main__":
    main()
