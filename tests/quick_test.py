#!/usr/bin/env python3
"""
Quick Strategy Tester - Rapid Prototyping and Backtesting CLI

A streamlined interface for testing strategies quickly without configuration overhead.
Usage:
    python quick_test.py --strategy momentum --symbols BTC_USD,ETH_USD --period 2024
    python quick_test.py --strategy composite --params lookback_window=30 --quick
    python quick_test.py --list-strategies
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent.parent  # Go up one level from tests/ to project root
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.utils.data_loader import DataLoader
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import BollingerBandsStrategy
from src.strategies.simple_pairs_strategy_fixed import SimplePairsStrategy, TrendFollowingStrategy, VolatilityBreakoutStrategy
from src.bootstrapping.core import AdvancedBootstrapping, BootstrapMethod


class QuickTester:
    """Fast strategy testing interface."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.strategies = {
            'momentum': MomentumStrategy,
            'mean_reversion': BollingerBandsStrategy,
            'pairs': SimplePairsStrategy,
            'trend': TrendFollowingStrategy,
            'breakout': VolatilityBreakoutStrategy
        }
        
    def list_strategies(self):
        """List available strategies."""
        print("📈 Available Strategies:")
        for name, strategy_class in self.strategies.items():
            print(f"  • {name}: {strategy_class.__doc__.split('.')[0] if strategy_class.__doc__ else 'No description'}")
    
    def quick_test(self, strategy_name: str, symbols: list, period: str, 
                   params: dict = None, quick_mode: bool = True):
        """Run quick strategy test."""
        
        print(f"🚀 Quick Testing: {strategy_name.upper()}")
        print(f"📊 Symbols: {', '.join(symbols)}")
        print(f"📅 Period: {period}")
        
        # Load data
        data = self._load_test_data(symbols, period)
        if data is None:
            return None
            
        # Initialize strategy
        strategy_class = self.strategies[strategy_name]
        if params:
            strategy = strategy_class(**params)
        else:
            strategy = strategy_class()
            
        # Run backtest
        print("⚡ Running backtest...")
        
        # All strategies now use single asset approach
        results = strategy.backtest(data[symbols[0]], symbols[0])
        
        # Quick analysis
        self._quick_analysis(results, quick_mode)
        
        return results
    
    def _load_test_data(self, symbols: list, period: str):
        """Load data for testing."""
        try:
            data = {}
            
            # Parse period
            if period == '2024':
                start_date, end_date = '2024-01-01', '2024-12-31'
            elif period == '2023':
                start_date, end_date = '2023-01-01', '2023-12-31'
            elif period == 'ytd':
                start_date = '2024-01-01'
                end_date = datetime.now().strftime('%Y-%m-%d')
            elif period == '6m' or period == 'last_6m':
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            elif period == '3m':
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            elif period == '1m':
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            elif period == '1y':
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            elif ',' in period:
                # Custom format YYYY-MM-DD,YYYY-MM-DD
                start_date, end_date = period.split(',')
            else:
                # Default to last 6 months if unrecognized
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
            
            for symbol in symbols:
                print(f"📥 Loading {symbol}...")
                symbol_data = self.data_loader.load_partitioned_crypto_data(
                    symbol=symbol, interval='1d',
                    start_date=start_date, end_date=end_date
                )
                if symbol_data is not None and len(symbol_data) > 0:
                    data[symbol] = symbol_data
                else:
                    print(f"❌ No data available for {symbol}")
                    return None
                    
            return data
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return None
    
    def _quick_analysis(self, results, quick_mode: bool):
        """Quick performance analysis."""
        
        if results is None or 'returns' not in results:
            print("❌ No results to analyze")
            return
            
        returns = results['returns'].dropna()
        if len(returns) == 0:
            print("❌ No return data")
            return
            
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = ann_return / volatility if volatility > 0 else 0
        
        max_dd = self._calculate_max_drawdown(returns)
        win_rate = (returns > 0).mean()
        
        print("\n📊 QUICK RESULTS:")
        print(f"  • Total Return: {total_return:.2%}")
        print(f"  • Annual Return: {ann_return:.2%}")
        print(f"  • Sharpe Ratio: {sharpe:.3f}")
        print(f"  • Max Drawdown: {max_dd:.2%}")
        print(f"  • Win Rate: {win_rate:.2%}")
        print(f"  • Total Trades: {len(returns)}")
        
        if not quick_mode:
            print("\n🔍 DETAILED ANALYSIS:")
            # Add bootstrap analysis
            try:
                from src.bootstrapping.core import BootstrapConfig
                
                # Create proper bootstrap configuration
                config = BootstrapConfig(n_sims=100, confidence_levels=[0.95])
                bootstrap = AdvancedBootstrapping(
                    ret_series=returns, 
                    config=config, 
                    method=BootstrapMethod.STATIONARY
                )
                
                # Run bootstrap simulation
                boot_results = bootstrap.run_bootstrap_simulation()
                
                # Extract and calculate confidence intervals for Sharpe ratio
                if 'simulated_stats' in boot_results and len(boot_results['simulated_stats']) > 0:
                    simulated_sharpes = [s.get('Sharpe', np.nan) for s in boot_results['simulated_stats']]
                    simulated_sharpes = [s for s in simulated_sharpes if not np.isnan(s)]
                    
                    if len(simulated_sharpes) > 0:
                        sharpe_ci_lower = np.percentile(simulated_sharpes, 2.5)
                        sharpe_ci_upper = np.percentile(simulated_sharpes, 97.5)
                        print(f"  • Bootstrap Sharpe CI: [{sharpe_ci_lower:.3f}, {sharpe_ci_upper:.3f}]")
                        
                        # Simple statistical significance check (if CI doesn't include 0)
                        is_significant = sharpe_ci_lower > 0 or sharpe_ci_upper < 0
                        print(f"  • Statistical Significance: {'Yes' if is_significant else 'No'}")
                    else:
                        print(f"  • Bootstrap analysis: No valid Sharpe ratios calculated")
                else:
                    print(f"  • Bootstrap analysis: No simulation results")
                    
            except Exception as e:
                print(f"  • Bootstrap analysis failed: {e}")
    
    def _calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative / peak) - 1
        return drawdown.min()


def main():
    parser = argparse.ArgumentParser(description='Quick Strategy Tester')
    parser.add_argument('--strategy', '-s', choices=['momentum', 'mean_reversion', 'pairs', 'trend', 'breakout'], 
                       help='Strategy to test')
    parser.add_argument('--symbols', default='BTC_USD,ETH_USD', 
                       help='Comma-separated symbols (default: BTC_USD,ETH_USD)')
    parser.add_argument('--period', '-p', default='2024', 
                       help='Test period: 2024, 2023, ytd, last_6m, or YYYY-MM-DD,YYYY-MM-DD')
    parser.add_argument('--params', nargs='*', default=[], 
                       help='Strategy parameters as key=value pairs')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode (skip detailed analysis)')
    parser.add_argument('--list-strategies', action='store_true', 
                       help='List available strategies')
    
    args = parser.parse_args()
    
    tester = QuickTester()
    
    if args.list_strategies:
        tester.list_strategies()
        return
    
    if not args.strategy:
        print("❌ Please specify a strategy with --strategy")
        tester.list_strategies()
        return
    
    # Parse parameters
    params = {}
    for param in args.params:
        if '=' in param:
            key, value = param.split('=', 1)
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except ValueError:
                params[key] = value
    
    symbols = args.symbols.split(',')
    results = tester.quick_test(
        strategy_name=args.strategy,
        symbols=symbols,
        period=args.period,
        params=params if params else None,
        quick_mode=args.quick
    )
    
    if results:
        print("\n✅ Test completed successfully!")
        print("💡 Use --quick=False for detailed statistical analysis")
    else:
        print("\n❌ Test failed!")


if __name__ == "__main__":
    main()
