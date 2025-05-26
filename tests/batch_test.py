#!/usr/bin/env python3
"""
Batch Backtesting Engine - Run Multiple Strategies and Configurations

Streamlined batch testing for strategy comparison and parameter optimization.
Usage:
    python batch_test.py --config configs/quick_batch.yaml
    python batch_test.py --strategies momentum,composite --symbols BTC_USD,ETH_USD
    python batch_test.py --optimize composite --param-range lookback_window=20,30,40
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import json
from datetime import datetime
from itertools import product
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


class BatchTester:
    """Batch testing engine for multiple strategies."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.strategies = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'composite': CompositePairTradingStrategy
        }
        self.results = {}
        
    def run_batch_test(self, config: dict):
        """Run batch test from configuration."""
        
        print("🚀 Starting Batch Backtest")
        print(f"📊 Strategies: {config.get('strategies', [])}")
        print(f"📈 Symbols: {config.get('symbols', [])}")
        print(f"📅 Period: {config.get('period', 'N/A')}")
        
        # Load data once
        data = self._load_batch_data(config['symbols'], config['period'])
        if not data:
            print("❌ Failed to load data")
            return
            
        # Run tests for each strategy
        all_results = {}
        
        for strategy_name in config['strategies']:
            print(f"\n🔄 Testing Strategy: {strategy_name.upper()}")
            
            if strategy_name in config.get('strategy_params', {}):
                param_configs = config['strategy_params'][strategy_name]
                if isinstance(param_configs, list):
                    # Multiple parameter sets
                    for i, params in enumerate(param_configs):
                        test_name = f"{strategy_name}_config_{i+1}"
                        result = self._run_single_test(strategy_name, data, params)
                        if result:
                            all_results[test_name] = result
                            all_results[test_name]['config'] = params
                else:
                    # Single parameter set
                    result = self._run_single_test(strategy_name, data, param_configs)
                    if result:
                        all_results[strategy_name] = result
                        all_results[strategy_name]['config'] = param_configs
            else:
                # Default parameters
                result = self._run_single_test(strategy_name, data)
                if result:
                    all_results[strategy_name] = result
                    all_results[strategy_name]['config'] = {}
        
        # Generate comparison report
        self._generate_batch_report(all_results, config)
        
        return all_results
    
    def run_parameter_optimization(self, strategy_name: str, symbols: list, 
                                 param_ranges: dict, period: str = '2024'):
        """Run parameter optimization for a strategy."""
        
        print(f"🎯 Optimizing {strategy_name.upper()}")
        print(f"📊 Parameter Ranges: {param_ranges}")
        
        # Load data
        data = self._load_batch_data(symbols, period)
        if not data:
            return None
            
        # Generate parameter combinations
        param_names = list(param_ranges.keys())
        param_values = [param_ranges[name] for name in param_names]
        param_combinations = list(product(*param_values))
        
        print(f"🔍 Testing {len(param_combinations)} parameter combinations...")
        
        best_result = None
        best_sharpe = -np.inf
        best_params = None
        all_results = {}
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            # Quick progress update
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Progress: {i+1}/{len(param_combinations)} ({(i+1)/len(param_combinations)*100:.1f}%)")
            
            try:
                result = self._run_single_test(strategy_name, data, params, verbose=False)
                if result and 'metrics' in result:
                    sharpe = result['metrics'].get('sharpe_ratio', -np.inf)
                    param_key = '_'.join([f"{k}={v}" for k, v in params.items()])
                    all_results[param_key] = {
                        'params': params,
                        'sharpe': sharpe,
                        'total_return': result['metrics'].get('total_return', 0),
                        'max_drawdown': result['metrics'].get('max_drawdown', 0)
                    }
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = params
                        best_result = result
                        
            except Exception as e:
                continue
        
        # Print optimization results
        print(f"\n🏆 OPTIMIZATION RESULTS:")
        print(f"  • Best Sharpe Ratio: {best_sharpe:.3f}")
        print(f"  • Best Parameters: {best_params}")
        
        # Save detailed results
        self._save_optimization_results(strategy_name, all_results, best_params)
        
        return best_result, best_params, all_results
    
    def _run_single_test(self, strategy_name: str, data: dict, params: dict = None, verbose: bool = True):
        """Run single strategy test."""
        
        try:
            strategy_class = self.strategies[strategy_name]
            
            if params:
                # Handle composite strategy's nested parameter structure
                if strategy_name == 'composite' and 'parameters' in params:
                    strategy = strategy_class(parameters=params['parameters'])
                else:
                    strategy = strategy_class(**params)
            else:
                strategy = strategy_class()
            
            # Run backtest based on strategy type
            symbols = list(data.keys())
            
            if strategy_name == 'composite' and len(symbols) >= 2:
                # Composite strategy expects a dictionary with both assets
                pair_data = {symbols[0]: data[symbols[0]], symbols[1]: data[symbols[1]]}
                results = strategy.backtest(pair_data)
            else:
                results = strategy.backtest(data[symbols[0]])
            
            if results and 'returns' in results:
                # Calculate metrics
                returns = results['returns'].dropna()
                metrics = self._calculate_metrics(returns)
                results['metrics'] = metrics
                
                if verbose:
                    print(f"  ✅ {strategy_name}: Sharpe={metrics['sharpe_ratio']:.3f}, Return={metrics['total_return']:.2%}")
                
                return results
                
        except Exception as e:
            if verbose:
                print(f"  ❌ {strategy_name}: {str(e)}")
            return None
    
    def _load_batch_data(self, symbols: list, period: str):
        """Load data for batch testing."""
        try:
            data = {}
            
            # Parse period
            if period == '2024':
                start_date, end_date = '2024-01-01', '2024-12-31'
            elif period == '2023':
                start_date, end_date = '2023-01-01', '2023-12-31'
            else:
                start_date, end_date = period.split(',')
            
            for symbol in symbols:
                symbol_data = self.data_loader.load_partitioned_crypto_data(
                    symbol=symbol, interval='1d',
                    start_date=start_date, end_date=end_date
                )
                if symbol_data is not None and len(symbol_data) > 0:
                    data[symbol] = symbol_data
                else:
                    print(f"❌ No data for {symbol}")
                    return None
                    
            return data
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return None
    
    def _calculate_metrics(self, returns):
        """Calculate performance metrics."""
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = ann_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative / peak) - 1
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (returns > 0).mean(),
            'num_trades': len(returns)
        }
    
    def _generate_batch_report(self, results: dict, config: dict):
        """Generate comparison report."""
        
        print("\n📊 BATCH TEST RESULTS SUMMARY")
        print("=" * 60)
        
        # Create summary table
        summary_data = []
        for name, result in results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                summary_data.append({
                    'Strategy': name,
                    'Total Return': f"{metrics['total_return']:.2%}",
                    'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                    'Max DD': f"{metrics['max_drawdown']:.2%}",
                    'Win Rate': f"{metrics['win_rate']:.2%}",
                    'Trades': metrics['num_trades']
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False))
            
            # Save detailed results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"results/exports/batch_test_{timestamp}.json"
            
            # Prepare results for JSON serialization
            json_results = {}
            for name, result in results.items():
                json_results[name] = {
                    'metrics': result['metrics'],
                    'config': result.get('config', {}),
                    'num_returns': len(result.get('returns', []))
                }
            
            Path(results_file).parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump({
                    'config': config,
                    'results': json_results,
                    'timestamp': timestamp
                }, f, indent=2)
            
            print(f"\n💾 Detailed results saved to: {results_file}")
        else:
            print("❌ No successful results to display")
    
    def _save_optimization_results(self, strategy_name: str, all_results: dict, best_params: dict):
        """Save optimization results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results/exports/optimization_{strategy_name}_{timestamp}.json"
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump({
                'strategy': strategy_name,
                'best_params': best_params,
                'all_results': all_results,
                'timestamp': timestamp
            }, f, indent=2)
        
        print(f"💾 Optimization results saved to: {filename}")


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Batch Strategy Tester')
    parser.add_argument('--config', '-c', help='Configuration YAML file')
    parser.add_argument('--strategies', help='Comma-separated strategy names')
    parser.add_argument('--symbols', default='BTC_USD,ETH_USD', help='Comma-separated symbols')
    parser.add_argument('--period', default='2024', help='Test period')
    parser.add_argument('--optimize', help='Strategy to optimize')
    parser.add_argument('--param-range', nargs='*', help='Parameter ranges for optimization')
    
    args = parser.parse_args()
    
    tester = BatchTester()
    
    if args.optimize and args.param_range:
        # Parameter optimization mode
        param_ranges = {}
        for param_spec in args.param_range:
            name, values_str = param_spec.split('=')
            values = [float(v) if '.' in v else int(v) for v in values_str.split(',')]
            param_ranges[name] = values
        
        symbols = args.symbols.split(',')
        tester.run_parameter_optimization(args.optimize, symbols, param_ranges, args.period)
        
    elif args.config:
        # Configuration file mode
        config = load_config(args.config)
        tester.run_batch_test(config)
        
    elif args.strategies:
        # Command line mode
        config = {
            'strategies': args.strategies.split(','),
            'symbols': args.symbols.split(','),
            'period': args.period
        }
        tester.run_batch_test(config)
        
    else:
        print("❌ Please specify --config, --strategies, or --optimize")
        print("Examples:")
        print("  python batch_test.py --strategies momentum,composite --symbols BTC_USD,ETH_USD")
        print("  python batch_test.py --optimize composite --param-range lookback_window=20,30,40")


if __name__ == "__main__":
    main()
