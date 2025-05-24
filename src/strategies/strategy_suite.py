"""
Strategy Test Suite

Framework for registering, testing, and comparing multiple trading strategies
with comprehensive analysis and reporting capabilities.
"""

from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import warnings
from pathlib import Path

from ..bootstrapping import AdvancedBootstrapping, BootstrapMethod, BootstrapConfig


class StrategyTestSuite:
    """
    Comprehensive framework for testing multiple trading strategies.
    
    Provides standardized interfaces for:
    - Strategy registration and management
    - Batch testing across multiple strategies
    - Performance comparison and ranking
    - Statistical significance testing
    - Report generation
    """
    
    def __init__(self, 
                 benchmark_series: Optional[pd.Series] = None,
                 timeframe: str = '1d',
                 bootstrap_config: Optional[BootstrapConfig] = None):
        """
        Initialize strategy test suite.
        
        Args:
            benchmark_series: Benchmark return series for comparison
            timeframe: Trading timeframe
            bootstrap_config: Bootstrap configuration
        """
        self.strategies: Dict[str, Callable] = {}
        self.strategy_params: Dict[str, Dict] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.benchmark_series = benchmark_series
        self.timeframe = timeframe
        self.bootstrap_config = bootstrap_config or BootstrapConfig(n_sims=1000)
        
    def register_strategy(self, 
                         name: str, 
                         strategy_func: Callable,
                         params: Optional[Dict] = None,
                         description: Optional[str] = None) -> None:
        """
        Register a new strategy for testing.
        
        Args:
            name: Strategy name (must be unique)
            strategy_func: Function that returns strategy returns
            params: Strategy parameters
            description: Strategy description
        """
        if name in self.strategies:
            warnings.warn(f"Strategy '{name}' already exists. Overwriting.")
            
        self.strategies[name] = strategy_func
        self.strategy_params[name] = {
            'params': params or {},
            'description': description or f"Strategy: {name}"
        }
    
    def run_single_strategy(self, 
                           name: str, 
                           data: pd.DataFrame,
                           method: BootstrapMethod = BootstrapMethod.STATIONARY) -> Dict[str, Any]:
        """
        Run comprehensive analysis for a single strategy.
        
        Args:
            name: Strategy name
            data: Market data
            method: Bootstrap method
            
        Returns:
            Strategy analysis results
        """
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not registered")
        
        # Execute strategy
        strategy_func = self.strategies[name]
        params = self.strategy_params[name]['params']
        
        try:
            strategy_returns = strategy_func(data, **params)
            
            if not isinstance(strategy_returns, pd.Series):
                raise ValueError(f"Strategy '{name}' must return a pandas Series")
                
        except Exception as e:
            raise RuntimeError(f"Error executing strategy '{name}': {e}")
        
        # Run bootstrap analysis
        bootstrap = AdvancedBootstrapping(
            ret_series=strategy_returns,
            benchmark_series=self.benchmark_series,
            timeframe=self.timeframe,
            method=method,
            config=self.bootstrap_config
        )
        
        # Run full analysis
        results = bootstrap.run_full_analysis()
        
        # Add strategy metadata
        results['strategy_name'] = name
        results['strategy_description'] = self.strategy_params[name]['description']
        results['strategy_params'] = params
        
        return results
    
    def run_comprehensive_analysis(self, 
                                 data: pd.DataFrame,
                                 strategies: Optional[List[str]] = None,
                                 methods: Optional[List[BootstrapMethod]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Run comprehensive analysis on multiple strategies.
        
        Args:
            data: Market data
            strategies: List of strategies to test (if None, tests all)
            methods: Bootstrap methods to use
            
        Returns:
            Complete analysis results for all strategies
        """
        if strategies is None:
            strategies = list(self.strategies.keys())
            
        if methods is None:
            methods = [BootstrapMethod.STATIONARY]
        
        all_results = {}
        
        for strategy_name in strategies:
            if strategy_name not in self.strategies:
                warnings.warn(f"Strategy '{strategy_name}' not found. Skipping.")
                continue
                
            strategy_results = {}
            
            for method in methods:
                try:
                    method_results = self.run_single_strategy(strategy_name, data, method)
                    strategy_results[method.value] = method_results
                    
                    print(f"✓ Completed {strategy_name} with {method.value} bootstrap")
                    
                except Exception as e:
                    warnings.warn(f"Failed to analyze {strategy_name} with {method.value}: {e}")
                    continue
            
            if strategy_results:
                all_results[strategy_name] = strategy_results
        
        self.results = all_results
        return all_results
    
    def compare_strategies(self, 
                          results: Optional[Dict[str, Dict[str, Any]]] = None,
                          metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare strategies across key metrics.
        
        Args:
            results: Analysis results (if None, uses stored results)
            metrics: Metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        if results is None:
            results = self.results
            
        if not results:
            raise ValueError("No results available for comparison")
        
        if metrics is None:
            metrics = ['Sharpe', 'Sortino', 'Calmar', 'MaxDrawdown', 'CumulativeReturn']
        
        comparison_data = []
        
        for strategy_name, strategy_results in results.items():
            for method, method_results in strategy_results.items():
                original_stats = method_results['original_stats']
                
                row = {
                    'Strategy': strategy_name,
                    'Method': method,
                }
                
                for metric in metrics:
                    if metric in original_stats:
                        row[metric] = original_stats[metric]
                    else:
                        row[metric] = np.nan
                
                # Add statistical significance if available
                if 'statistical_tests' in method_results:
                    emp_p_vals = method_results['statistical_tests'].get('empirical_p_values', {})
                    for metric in metrics:
                        if metric in emp_p_vals:
                            row[f'{metric}_pvalue'] = emp_p_vals[metric]['p_two_sided']
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def rank_strategies(self, 
                       results: Optional[Dict[str, Dict[str, Any]]] = None,
                       ranking_metric: str = 'Sharpe',
                       method: str = 'stationary') -> pd.DataFrame:
        """
        Rank strategies by specified metric.
        
        Args:
            results: Analysis results
            ranking_metric: Metric to rank by
            method: Bootstrap method to use for ranking
            
        Returns:
            Ranked strategies DataFrame
        """
        comparison_df = self.compare_strategies(results)
        
        # Filter by method
        method_df = comparison_df[comparison_df['Method'] == method].copy()
        
        if method_df.empty:
            raise ValueError(f"No results found for method '{method}'")
        
        if ranking_metric not in method_df.columns:
            raise ValueError(f"Metric '{ranking_metric}' not found in results")
        
        # Sort by metric (descending for most metrics, ascending for drawdown)
        ascending = ranking_metric in ['MaxDrawdown', 'UlcerIndex', 'PainIndex']
        ranked_df = method_df.sort_values(ranking_metric, ascending=ascending)
        
        # Add rank column
        ranked_df['Rank'] = range(1, len(ranked_df) + 1)
        
        # Reorder columns
        cols = ['Rank', 'Strategy', ranking_metric] + [c for c in ranked_df.columns 
                                                      if c not in ['Rank', 'Strategy', ranking_metric, 'Method']]
        
        return ranked_df[cols]
    
    def statistical_significance_matrix(self, 
                                      results: Optional[Dict[str, Dict[str, Any]]] = None,
                                      metric: str = 'Sharpe',
                                      alpha: float = 0.05) -> pd.DataFrame:
        """
        Create pairwise statistical significance matrix.
        
        Args:
            results: Analysis results
            metric: Metric to test
            alpha: Significance level
            
        Returns:
            Significance matrix DataFrame
        """
        if results is None:
            results = self.results
        
        strategies = list(results.keys())
        n_strategies = len(strategies)
        
        # Initialize matrix
        sig_matrix = np.full((n_strategies, n_strategies), np.nan)
        
        # This is a simplified implementation
        # In practice, we'd need to implement proper pairwise bootstrap tests
        
        for i, strat1 in enumerate(strategies):
            for j, strat2 in enumerate(strategies):
                if i == j:
                    sig_matrix[i, j] = 0.0  # Same strategy
                else:
                    # Placeholder for actual pairwise test
                    # Would implement bootstrap difference test here
                    sig_matrix[i, j] = 0.5  # Placeholder
        
        return pd.DataFrame(sig_matrix, index=strategies, columns=strategies)
    
    def optimization_analysis(self, 
                            strategy_name: str,
                            data: pd.DataFrame,
                            param_grid: Dict[str, List],
                            optimization_metric: str = 'Sharpe') -> Dict[str, Any]:
        """
        Perform parameter optimization analysis.
        
        Args:
            strategy_name: Strategy to optimize
            data: Market data
            param_grid: Parameter grid for optimization
            optimization_metric: Metric to optimize
            
        Returns:
            Optimization results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not registered")
        
        from itertools import product
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        optimization_results = []
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            try:
                # Temporarily update strategy parameters
                original_params = self.strategy_params[strategy_name]['params'].copy()
                self.strategy_params[strategy_name]['params'].update(params)
                
                # Run analysis
                results = self.run_single_strategy(strategy_name, data)
                
                # Extract metric
                metric_value = results['original_stats'].get(optimization_metric, np.nan)
                
                optimization_results.append({
                    'params': params.copy(),
                    'metric_value': metric_value,
                    **results['original_stats']
                })
                
                # Restore original parameters
                self.strategy_params[strategy_name]['params'] = original_params
                
                print(f"✓ Completed optimization {i+1}/{len(param_combinations)}")
                
            except Exception as e:
                warnings.warn(f"Optimization failed for params {params}: {e}")
                continue
        
        if not optimization_results:
            raise RuntimeError("All optimization runs failed")
        
        # Find best parameters
        opt_df = pd.DataFrame(optimization_results)
        
        # Sort by optimization metric
        ascending = optimization_metric in ['MaxDrawdown', 'UlcerIndex']
        best_idx = opt_df[optimization_metric].idxmax() if not ascending else opt_df[optimization_metric].idxmin()
        best_params = opt_df.loc[best_idx, 'params']
        best_value = opt_df.loc[best_idx, optimization_metric]
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'optimization_metric': optimization_metric,
            'all_results': optimization_results,
            'results_df': opt_df
        }
    
    def export_results(self, 
                      output_dir: str = 'results/strategy_comparison/',
                      results: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """
        Export all strategy results to files.
        
        Args:
            output_dir: Output directory
            results: Results to export
        """
        if results is None:
            results = self.results
            
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export comparison table
        comparison_df = self.compare_strategies(results)
        comparison_df.to_csv(f'{output_dir}/strategy_comparison.csv', index=False)
        
        # Export individual strategy results
        for strategy_name, strategy_results in results.items():
            strategy_dir = Path(output_dir) / strategy_name
            strategy_dir.mkdir(exist_ok=True)
            
            for method, method_results in strategy_results.items():
                # Export statistics
                stats_df = pd.DataFrame([method_results['original_stats']])
                stats_df.to_csv(strategy_dir / f'{method}_original_stats.csv', index=False)
                
                # Export simulation results
                if 'simulated_stats' in method_results:
                    sim_df = pd.DataFrame(method_results['simulated_stats'])
                    sim_df.to_csv(strategy_dir / f'{method}_simulated_stats.csv', index=False)
                
                # Export statistical tests
                if 'statistical_tests' in method_results:
                    tests_df = pd.DataFrame([method_results['statistical_tests']])
                    tests_df.to_csv(strategy_dir / f'{method}_statistical_tests.csv', index=False)
    
    def generate_comprehensive_report(self, 
                                    output_dir: str = 'results/strategy_reports/',
                                    results: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """
        Generate comprehensive HTML report for all strategies.
        
        Args:
            output_dir: Output directory
            results: Results to include in report
            
        Returns:
            Path to generated report
        """
        if results is None:
            results = self.results
            
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # This would generate a comprehensive multi-strategy report
        # For now, return a placeholder
        
        report_path = Path(output_dir) / 'strategy_comparison_report.html'
        
        # Generate basic comparison report
        comparison_df = self.compare_strategies(results)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Strategy Comparison Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Strategy Comparison Table</h2>
            {comparison_df.to_html(classes='comparison-table', table_id='comparison')}
            
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def get_strategy_list(self) -> List[Dict[str, Any]]:
        """
        Get list of registered strategies with metadata.
        
        Returns:
            List of strategy information
        """
        strategy_list = []
        
        for name, func in self.strategies.items():
            strategy_info = {
                'name': name,
                'function': func.__name__,
                'description': self.strategy_params[name]['description'],
                'params': self.strategy_params[name]['params'],
                'has_results': name in self.results
            }
            strategy_list.append(strategy_info)
        
        return strategy_list
    
    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results.clear()
    
    def remove_strategy(self, name: str) -> None:
        """Remove a registered strategy."""
        if name in self.strategies:
            del self.strategies[name]
            del self.strategy_params[name]
            
        if name in self.results:
            del self.results[name]
