"""
Parameter optimization utilities for strategy optimization and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Callable, Any, Optional
from dataclasses import dataclass
from itertools import product
import warnings

try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    n_evaluations: int
    optimization_time: float
    success: bool
    message: str


class ParameterOptimizer:
    """
    Base class for parameter optimization.
    """
    
    def __init__(self, 
                 objective_function: Callable,
                 maximize: bool = True,
                 n_jobs: int = 1,
                 random_state: int = 42):
        """
        Initialize parameter optimizer.
        
        Parameters:
        -----------
        objective_function : callable
            Function to optimize (takes parameter dict, returns score)
        maximize : bool
            Whether to maximize (True) or minimize (False) the objective
        n_jobs : int
            Number of parallel jobs
        random_state : int
            Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.maximize = maximize
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.results_history = []
    
    def _evaluate_objective(self, params: Dict[str, Any]) -> float:
        """
        Evaluate objective function with error handling.
        
        Parameters:
        -----------
        params : dict
            Parameter values to evaluate
            
        Returns:
        --------
        float
            Objective function value
        """
        try:
            score = self.objective_function(params)
            
            # Handle different return types
            if isinstance(score, (tuple, list)):
                score = score[0]  # Take first element if tuple/list
            
            # Convert to float and handle NaN/inf
            score = float(score)
            if not np.isfinite(score):
                score = -np.inf if self.maximize else np.inf
                
            # Store result
            result = params.copy()
            result['score'] = score
            self.results_history.append(result)
            
            return score
            
        except Exception as e:
            warnings.warn(f"Objective function evaluation failed: {e}")
            return -np.inf if self.maximize else np.inf
    
    def get_best_result(self) -> Optional[Dict[str, Any]]:
        """Get the best result from optimization history."""
        if not self.results_history:
            return None
        
        if self.maximize:
            best_result = max(self.results_history, key=lambda x: x['score'])
        else:
            best_result = min(self.results_history, key=lambda x: x['score'])
        
        return best_result


class GridSearchOptimizer(ParameterOptimizer):
    """
    Grid search parameter optimization.
    """
    
    def optimize(self, 
                param_grid: Dict[str, List[Any]],
                cv_folds: int = 5) -> OptimizationResult:
        """
        Perform grid search optimization.
        
        Parameters:
        -----------
        param_grid : dict
            Parameter grid to search
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        import time
        start_time = time.time()
        
        try:
            # Generate all parameter combinations
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(product(*param_values))
            
            best_score = -np.inf if self.maximize else np.inf
            best_params = {}
            
            for combination in param_combinations:
                params = dict(zip(param_names, combination))
                score = self._evaluate_objective(params)
                
                if self.maximize and score > best_score:
                    best_score = score
                    best_params = params
                elif not self.maximize and score < best_score:
                    best_score = score
                    best_params = params
            
            optimization_time = time.time() - start_time
            
            return OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                all_results=self.results_history.copy(),
                n_evaluations=len(param_combinations),
                optimization_time=optimization_time,
                success=True,
                message="Grid search completed successfully"
            )
            
        except Exception as e:
            optimization_time = time.time() - start_time
            return OptimizationResult(
                best_params={},
                best_score=-np.inf if self.maximize else np.inf,
                all_results=self.results_history.copy(),
                n_evaluations=len(self.results_history),
                optimization_time=optimization_time,
                success=False,
                message=f"Grid search failed: {str(e)}"
            )


class RandomSearchOptimizer(ParameterOptimizer):
    """
    Random search parameter optimization.
    """
    
    def optimize(self, 
                param_distributions: Dict[str, Any],
                n_iter: int = 100) -> OptimizationResult:
        """
        Perform random search optimization.
        
        Parameters:
        -----------
        param_distributions : dict
            Parameter distributions to sample from
        n_iter : int
            Number of random samples
            
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        import time
        start_time = time.time()
        
        try:
            np.random.seed(self.random_state)
            
            best_score = -np.inf if self.maximize else np.inf
            best_params = {}
            
            for i in range(n_iter):
                params = {}
                for param_name, distribution in param_distributions.items():
                    if hasattr(distribution, 'rvs'):  # scipy distribution
                        params[param_name] = distribution.rvs()
                    elif isinstance(distribution, (list, tuple)):  # choice from list
                        params[param_name] = np.random.choice(distribution)
                    elif isinstance(distribution, dict):  # uniform/normal distribution
                        if distribution['type'] == 'uniform':
                            params[param_name] = np.random.uniform(
                                distribution['low'], distribution['high']
                            )
                        elif distribution['type'] == 'normal':
                            params[param_name] = np.random.normal(
                                distribution['loc'], distribution['scale']
                            )
                        elif distribution['type'] == 'log_uniform':
                            params[param_name] = np.exp(np.random.uniform(
                                np.log(distribution['low']), 
                                np.log(distribution['high'])
                            ))
                    else:
                        params[param_name] = distribution
                
                score = self._evaluate_objective(params)
                
                if self.maximize and score > best_score:
                    best_score = score
                    best_params = params
                elif not self.maximize and score < best_score:
                    best_score = score
                    best_params = params
            
            optimization_time = time.time() - start_time
            
            return OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                all_results=self.results_history.copy(),
                n_evaluations=n_iter,
                optimization_time=optimization_time,
                success=True,
                message="Random search completed successfully"
            )
            
        except Exception as e:
            optimization_time = time.time() - start_time
            return OptimizationResult(
                best_params={},
                best_score=-np.inf if self.maximize else np.inf,
                all_results=self.results_history.copy(),
                n_evaluations=len(self.results_history),
                optimization_time=optimization_time,
                success=False,
                message=f"Random search failed: {str(e)}"
            )


class BayesianOptimizer(ParameterOptimizer):
    """
    Bayesian optimization using Gaussian processes.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for BayesianOptimizer")
    
    def optimize(self, 
                param_space: List[Any],
                param_names: List[str],
                n_calls: int = 100,
                n_initial_points: int = 10,
                acquisition_function: str = 'gp_hedge') -> OptimizationResult:
        """
        Perform Bayesian optimization.
        
        Parameters:
        -----------
        param_space : list
            Parameter space definitions (Real, Integer, Categorical)
        param_names : list
            Parameter names corresponding to param_space
        n_calls : int
            Number of function evaluations
        n_initial_points : int
            Number of random initial points
        acquisition_function : str
            Acquisition function to use
            
        Returns:
        --------
        OptimizationResult
            Optimization results
        """
        import time
        start_time = time.time()
        
        try:
            @use_named_args(param_space)
            def objective(**params):
                score = self._evaluate_objective(params)
                # Negate score if maximizing (skopt minimizes)
                return -score if self.maximize else score
            
            # Choose optimizer based on acquisition function
            if acquisition_function == 'gp_hedge':
                result = gp_minimize(
                    objective,
                    param_space,
                    n_calls=n_calls,
                    n_initial_points=n_initial_points,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            elif acquisition_function == 'forest':
                result = forest_minimize(
                    objective,
                    param_space,
                    n_calls=n_calls,
                    n_initial_points=n_initial_points,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            elif acquisition_function == 'gbrt':
                result = gbrt_minimize(
                    objective,
                    param_space,
                    n_calls=n_calls,
                    n_initial_points=n_initial_points,
                    random_state=self.random_state,
                    n_jobs=self.n_jobs
                )
            else:
                raise ValueError(f"Unknown acquisition function: {acquisition_function}")
            
            # Extract best parameters
            best_params = dict(zip(param_names, result.x))
            best_score = -result.fun if self.maximize else result.fun
            
            optimization_time = time.time() - start_time
            
            return OptimizationResult(
                best_params=best_params,
                best_score=best_score,
                all_results=self.results_history.copy(),
                n_evaluations=len(result.func_vals),
                optimization_time=optimization_time,
                success=True,
                message="Bayesian optimization completed successfully"
            )
            
        except Exception as e:
            optimization_time = time.time() - start_time
            return OptimizationResult(
                best_params={},
                best_score=-np.inf if self.maximize else np.inf,
                all_results=self.results_history.copy(),
                n_evaluations=len(self.results_history),
                optimization_time=optimization_time,
                success=False,
                message=f"Bayesian optimization failed: {str(e)}"
            )


def optimize_strategy_parameters(strategy_class: type,
                               data: pd.DataFrame,
                               param_space: Dict[str, Any],
                               optimization_method: str = 'grid',
                               metric: str = 'sharpe_ratio',
                               **optimizer_kwargs) -> OptimizationResult:
    """
    Optimize strategy parameters.
    
    Parameters:
    -----------
    strategy_class : type
        Strategy class to optimize
    data : pd.DataFrame
        Price/return data
    param_space : dict
        Parameter space to search
    optimization_method : str
        Optimization method ('grid', 'random', 'bayesian')
    metric : str
        Metric to optimize
    **optimizer_kwargs
        Additional arguments for optimizer
        
    Returns:
    --------
    OptimizationResult
        Optimization results
    """
    def objective_function(params):
        try:
            # Create strategy instance
            strategy = strategy_class(**params)
            
            # Run backtest
            results = strategy.backtest(data)
            
            # Extract metric
            if metric == 'sharpe_ratio':
                return results.performance_metrics.get('sharpe_ratio', -np.inf)
            elif metric == 'total_return':
                return results.performance_metrics.get('total_return', -np.inf)
            elif metric == 'max_drawdown':
                return -results.risk_metrics.get('max_drawdown', np.inf)  # Minimize drawdown
            else:
                return results.performance_metrics.get(metric, -np.inf)
                
        except Exception as e:
            warnings.warn(f"Strategy evaluation failed: {e}")
            return -np.inf
    
    # Create optimizer
    if optimization_method == 'grid':
        optimizer = GridSearchOptimizer(objective_function, maximize=True)
        return optimizer.optimize(param_space, **optimizer_kwargs)
    
    elif optimization_method == 'random':
        optimizer = RandomSearchOptimizer(objective_function, maximize=True)
        return optimizer.optimize(param_space, **optimizer_kwargs)
    
    elif optimization_method == 'bayesian':
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        
        optimizer = BayesianOptimizer(objective_function, maximize=True)
        
        # Convert param_space to skopt format
        space = []
        names = []
        for name, values in param_space.items():
            names.append(name)
            if isinstance(values, list) and all(isinstance(x, (int, float)) for x in values):
                # Numerical range
                space.append(Real(min(values), max(values), name=name))
            elif isinstance(values, list):
                # Categorical
                space.append(Categorical(values, name=name))
            elif isinstance(values, dict):
                if values['type'] == 'real':
                    space.append(Real(values['low'], values['high'], name=name))
                elif values['type'] == 'integer':
                    space.append(Integer(values['low'], values['high'], name=name))
        
        return optimizer.optimize(space, names, **optimizer_kwargs)
    
    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")


def walk_forward_optimization(strategy_class: type,
                            data: pd.DataFrame,
                            param_space: Dict[str, Any],
                            window_size: int = 252,
                            step_size: int = 63,
                            min_periods: int = 126) -> Dict[str, Any]:
    """
    Perform walk-forward optimization.
    
    Parameters:
    -----------
    strategy_class : type
        Strategy class to optimize
    data : pd.DataFrame
        Price/return data
    param_space : dict
        Parameter space to search
    window_size : int
        Optimization window size
    step_size : int
        Step size for walk-forward
    min_periods : int
        Minimum periods required
        
    Returns:
    --------
    dict
        Walk-forward optimization results
    """
    results = {
        'optimization_results': [],
        'out_of_sample_performance': [],
        'parameter_stability': {},
        'summary': {}
    }
    
    n_periods = len(data)
    start_idx = min_periods
    
    while start_idx + window_size < n_periods:
        end_idx = min(start_idx + window_size, n_periods)
        
        # In-sample data for optimization
        train_data = data.iloc[start_idx:end_idx]
        
        # Out-of-sample data for validation
        test_start = end_idx
        test_end = min(test_start + step_size, n_periods)
        test_data = data.iloc[test_start:test_end]
        
        if len(test_data) == 0:
            break
        
        # Optimize on training data
        opt_result = optimize_strategy_parameters(
            strategy_class=strategy_class,
            data=train_data,
            param_space=param_space,
            optimization_method='grid'
        )
        
        # Test on out-of-sample data
        strategy = strategy_class(**opt_result.best_params)
        oos_results = strategy.backtest(test_data)
        
        results['optimization_results'].append({
            'train_period': (start_idx, end_idx),
            'test_period': (test_start, test_end),
            'best_params': opt_result.best_params,
            'in_sample_score': opt_result.best_score
        })
        
        results['out_of_sample_performance'].append({
            'period': (test_start, test_end),
            'performance': oos_results.performance_metrics,
            'params_used': opt_result.best_params
        })
        
        start_idx += step_size
    
    # Calculate parameter stability
    all_params = [r['best_params'] for r in results['optimization_results']]
    for param_name in param_space.keys():
        param_values = [params.get(param_name) for params in all_params]
        param_values = [v for v in param_values if v is not None]
        
        if param_values:
            results['parameter_stability'][param_name] = {
                'mean': np.mean(param_values),
                'std': np.std(param_values),
                'min': np.min(param_values),
                'max': np.max(param_values)
            }
    
    # Calculate summary statistics
    oos_returns = [p['performance'].get('total_return', 0) 
                   for p in results['out_of_sample_performance']]
    oos_sharpe = [p['performance'].get('sharpe_ratio', 0) 
                  for p in results['out_of_sample_performance']]
    
    results['summary'] = {
        'total_periods': len(results['optimization_results']),
        'avg_oos_return': np.mean(oos_returns),
        'avg_oos_sharpe': np.mean(oos_sharpe),
        'oos_return_std': np.std(oos_returns),
        'positive_periods': sum(1 for r in oos_returns if r > 0)
    }
    
    return results
