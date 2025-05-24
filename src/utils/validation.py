"""
Data Validation Module

Comprehensive validation utilities for:
- Input data validation
- Parameter validation
- Strategy validation
- Portfolio validation
- Result validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from datetime import datetime, timedelta


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


class DataValidator:
    """
    Comprehensive data validation for trading strategies and portfolios
    """
    
    @staticmethod
    def validate_returns(returns: Union[pd.Series, pd.DataFrame, np.ndarray], 
                        allow_missing: bool = False,
                        min_observations: int = 30) -> bool:
        """
        Validate return data
        
        Args:
            returns: Return data
            allow_missing: Whether to allow missing values
            min_observations: Minimum number of observations required
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
        elif isinstance(returns, pd.DataFrame):
            # Validate each column
            for col in returns.columns:
                DataValidator.validate_returns(returns[col], allow_missing, min_observations)
            return True
        
        # Check if empty
        if len(returns) == 0:
            raise ValidationError("Return data is empty")
        
        # Check minimum observations
        if len(returns) < min_observations:
            raise ValidationError(f"Insufficient observations: {len(returns)} < {min_observations}")
        
        # Check for missing values
        if not allow_missing and returns.isnull().any():
            raise ValidationError("Return data contains missing values")
        
        # Check for infinite values
        if np.isinf(returns).any():
            raise ValidationError("Return data contains infinite values")
        
        # Check for extreme values (returns > 100% or < -100%)
        if (returns > 1.0).any():
            warnings.warn("Return data contains values > 100%")
        
        if (returns < -1.0).any():
            warnings.warn("Return data contains values < -100%")
        
        # Check for constant returns
        if returns.std() == 0:
            warnings.warn("Return data has zero variance (constant returns)")
        
        return True
    
    @staticmethod
    def validate_prices(prices: Union[pd.Series, pd.DataFrame, np.ndarray],
                       allow_negative: bool = False) -> bool:
        """
        Validate price data
        
        Args:
            prices: Price data
            allow_negative: Whether to allow negative prices
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)
        elif isinstance(prices, pd.DataFrame):
            # Validate each column
            for col in prices.columns:
                DataValidator.validate_prices(prices[col], allow_negative)
            return True
        
        # Check if empty
        if len(prices) == 0:
            raise ValidationError("Price data is empty")
        
        # Check for missing values
        if prices.isnull().any():
            raise ValidationError("Price data contains missing values")
        
        # Check for infinite values
        if np.isinf(prices).any():
            raise ValidationError("Price data contains infinite values")
        
        # Check for negative prices
        if not allow_negative and (prices < 0).any():
            raise ValidationError("Price data contains negative values")
        
        # Check for zero prices
        if (prices == 0).any():
            warnings.warn("Price data contains zero values")
        
        # Check for monotonic prices
        if prices.nunique() == 1:
            warnings.warn("Price data is constant")
        
        return True
    
    @staticmethod
    def validate_weights(weights: Union[np.ndarray, pd.Series, List[float]],
                        tolerance: float = 1e-6,
                        allow_negative: bool = False,
                        allow_leverage: bool = False) -> bool:
        """
        Validate portfolio weights
        
        Args:
            weights: Portfolio weights
            tolerance: Tolerance for weight sum validation
            allow_negative: Whether to allow negative weights (short positions)
            allow_leverage: Whether to allow leveraged portfolios (sum > 1)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        weights = np.array(weights)
        
        # Check if empty
        if len(weights) == 0:
            raise ValidationError("Weights array is empty")
        
        # Check for missing values
        if np.isnan(weights).any():
            raise ValidationError("Weights contain NaN values")
        
        # Check for infinite values
        if np.isinf(weights).any():
            raise ValidationError("Weights contain infinite values")
        
        # Check for negative weights
        if not allow_negative and (weights < 0).any():
            raise ValidationError("Weights contain negative values")
        
        # Check weight sum
        weight_sum = np.sum(weights)
        
        if not allow_leverage:
            if abs(weight_sum - 1.0) > tolerance:
                raise ValidationError(f"Weights do not sum to 1.0: sum = {weight_sum}")
        else:
            if weight_sum <= 0:
                raise ValidationError(f"Weight sum must be positive: sum = {weight_sum}")
        
        return True
    
    @staticmethod
    def validate_dates(dates: Union[pd.DatetimeIndex, pd.Series, List],
                      min_date: Optional[datetime] = None,
                      max_date: Optional[datetime] = None,
                      check_sorted: bool = True,
                      check_duplicates: bool = True) -> bool:
        """
        Validate date data
        
        Args:
            dates: Date data
            min_date: Minimum allowed date
            max_date: Maximum allowed date
            check_sorted: Whether to check if dates are sorted
            check_duplicates: Whether to check for duplicate dates
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if isinstance(dates, list):
            dates = pd.Series(dates)
        elif isinstance(dates, pd.DatetimeIndex):
            dates = pd.Series(dates)
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(dates):
            try:
                dates = pd.to_datetime(dates)
            except:
                raise ValidationError("Cannot convert dates to datetime format")
        
        # Check if empty
        if len(dates) == 0:
            raise ValidationError("Date data is empty")
        
        # Check for missing values
        if dates.isnull().any():
            raise ValidationError("Date data contains missing values")
        
        # Check date range
        if min_date is not None and (dates < min_date).any():
            raise ValidationError(f"Dates contain values before minimum date: {min_date}")
        
        if max_date is not None and (dates > max_date).any():
            raise ValidationError(f"Dates contain values after maximum date: {max_date}")
        
        # Check if sorted
        if check_sorted and not dates.is_monotonic_increasing:
            raise ValidationError("Dates are not sorted in ascending order")
        
        # Check for duplicates
        if check_duplicates and dates.duplicated().any():
            raise ValidationError("Date data contains duplicate dates")
        
        return True
    
    @staticmethod
    def validate_strategy_parameters(parameters: Dict[str, Any],
                                   parameter_bounds: Dict[str, Tuple[float, float]],
                                   required_parameters: Optional[List[str]] = None) -> bool:
        """
        Validate strategy parameters
        
        Args:
            parameters: Strategy parameters
            parameter_bounds: Dictionary of parameter bounds
            required_parameters: List of required parameters
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check required parameters
        if required_parameters:
            for param in required_parameters:
                if param not in parameters:
                    raise ValidationError(f"Required parameter missing: {param}")
        
        # Check parameter bounds
        for param_name, value in parameters.items():
            if param_name in parameter_bounds:
                min_val, max_val = parameter_bounds[param_name]
                
                if not isinstance(value, (int, float)):
                    continue  # Skip non-numeric parameters
                
                if value < min_val or value > max_val:
                    raise ValidationError(
                        f"Parameter {param_name} = {value} is outside bounds [{min_val}, {max_val}]"
                    )
        
        return True
    
    @staticmethod
    def validate_trade_data(trades: pd.DataFrame,
                           required_columns: Optional[List[str]] = None) -> bool:
        """
        Validate trade data
        
        Args:
            trades: Trade data DataFrame
            required_columns: List of required columns
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        if required_columns is None:
            required_columns = ['timestamp', 'symbol', 'quantity', 'price', 'side']
        
        # Check if DataFrame is empty
        if len(trades) == 0:
            raise ValidationError("Trade data is empty")
        
        # Check required columns
        for col in required_columns:
            if col not in trades.columns:
                raise ValidationError(f"Required column missing: {col}")
        
        # Validate specific columns
        if 'quantity' in trades.columns:
            if trades['quantity'].isnull().any():
                raise ValidationError("Quantity column contains missing values")
            
            if (trades['quantity'] == 0).any():
                warnings.warn("Trade data contains zero quantities")
        
        if 'price' in trades.columns:
            if trades['price'].isnull().any():
                raise ValidationError("Price column contains missing values")
            
            if (trades['price'] <= 0).any():
                raise ValidationError("Price column contains non-positive values")
        
        if 'side' in trades.columns:
            valid_sides = ['buy', 'sell', 'long', 'short', 1, -1]
            if not trades['side'].isin(valid_sides).all():
                raise ValidationError("Side column contains invalid values")
        
        if 'timestamp' in trades.columns:
            DataValidator.validate_dates(trades['timestamp'])
        
        return True
    
    @staticmethod
    def validate_correlation_matrix(corr_matrix: Union[pd.DataFrame, np.ndarray],
                                   tolerance: float = 1e-6) -> bool:
        """
        Validate correlation matrix
        
        Args:
            corr_matrix: Correlation matrix
            tolerance: Numerical tolerance
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        corr_matrix = np.array(corr_matrix)
        
        # Check if square
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            raise ValidationError("Correlation matrix is not square")
        
        # Check if symmetric
        if not np.allclose(corr_matrix, corr_matrix.T, atol=tolerance):
            raise ValidationError("Correlation matrix is not symmetric")
        
        # Check diagonal elements
        diagonal = np.diag(corr_matrix)
        if not np.allclose(diagonal, 1.0, atol=tolerance):
            raise ValidationError("Correlation matrix diagonal elements are not 1.0")
        
        # Check if positive semi-definite
        eigenvalues = np.linalg.eigvals(corr_matrix)
        if (eigenvalues < -tolerance).any():
            raise ValidationError("Correlation matrix is not positive semi-definite")
        
        # Check correlation bounds
        if (corr_matrix > 1 + tolerance).any() or (corr_matrix < -1 - tolerance).any():
            raise ValidationError("Correlation matrix contains values outside [-1, 1]")
        
        return True
    
    @staticmethod
    def validate_performance_metrics(metrics: Dict[str, float]) -> bool:
        """
        Validate performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Check for required metrics
        common_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        
        for metric in common_metrics:
            if metric in metrics:
                value = metrics[metric]
                
                # Check for NaN or infinite values
                if np.isnan(value) or np.isinf(value):
                    warnings.warn(f"Metric {metric} contains NaN or infinite value: {value}")
                
                # Specific validations
                if metric == 'sharpe_ratio' and abs(value) > 10:
                    warnings.warn(f"Unusually high Sharpe ratio: {value}")
                
                if metric == 'max_drawdown' and value > 0:
                    warnings.warn(f"Max drawdown should be negative: {value}")
                
                if metric == 'volatility' and value < 0:
                    warnings.warn(f"Volatility should be positive: {value}")
        
        return True


class ParameterValidator:
    """
    Specialized validator for strategy and optimization parameters
    """
    
    @staticmethod
    def validate_optimization_bounds(bounds: Dict[str, Tuple[float, float]]) -> bool:
        """
        Validate optimization parameter bounds
        
        Args:
            bounds: Dictionary of parameter bounds
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        for param_name, (min_val, max_val) in bounds.items():
            if min_val >= max_val:
                raise ValidationError(f"Invalid bounds for {param_name}: min {min_val} >= max {max_val}")
            
            if np.isnan(min_val) or np.isnan(max_val):
                raise ValidationError(f"Bounds for {param_name} contain NaN values")
            
            if np.isinf(min_val) or np.isinf(max_val):
                raise ValidationError(f"Bounds for {param_name} contain infinite values")
        
        return True
    
    @staticmethod
    def validate_risk_parameters(risk_params: Dict[str, float]) -> bool:
        """
        Validate risk management parameters
        
        Args:
            risk_params: Dictionary of risk parameters
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Common risk parameters and their expected ranges
        param_ranges = {
            'max_position_size': (0.0, 1.0),
            'stop_loss': (0.0, 1.0),
            'take_profit': (0.0, 10.0),
            'max_leverage': (1.0, 10.0),
            'var_limit': (0.0, 1.0),
            'max_drawdown_limit': (0.0, 1.0)
        }
        
        for param_name, value in risk_params.items():
            if param_name in param_ranges:
                min_val, max_val = param_ranges[param_name]
                
                if not isinstance(value, (int, float)):
                    raise ValidationError(f"Risk parameter {param_name} must be numeric")
                
                if value < min_val or value > max_val:
                    raise ValidationError(
                        f"Risk parameter {param_name} = {value} is outside expected range [{min_val}, {max_val}]"
                    )
        
        return True


def validate_input_data(data: Union[pd.DataFrame, pd.Series, np.ndarray],
                       data_type: str = 'returns',
                       **kwargs) -> bool:
    """
    Convenience function for validating input data
    
    Args:
        data: Input data
        data_type: Type of data ('returns', 'prices', 'weights', 'trades')
        **kwargs: Additional validation parameters
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    validator_map = {
        'returns': DataValidator.validate_returns,
        'prices': DataValidator.validate_prices,
        'weights': DataValidator.validate_weights,
        'trades': DataValidator.validate_trade_data
    }
    
    if data_type not in validator_map:
        raise ValidationError(f"Unknown data type: {data_type}")
    
    return validator_map[data_type](data, **kwargs)


def create_validation_report(data: Dict[str, Any],
                           validators: Dict[str, Callable]) -> Dict[str, Dict[str, Any]]:
    """
    Create comprehensive validation report
    
    Args:
        data: Dictionary of data to validate
        validators: Dictionary of validation functions
        
    Returns:
        Validation report dictionary
    """
    report = {}
    
    for data_name, data_value in data.items():
        report[data_name] = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if data_name in validators:
            try:
                # Capture warnings
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    
                    validators[data_name](data_value)
                    
                    # Record warnings
                    for warning in w:
                        report[data_name]['warnings'].append(str(warning.message))
                        
            except ValidationError as e:
                report[data_name]['valid'] = False
                report[data_name]['errors'].append(str(e))
            except Exception as e:
                report[data_name]['valid'] = False
                report[data_name]['errors'].append(f"Unexpected error: {str(e)}")
    
    return report
