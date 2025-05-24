"""
Utilities Module

This module provides utility functions for data loading, preprocessing,
and common calculations used throughout the backtesting framework.
"""

from .data_loader import DataLoader, load_sample_data, generate_synthetic_data, generate_multi_asset_data
from .metrics import PerformanceMetrics, RiskMetrics, calculate_metrics
from .validation import validate_input_data, create_validation_report, DataValidator
from .portfolio import PortfolioUtils, rebalance_portfolio
from .optimization import ParameterOptimizer, GridSearchOptimizer, BayesianOptimizer

__all__ = [
    'DataLoader', 'load_sample_data', 'generate_synthetic_data', 'generate_multi_asset_data',
    'PerformanceMetrics', 'RiskMetrics', 'calculate_metrics',
    'validate_input_data', 'create_validation_report', 'DataValidator',
    'PortfolioUtils', 'rebalance_portfolio',
    'ParameterOptimizer', 'GridSearchOptimizer', 'BayesianOptimizer'
]
