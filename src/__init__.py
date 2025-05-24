"""
Trading Strategy Analyzer

A comprehensive framework for analyzing trading strategies using advanced 
statistical methods, Monte Carlo simulations, and robust backtesting techniques.
"""

__version__ = "1.0.0"
__author__ = "Trading Strategy Analyzer Team"
__email__ = "contact@tradingstrategyanalyzer.com"

from .bootstrapping import AdvancedBootstrapping
from .strategies import StrategyTestSuite
from .utils import load_sample_data, calculate_metrics
from .analysis import PerformanceAnalyzer, RiskAnalyzer

__all__ = [
    "AdvancedBootstrapping",
    "StrategyTestSuite", 
    "PerformanceAnalyzer",
    "RiskAnalyzer",
    "load_sample_data",
    "calculate_metrics",
]
