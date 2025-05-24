"""
Trading Strategies Module

This module contains various trading strategy implementations for backtesting
and analysis. All strategies inherit from the BaseStrategy class.
"""

from .base_strategy import BaseStrategy, Signal, Position, Trade
from .strategy_suite import StrategyTestSuite

# Momentum strategies
from .momentum import (
    MomentumStrategy,
    VolumeMomentumStrategy, 
    RiskAdjustedMomentumStrategy,
    DualMomentumStrategy,
    CrossSectionalMomentumStrategy
)

# Mean reversion strategies
from .mean_reversion import (
    MeanReversionStrategy,  # Added this line
    BollingerBandsStrategy,
    StatisticalArbitrageStrategy,
    PairsStrategy,
    OrnsteinUhlenbeckStrategy
)

# Trend following strategies
from .trend_following import (
    MovingAverageCrossoverStrategy,
    TripleMovingAverageStrategy,
    BreakoutStrategy,
    ChannelBreakoutStrategy,
    TrendFilterStrategy
)

__all__ = [
    # Base classes
    'BaseStrategy', 'Signal', 'Position', 'Trade', 'StrategyTestSuite',
    
    # Momentum strategies
    'MomentumStrategy', 'VolumeMomentumStrategy', 'RiskAdjustedMomentumStrategy',
    'DualMomentumStrategy', 'CrossSectionalMomentumStrategy',
    
    # Mean reversion strategies
    'MeanReversionStrategy',  # Added this line
    'BollingerBandsStrategy', 'StatisticalArbitrageStrategy', 
    'PairsStrategy', 'OrnsteinUhlenbeckStrategy',
    
    # Trend following strategies
    'MovingAverageCrossoverStrategy', 'TripleMovingAverageStrategy',
    'BreakoutStrategy', 'ChannelBreakoutStrategy', 'TrendFilterStrategy'
]
