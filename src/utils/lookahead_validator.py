#!/usr/bin/env python3
"""
Look-Ahead Bias Validator

This module provides tools to detect and prevent look-ahead bias in trading strategies.
Look-ahead bias occurs when a strategy uses future information that wouldn't be
available at the time of making trading decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import warnings


class LookAheadValidator:
    """
    Validator class to detect look-ahead bias in trading strategies
    """
    
    def __init__(self):
        self.violations = []
        self.warnings = []
    
    def validate_strategy_signals(
        self, 
        strategy, 
        data: pd.DataFrame, 
        test_indices: List[int] = None
    ) -> Dict[str, Any]:
        """
        Validate that a strategy doesn't use look-ahead bias.
        
        Args:
            strategy: Strategy instance to test
            data: Historical data
            test_indices: Specific indices to test (if None, tests random sample)
            
        Returns:
            Validation results dictionary
        """
        self.violations = []
        self.warnings = []
        
        if test_indices is None:
            # Test 10 random points in the data
            test_indices = np.random.choice(
                range(50, len(data) - 10), 
                min(10, len(data) - 60), 
                replace=False
            )
        
        for test_idx in test_indices:
            self._test_single_point(strategy, data, test_idx)
        
        # Additional tests
        self._test_data_access_pattern(strategy, data)
        
        return {
            'is_valid': len(self.violations) == 0,
            'violations': self.violations,
            'warnings': self.warnings,
            'tests_performed': len(test_indices),
            'summary': self._generate_summary()
        }
    
    def _test_single_point(self, strategy, data: pd.DataFrame, test_idx: int):
        """Test strategy behavior at a single point in time."""
        
        # Test 1: Signal should be identical when we truncate future data
        historical_data = data.iloc[:test_idx+1]
        
        try:
            # Get signal with full data (this should be prevented by our framework)
            signal_full = strategy.generate_signals(data, test_idx)
            
            # Get signal with only historical data
            signal_historical = strategy.generate_signals(historical_data, test_idx)
            
            if signal_full != signal_historical:
                self.violations.append({
                    'type': 'signal_inconsistency',
                    'index': test_idx,
                    'message': f'Signal differs when future data is removed: {signal_full} vs {signal_historical}'
                })
        except Exception as e:
            self.warnings.append({
                'type': 'signal_generation_error',
                'index': test_idx,
                'message': f'Error generating signal: {str(e)}'
            })
    
    def _test_data_access_pattern(self, strategy, data: pd.DataFrame):
        """Test strategy's data access patterns."""
        
        # Test if strategy method properly handles truncated data
        try:
            # Test with minimal data
            minimal_data = data.iloc[:20]
            strategy.generate_signals(minimal_data, 19)
            
            # Test with gradually increasing data sizes
            for size in [50, 100, min(200, len(data))]:
                if size < len(data):
                    partial_data = data.iloc[:size]
                    strategy.generate_signals(partial_data, size-1)
                    
        except Exception as e:
            self.violations.append({
                'type': 'data_handling_error',
                'message': f'Strategy failed with partial data: {str(e)}'
            })
    
    def _generate_summary(self) -> str:
        """Generate a summary of validation results."""
        if len(self.violations) == 0:
            return "✅ PASS: No look-ahead bias detected"
        else:
            return f"❌ FAIL: {len(self.violations)} look-ahead bias violations detected"


def validate_all_strategies(strategies: List, data: pd.DataFrame) -> Dict[str, Dict]:
    """
    Validate multiple strategies for look-ahead bias.
    
    Args:
        strategies: List of strategy instances
        data: Historical data for testing
        
    Returns:
        Dictionary of validation results for each strategy
    """
    validator = LookAheadValidator()
    results = {}
    
    for strategy in strategies:
        strategy_name = getattr(strategy, 'name', str(strategy))
        print(f"🔍 Validating {strategy_name} for look-ahead bias...")
        
        results[strategy_name] = validator.validate_strategy_signals(strategy, data)
        
        if results[strategy_name]['is_valid']:
            print(f"✅ {strategy_name}: PASSED")
        else:
            print(f"❌ {strategy_name}: FAILED - {len(results[strategy_name]['violations'])} violations")
            for violation in results[strategy_name]['violations']:
                print(f"   - {violation['message']}")
    
    return results


def create_bias_detection_report(validation_results: Dict[str, Dict]) -> str:
    """Create a detailed report of look-ahead bias detection results."""
    
    report = ["# Look-Ahead Bias Detection Report\n"]
    report.append(f"Generated: {pd.Timestamp.now()}\n")
    
    total_strategies = len(validation_results)
    passed_strategies = sum(1 for r in validation_results.values() if r['is_valid'])
    
    report.append(f"## Summary")
    report.append(f"- Total strategies tested: {total_strategies}")
    report.append(f"- Strategies passed: {passed_strategies}")
    report.append(f"- Strategies failed: {total_strategies - passed_strategies}")
    report.append(f"- Success rate: {passed_strategies/total_strategies*100:.1f}%\n")
    
    for strategy_name, results in validation_results.items():
        report.append(f"## {strategy_name}")
        report.append(f"**Status:** {'✅ PASSED' if results['is_valid'] else '❌ FAILED'}")
        report.append(f"**Tests performed:** {results['tests_performed']}")
        
        if results['violations']:
            report.append("**Violations:**")
            for violation in results['violations']:
                report.append(f"- {violation['message']}")
        
        if results['warnings']:
            report.append("**Warnings:**")
            for warning in results['warnings']:
                report.append(f"- {warning['message']}")
        
        report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    print("Look-Ahead Bias Validator")
    print("This module provides tools to detect look-ahead bias in trading strategies.")
