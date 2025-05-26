# Performance Optimization Guide

## Overview
This document outlines the performance optimizations implemented in the trading engine, particularly for bootstrap and Monte Carlo analysis.

## Bootstrap Analysis Optimizations

### Development vs Production Configurations

#### Development Mode (Current Default)
- **n_sims**: 100 (vs 1000 in production)
- **method**: IID Bootstrap (vs Block Bootstrap)
- **batch_size**: 50 (vs 100 in production)
- **Impact**: ~10x faster execution (2-5 seconds vs 20-50 seconds per symbol)

#### When to Use Each Mode
- **Development**: Strategy prototyping, parameter testing, quick validation
- **Production**: Final analysis, publication-ready results, regulatory compliance
- **Quick Test**: Initial sanity checks, CI/CD pipelines
- **Research**: Academic research, extended analysis

### Configuration Switching

To switch configurations, modify the bootstrap setup in your analysis:

```python
# Development (default)
config = BootstrapConfig(n_sims=100, batch_size=50, block_length=10)
bootstrapper = AdvancedBootstrapping(
    ret_series=returns,
    method=BootstrapMethod.IID,  # Fastest
    config=config
)

# Production
config = BootstrapConfig(n_sims=1000, batch_size=100, block_length=10)
bootstrapper = AdvancedBootstrapping(
    ret_series=returns,
    method=BootstrapMethod.BLOCK,  # More rigorous
    config=config
)
```

## Monte Carlo Optimizations

### Validation Configuration
- **Development**: 200 bootstrap samples, 200 permutation samples
- **Production**: 1000 bootstrap samples, 1000 permutation samples

## Performance Monitoring

### Execution Timing
All bootstrap and Monte Carlo operations now include timing information:
- Per-symbol bootstrap timing
- Per-symbol validation timing
- Total analysis duration

### Memory Optimization
- Batch processing for large simulations
- Reduced memory footprint for development testing
- Garbage collection optimization

## Best Practices

### For Development
1. Use development configuration for initial testing
2. Profile performance-critical sections
3. Monitor memory usage during large backtests
4. Use quick_test configuration for CI/CD

### For Production
1. Switch to production configuration for final analysis
2. Run extended validation with research configuration
3. Document configuration used in reports
4. Archive results with configuration metadata

## Configuration Files

Use YAML configuration files in `config/` directory:
- `bootstrap_configs.yaml`: Bootstrap-specific configurations
- `batch_configs/`: Batch testing configurations
- `strategy_configs/`: Strategy-specific parameters

## Performance Benchmarks

### Typical Execution Times (per symbol)
- **Quick Test**: < 1 second
- **Development**: 2-5 seconds
- **Production**: 20-50 seconds
- **Research**: 1-5 minutes

### Memory Usage
- **Development**: ~50-100 MB per symbol
- **Production**: ~200-500 MB per symbol
- **Research**: ~500MB-1GB per symbol

## Migration Guide

### From Legacy Configuration
Old configuration files should be updated to use the new profile system:

```yaml
# Old format
bootstrap:
  n_sims: 1000
  batch_size: 100

# New format
bootstrap:
  profile: "development"  # or "production", "quick_test", "research"
```

### Backwards Compatibility
The system maintains backwards compatibility with direct parameter specification while encouraging the use of configuration profiles.
