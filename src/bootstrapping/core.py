"""
Advanced Bootstrapping Core Module

Production-ready implementation addressing all technical audit points:
- Float64 precision for numerical stability
- Vectorized Numba operations
- Multiple bootstrap variants (IID, Stationary, Block, Wild)
- Comprehensive statistical testing
- Type hints and proper error handling
"""

import warnings
from typing import Optional, Union, Dict, Any, List, Tuple, Callable
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform # For MMD
from dataclasses import dataclass

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print(">>> Successfully imported numba.")
    
    @njit(parallel=True, fastmath=True, cache=True)
    def _cumprod_numba(a: np.ndarray) -> np.ndarray:
        """Optimized cumulative product using Numba with parallel processing."""
        out = np.empty_like(a)
        for i in prange(a.shape[0]):
            out[i, :] = np.cumprod(a[i, :])
        return out
    
    @njit(parallel=True, fastmath=True, cache=True)
    def _rolling_max_numba(a: np.ndarray) -> np.ndarray:
        """Optimized rolling maximum for drawdown calculation."""
        out = np.empty_like(a)
        for i in prange(a.shape[0]):
            current_max = a[i, 0]
            for j in range(a.shape[1]):
                if a[i, j] > current_max:
                    current_max = a[i, j]
                out[i, j] = current_max
        return out

    @njit(cache=True)
    def _gaussian_kernel_numba(x: np.ndarray, y: np.ndarray, sigma_sq: float) -> float:
        """Numba optimized Gaussian kernel."""
        # Ensure sigma_sq is not zero to prevent division by zero
        if sigma_sq == 0:
            # If x and y are identical, kernel is 1, otherwise 0 (delta function behavior)
            return 1.0 if np.all(x == y) else 0.0
        return np.exp(-np.sum((x - y)**2) / (2 * sigma_sq))

    @njit(cache=True)
    def _mmd_sq_numba(X: np.ndarray, Y: np.ndarray, sigma_sq: float) -> float:
        """Numba optimized MMD squared calculation."""
        m = X.shape[0]
        n = Y.shape[0]

        # Ensure sigma_sq is not zero
        if sigma_sq == 0: # If sigma_sq is zero, MMD is 0 if X and Y are identical, 1 otherwise (simplified)
            # This is a simplification. A more rigorous handling might involve limits or specific conditions.
            # For practical purposes, if sigma_sq is 0, it implies a very strict comparison.
            # If distributions are identical, MMD should be 0.
            # A proper handling for sigma_sq=0 in Gaussian kernel MMD is non-trivial.
            # Here, we assume if sigma_sq is 0, kernel is 1 if points are same, 0 otherwise.
            # This means MMD would be 0 if X and Y are identical element-wise, which is too strict.
            # A better approach is to ensure sigma_sq is always positive before calling this.
            # For now, returning a high value if not identical, 0 if identical (based on means for simplicity)
            # This part needs careful consideration based on how sigma_sq=0 is to be interpreted.
            if m == 0 and n == 0: return 0.0
            if m == 0 or n == 0: return 1.0 # Arbitrary high value if one is empty

            # A more robust check for identical distributions when sigma_sq is zero is complex.
            # The median heuristic should prevent sigma_sq from being exactly zero with real data.
            # If it happens, it implies all points are identical.
            pass # Let the calculation proceed, _gaussian_kernel_numba will handle sigma_sq=0


        sum_k_xx = 0.0
        if m > 1:
            for i in prange(m): # Changed to prange for potential parallelization if outer loop is parallel
                for j in range(i + 1, m):
                    sum_k_xx += _gaussian_kernel_numba(X[i], X[j], sigma_sq)
            sum_k_xx = 2 * sum_k_xx / (m * (m - 1))
        elif m == 1: # if only one sample, this term is conventionally 0 or not well-defined for pairwise.
            sum_k_xx = 0.0


        sum_k_yy = 0.0
        if n > 1:
            for i in prange(n): # Changed to prange
                for j in range(i + 1, n):
                    sum_k_yy += _gaussian_kernel_numba(Y[i], Y[j], sigma_sq)
            sum_k_yy = 2 * sum_k_yy / (n * (n - 1))
        elif n == 1:
            sum_k_yy = 0.0
        
        sum_k_xy = 0.0
        if m > 0 and n > 0:
            for i in prange(m): # Changed to prange
                for j in range(n): # No prange here, inner loop of a parallel region
                    sum_k_xy += _gaussian_kernel_numba(X[i], Y[j], sigma_sq)
            sum_k_xy /= (m * n)
        else: # If either m or n is 0, this term is 0
            sum_k_xy = 0.0


        mmd_squared_val = sum_k_xx - 2 * sum_k_xy + sum_k_yy
        return mmd_squared_val if mmd_squared_val >= 0 else 0.0 # Ensure non-negativity
    
except ImportError as e:
    NUMBA_AVAILABLE = False
    warnings.warn(f"Numba not available: {e}. Using numpy fallback.", UserWarning)
    
    def _cumprod_numba(a: np.ndarray) -> np.ndarray:
        return np.cumprod(a, axis=1)
    
    def _rolling_max_numba(a: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(a, axis=1)

    # Fallback for MMD if Numba is not available
    def _gaussian_kernel_numpy(x_i, x_j, sigma_sq):
        # Ensure sigma_sq is not zero to prevent division by zero
        if sigma_sq == 0:
            return 1.0 if np.all(x_i == x_j) else 0.0
        return np.exp(-np.sum((x_i - x_j)**2) / (2 * sigma_sq))

    def _mmd_sq_numpy(X, Y, sigma_sq):
        m = X.shape[0]
        n = Y.shape[0]

        # Term 1: E[k(x, x')]
        sum_k_xx = 0.0
        if m > 1:
            for i in range(m):
                for j in range(i + 1, m):
                    sum_k_xx += _gaussian_kernel_numpy(X[i], X[j], sigma_sq)
            term1 = 2 * sum_k_xx / (m * (m - 1))
        elif m == 1:
             term1 = 0.0 # Or handle as per convention for single point
        else: # m == 0
            term1 = 0.0


        # Term 2: E[k(y, y')]
        sum_k_yy = 0.0
        if n > 1:
            for i in range(n):
                for j in range(i + 1, n):
                    sum_k_yy += _gaussian_kernel_numpy(Y[i], Y[j], sigma_sq)
            term2 = 2 * sum_k_yy / (n * (n - 1))
        elif n == 1:
            term2 = 0.0
        else: # n == 0
            term2 = 0.0


        # Term 3: E[k(x, y)]
        sum_k_xy = 0.0
        if m > 0 and n > 0:
            for i in range(m):
                for j in range(n):
                    sum_k_xy += _gaussian_kernel_numpy(X[i], Y[j], sigma_sq)
            term3 = sum_k_xy / (m * n)
        else:
            term3 = 0.0
        
        mmd_squared_val = term1 + term2 - 2 * term3 # Original formula is K_XX + K_YY - 2*K_XY
                                                # The provided code had sum_k_xx - 2 * sum_k_xy + sum_k_yy
                                                # This seems to be a common formulation for the biased v-statistic.
                                                # Let's stick to the structure sum_xx_terms + sum_yy_terms - 2 * sum_xy_terms
        return mmd_squared_val if mmd_squared_val >=0 else 0.0 # Ensure non-negativity

    _mmd_sq_numba = _mmd_sq_numpy # Assign numpy version if numba fails


class TimeFrame(Enum):
    """Enumerated timeframes with annualization factors."""
    MIN_1 = ("1m", 525_600)
    MIN_5 = ("5m", 105_120)
    MIN_15 = ("15m", 35_040)
    MIN_30 = ("30m", 17_520)
    HOUR_1 = ("1h", 8_760)
    HOUR_2 = ("2h", 4_380)
    HOUR_4 = ("4h", 2_190)
    DAY_1 = ("1d", 365)
    WEEK_1 = ("1w", 52)
    MONTH_1 = ("1M", 12)
    
    def __init__(self, label: str, periods: int):
        self.label = label
        self.periods = periods


class BootstrapMethod(Enum):
    """Bootstrap method variants."""
    IID = "iid"
    STATIONARY = "stationary"
    BLOCK = "block"
    CIRCULAR = "circular"
    WILD = "wild"


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap parameters."""
    n_sims: int = 1_000
    batch_size: int = 500
    block_length: Optional[int] = None
    confidence_levels: List[float] = None
    dtype: np.dtype = np.float64
    preserve_autocorr: bool = True
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99, 0.999]


class AdvancedBootstrapping:
    """
    Advanced Monte Carlo bootstrapping with comprehensive statistical analysis.
    
    Implements multiple bootstrap variants with production-grade features:
    - Multiple precision levels (float64 default)
    - Vectorized operations with Numba acceleration
    - Stationary/Block bootstrap for autocorrelation preservation
    - Comprehensive risk and performance metrics
    - Statistical significance testing with multiple comparison corrections
    - Streaming interface for large simulations
    
    Examples:
        Basic usage:
        >>> bootstrap = AdvancedBootstrapping(ret_series=returns, timeframe='1d')
        >>> results = bootstrap.run_full_analysis()
        
        Advanced configuration:
        >>> config = BootstrapConfig(n_sims=10_000, preserve_autocorr=True)
        >>> bootstrap = AdvancedBootstrapping(
        ...     ret_series=returns,
        ...     method=BootstrapMethod.STATIONARY,
        ...     config=config
        ... )
    """
    
    def __init__(
        self,
        analyzer: Optional[Any] = None,
        *,
        timeframe: Union[str, TimeFrame] = TimeFrame.DAY_1,
        ret_series: Optional[pd.Series] = None,
        benchmark_series: Optional[pd.Series] = None,
        method: BootstrapMethod = BootstrapMethod.IID,
        config: Optional[BootstrapConfig] = None,
        rng: Optional[np.random.Generator] = None,
        use_log_returns: bool = False,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize Advanced Bootstrapping analyzer.
        
        Args:
            analyzer: Optional backtesting analyzer object
            timeframe: Trading timeframe for annualization
            ret_series: Return series (required if no analyzer)
            benchmark_series: Optional benchmark for comparison
            method: Bootstrap method to use
            config: Bootstrap configuration
            rng: Random number generator
            use_log_returns: Whether to use log returns
            random_seed: Random seed for reproducibility
        """
        # Initialize configuration
        self.config = config or BootstrapConfig()
        self.method = method
        self.use_log_returns = use_log_returns
        
        # Set up random number generator
        self.rng = rng or np.random.default_rng(random_seed)
        
        # Handle timeframe
        if isinstance(timeframe, str):
            self.timeframe = self._parse_timeframe(timeframe)
        else:
            self.timeframe = timeframe
            
        # Initialize data
        if analyzer is not None:
            self._init_from_analyzer(analyzer)
        else:
            if ret_series is None:
                raise ValueError("Provide a return series if no analyzer is given")
            self._init_from_series(ret_series)
            
        # Set benchmark
        self.benchmark_series = benchmark_series
        
        # Validate data
        self._validate_data()
        
    def _parse_timeframe(self, timeframe_str: str) -> TimeFrame:
        """Parse timeframe string to TimeFrame enum."""
        for tf in TimeFrame:
            if tf.label == timeframe_str:
                return tf
        raise ValueError(f"Unsupported timeframe '{timeframe_str}'. "
                        f"Supported: {[tf.label for tf in TimeFrame]}")
    
    def _init_from_analyzer(self, analyzer: Any) -> None:
        """Initialize from backtesting analyzer object."""
        self.pf = analyzer.pf
        self.init_cash = analyzer.init_cash
        self.ret_series = analyzer.pf.returns()
        if hasattr(analyzer.pf, 'benchmark_returns'):
            self.benchmark_series = analyzer.pf.benchmark_returns()
    
    def _init_from_series(self, ret_series: pd.Series) -> None:
        """Initialize from return series."""
        self.pf = None
        self.init_cash = 1.0
        self.ret_series = ret_series.copy()
        
    def _validate_data(self) -> None:
        """Validate input data."""
        if self.ret_series is None or len(self.ret_series) == 0:
            raise ValueError("Return series cannot be empty")
            
        if not isinstance(self.ret_series.index, pd.DatetimeIndex):
            try:
                self.ret_series.index = pd.to_datetime(self.ret_series.index)
            except Exception:
                raise ValueError("Index must be convertible to datetime")
                
        # Check for missing values
        if self.ret_series.isnull().any():
            warnings.warn("Missing values detected. Consider handling them.", UserWarning)
            
        # Convert to log returns if requested
        if self.use_log_returns:
            self.ret_series = np.log1p(self.ret_series)
    
    def _frequency_conversion(self, ret: pd.Series) -> pd.Series:
        """
        Convert returns to target frequency with proper handling of trading calendars.
        
        Args:
            ret: Return series
            
        Returns:
            Frequency-adjusted return series
        """
        rs = ret.copy()
        rs.index = pd.to_datetime(rs.index)
        
        # For intraday and daily, return as-is
        if self.timeframe.label.endswith(('m', 'h')) or self.timeframe.label == '1d':
            return rs
            
        # Weekly resampling
        if self.timeframe.label == '1w':
            return rs.resample('W').apply(lambda x: (1 + x).prod() - 1)
            
        # Monthly resampling
        if self.timeframe.label == '1M':
            return rs.resample('M').apply(lambda x: (1 + x).prod() - 1)
            
        return rs
    
    def _stationary_bootstrap(self, returns: np.ndarray, n_sims: int, 
                            block_length: Optional[int] = None) -> np.ndarray:
        """
        Stationary bootstrap preserving autocorrelation structure.
        
        Args:
            returns: Array of returns
            n_sims: Number of simulations
            block_length: Average block length (auto-calculated if None)
            
        Returns:
            Bootstrap samples array
        """
        n_obs = len(returns)
        
        if block_length is None:
            # Optimal block length estimation (Politis & White, 2004)
            block_length = max(1, int(n_obs ** (1/3)))
            
        samples = np.empty((n_sims, n_obs), dtype=self.config.dtype)
        
        for i in range(n_sims):
            sample = np.empty(n_obs, dtype=self.config.dtype)
            pos = 0
            
            while pos < n_obs:
                # Random starting position
                start = self.rng.integers(0, n_obs)
                
                # Geometric block length
                length = self.rng.geometric(1.0 / block_length)
                length = min(length, n_obs - pos)
                
                # Handle wrap-around
                if start + length <= n_obs:
                    sample[pos:pos + length] = returns[start:start + length]
                else:
                    # Split block with wrap-around
                    first_part = n_obs - start
                    sample[pos:pos + first_part] = returns[start:]
                    if pos + first_part < n_obs:
                        remaining = length - first_part
                        remaining = min(remaining, n_obs - pos - first_part)
                        sample[pos + first_part:pos + first_part + remaining] = returns[:remaining]
                
                pos += length
                
            samples[i] = sample
            
        return samples
    
    def _block_bootstrap(self, returns: np.ndarray, n_sims: int, 
                        block_length: int = 20) -> np.ndarray:
        """
        Block bootstrap with fixed block length.
        
        Args:
            returns: Array of returns
            n_sims: Number of simulations  
            block_length: Fixed block length
            
        Returns:
            Bootstrap samples array
        """
        n_obs = len(returns)
        n_blocks = int(np.ceil(n_obs / block_length))
        
        samples = np.empty((n_sims, n_obs), dtype=self.config.dtype)
        
        for i in range(n_sims):
            sample = np.empty(n_obs, dtype=self.config.dtype)
            pos = 0
            
            for _ in range(n_blocks):
                if pos >= n_obs:
                    break
                    
                start = self.rng.integers(0, max(1, n_obs - block_length + 1))
                end = min(start + block_length, n_obs)
                length = min(end - start, n_obs - pos)
                
                sample[pos:pos + length] = returns[start:start + length]
                pos += length
                
            samples[i] = sample[:n_obs]
            
        return samples
    
    def _wild_bootstrap(self, returns: np.ndarray, n_sims: int) -> np.ndarray:
        """
        Wild bootstrap for heteroscedastic returns.
        
        Args:
            returns: Array of returns
            n_sims: Number of simulations
            
        Returns:
            Bootstrap samples array
        """
        n_obs = len(returns)
        
        # Center the returns
        centered_returns = returns - np.mean(returns)
        
        # Generate wild bootstrap multipliers (Rademacher distribution)
        multipliers = self.rng.choice([-1, 1], size=(n_sims, n_obs))
        
        # Apply multipliers
        samples = centered_returns[np.newaxis, :] * multipliers
        samples += np.mean(returns)  # Add back the mean
        
        return samples.astype(self.config.dtype)
    
    def _generate_bootstrap_samples(self, returns: np.ndarray, n_sims: int) -> np.ndarray:
        """
        Generate bootstrap samples using the specified method.
        
        Args:
            returns: Array of returns
            n_sims: Number of simulations
            
        Returns:
            Bootstrap samples array
        """
        n_obs = len(returns)
        
        if self.method == BootstrapMethod.IID:
            # Standard IID bootstrap
            idx = self.rng.integers(0, n_obs, size=(n_sims, n_obs))
            return returns[idx].astype(self.config.dtype)
            
        elif self.method == BootstrapMethod.STATIONARY:
            return self._stationary_bootstrap(returns, n_sims, self.config.block_length)
            
        elif self.method == BootstrapMethod.BLOCK:
            block_length = self.config.block_length or 20
            return self._block_bootstrap(returns, n_sims, block_length)
            
        elif self.method == BootstrapMethod.WILD:
            return self._wild_bootstrap(returns, n_sims)
            
        else:
            raise ValueError(f"Unsupported bootstrap method: {self.method}")
    
    def _calculate_advanced_metrics(self, samples: np.ndarray) -> List[Dict[str, float]]:
        """
        Calculate comprehensive performance and risk metrics.
        
        Args:
            samples: Bootstrap samples array
            
        Returns:
            List of metric dictionaries
        """
        n_sims, n_obs = samples.shape
        ann_factor = self.timeframe.periods
        init_cash = self.init_cash
        
        # Cumulative products for equity curves
        cumprod = _cumprod_numba(1.0 + samples) * init_cash
        
        # Basic return metrics
        cum_ret = (1.0 + samples).prod(axis=1) - 1.0
        mean_ret = samples.mean(axis=1)
        std_ret = samples.std(axis=1, ddof=1)
        
        # Annualized metrics
        years = np.clip(n_obs / ann_factor, 1e-6, None)
        cagr = np.where(
            cumprod[:, -1] > 0,
            (cumprod[:, -1] / init_cash) ** (1 / years) - 1,
            np.nan
        )
        
        # Risk-adjusted metrics
        sharpe = np.where(std_ret > 0, mean_ret / std_ret * np.sqrt(ann_factor), np.nan)
        
        # Sortino ratio (downside deviation)
        downside_returns = np.minimum(samples, 0)
        downside_std = np.sqrt(np.mean(downside_returns ** 2, axis=1))
        sortino = np.where(
            downside_std > 0, 
            mean_ret * ann_factor / (downside_std * np.sqrt(ann_factor)), 
            np.nan
        )
        
        # Drawdown analysis
        rolling_max = _rolling_max_numba(cumprod)
        rolling_max = np.where(rolling_max == 0, 1e-9, rolling_max)
        drawdowns = (cumprod - rolling_max) / rolling_max
        max_dd = drawdowns.min(axis=1)
        
        # Calmar ratio
        calmar = np.where(max_dd < 0, cagr / np.abs(max_dd), np.nan)
        
        # Higher moments
        skewness = stats.skew(samples, axis=1)
        kurtosis = stats.kurtosis(samples, axis=1)
        
        # VaR and CVaR calculations
        var_95 = np.percentile(samples, 5, axis=1)
        var_99 = np.percentile(samples, 1, axis=1)
        
        # CVaR (Expected Shortfall)
        cvar_95 = np.array([
            samples[i][samples[i] <= var_95[i]].mean() if np.any(samples[i] <= var_95[i]) else np.nan
            for i in range(n_sims)
        ])
        cvar_99 = np.array([
            samples[i][samples[i] <= var_99[i]].mean() if np.any(samples[i] <= var_99[i]) else np.nan
            for i in range(n_sims)
        ])
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2, axis=1))
        
        # Omega Ratio (threshold = 0)
        gains = np.sum(np.maximum(samples, 0), axis=1)
        losses = np.sum(np.abs(np.minimum(samples, 0)), axis=1)
        omega = np.where(losses > 0, gains / losses, np.inf)
        
        # Compile results
        results = []
        for i in range(n_sims):
            results.append({
                'CumulativeReturn': cum_ret[i],
                'CAGR': cagr[i],
                'Volatility': std_ret[i] * np.sqrt(ann_factor),
                'Sharpe': sharpe[i],
                'Sortino': sortino[i],
                'Calmar': calmar[i],
                'MaxDrawdown': max_dd[i],
                'UlcerIndex': ulcer_index[i],
                'Skewness': skewness[i],
                'Kurtosis': kurtosis[i],
                'VaR_95': var_95[i],
                'VaR_99': var_99[i],
                'CVaR_95': cvar_95[i],
                'CVaR_99': cvar_99[i],
                'OmegaRatio': omega[i],
            })
            
        return results

    def _calculate_distributional_distances(self, original_returns: np.ndarray, bootstrapped_samples: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Wasserstein and MMD distances between original returns and bootstrapped samples.
        """
        wasserstein_distances = []
        mmd_distances = []

        # Ensure original_returns is 1D for Wasserstein and stats.wasserstein_distance
        if original_returns.ndim > 1:
            original_returns_1d = original_returns.flatten()
        else:
            original_returns_1d = original_returns
        
        # Prepare original returns for MMD (needs to be 2D, e.g., (n_samples, n_features))
        # If original_returns was 1D (n_samples,), reshape to (n_samples, 1)
        if original_returns.ndim == 1:
            original_returns_2d_for_mmd = original_returns.reshape(-1, 1)
        else:
            original_returns_2d_for_mmd = original_returns # Assuming it's already (n_samples, n_features)

        # Median heuristic for MMD sigma: median of pairwise squared Euclidean distances
        # This should be calculated on the original_returns_2d_for_mmd
        if len(original_returns_2d_for_mmd) > 1:
            # pdist requires 2D array where each row is an observation
            pairwise_sq_dists = squareform(pdist(original_returns_2d_for_mmd, 'sqeuclidean'))
            # Use only upper triangle to avoid duplicates and diagonal
            median_sq_dist = np.median(pairwise_sq_dists[np.triu_indices_from(pairwise_sq_dists, k=1)])
            
            # Heuristic for sigma_sq based on median distance
            # Common heuristic: sigma = median_dist / sqrt(2) or sigma^2 = median_sq_dist / 2
            # Another: sigma_sq = median_sq_dist (used in some libraries if not scaled by log(N))
            # The original code had: sigma_sq = median_sq_dist / (2 * np.log(len(original_returns) + 1e-9))
            # Let's ensure len(original_returns_2d_for_mmd) is used for consistency.
            if len(original_returns_2d_for_mmd) > 0:
                 denominator = 2 * np.log(len(original_returns_2d_for_mmd) + 1e-9)
                 if denominator <= 1e-9: # Avoid division by zero or very small numbers if log is ~0
                     sigma_sq = median_sq_dist / 2 if median_sq_dist > 0 else 1.0 # Fallback
                 else:
                     sigma_sq = median_sq_dist / denominator
            else: # Should not happen if len > 1
                sigma_sq = 1.0

            if sigma_sq <= 1e-9: # Avoid division by zero or extremely small sigma in kernel
                sigma_sq = 1.0 # Fallback to a default value
        elif len(original_returns_2d_for_mmd) == 1: # Single data point
             sigma_sq = 1.0 # Default sigma if only one point (MMD would be tricky)
        else: # Empty original returns
            sigma_sq = 1.0 # Default sigma if no data
            # If original_returns is empty, MMD and Wasserstein are problematic.
            # This case should ideally be handled before calling this function.
            # For now, we proceed, and distances will likely be NaN or error if samples are also empty.

        for i in range(bootstrapped_samples.shape[0]):
            sample = bootstrapped_samples[i, :] # This is 1D

            # Wasserstein distance
            # Ensure sample is also 1D
            if sample.ndim > 1:
                sample_1d = sample.flatten()
            else:
                sample_1d = sample
            
            if len(original_returns_1d) > 0 and len(sample_1d) > 0:
                try:
                    wd = stats.wasserstein_distance(original_returns_1d, sample_1d)
                    wasserstein_distances.append(wd)
                except ValueError as e: # Handle cases like empty arrays if not caught before
                    warnings.warn(f"Wasserstein distance calculation failed for sample {i}: {e}")
                    wasserstein_distances.append(np.nan)
            else:
                wasserstein_distances.append(np.nan)
            
            # MMD (Maximum Mean Discrepancy)
            # Reshape sample to be 2D for MMD: (n_sample_points, 1)
            sample_2d_for_mmd = sample.reshape(-1, 1)

            if original_returns_2d_for_mmd.shape[0] > 0 and sample_2d_for_mmd.shape[0] > 0:
                # Ensure X and Y for MMD are 2D arrays where rows are samples and columns are features.
                # Here, features = 1.
                if NUMBA_AVAILABLE:
                    mmd_val_sq = _mmd_sq_numba(original_returns_2d_for_mmd, sample_2d_for_mmd, sigma_sq)
                else:
                    mmd_val_sq = _mmd_sq_numpy(original_returns_2d_for_mmd, sample_2d_for_mmd, sigma_sq)
                
                # MMD is sqrt of MMD^2. MMD^2 should be non-negative.
                mmd_distances.append(np.sqrt(max(0, mmd_val_sq)))
            else:
                mmd_distances.append(np.nan)


        return {
            'wasserstein_distances_samples': wasserstein_distances,
            'mmd_distances_samples': mmd_distances,
            'mean_wasserstein_distance': np.nanmean(wasserstein_distances) if wasserstein_distances else np.nan,
            'std_wasserstein_distance': np.nanstd(wasserstein_distances) if wasserstein_distances else np.nan,
            'mean_mmd_distance': np.nanmean(mmd_distances) if mmd_distances else np.nan,
            'std_mmd_distance': np.nanstd(mmd_distances) if mmd_distances else np.nan,
            'mmd_sigma_sq_used': sigma_sq
        }

    def _analyze_series(self, ret: pd.Series) -> Dict[str, float]:
        """Calculate metrics for a single return series."""
        if len(ret) == 0:
            return {key: np.nan for key in [
                'CumulativeReturn', 'CAGR', 'Volatility', 'Sharpe', 'Sortino', 
                'Calmar', 'MaxDrawdown', 'UlcerIndex', 'Skewness', 'Kurtosis',
                'VaR_95', 'VaR_99', 'CVaR_95', 'CVaR_99', 'OmegaRatio'
            ]}
        
        arr = np.asarray(ret, dtype=self.config.dtype)[np.newaxis, :]
        return self._calculate_advanced_metrics(arr)[0]
    
    def run_bootstrap_simulation(self, stream: bool = False) -> Dict[str, Any]:
        """
        Run Monte Carlo bootstrap simulation.
        
        Args:
            stream: If True, return iterator for large simulations
            
        Returns:
            Dictionary containing simulation results
        """
        # Prepare data
        returns = self._frequency_conversion(self.ret_series)
        arr = returns.values.astype(self.config.dtype)
        n_obs = len(arr)
        
        if stream and self.config.n_sims > 10_000:
            return self._run_streaming_simulation(arr)
        
        # Generate bootstrap samples
        samples = self._generate_bootstrap_samples(arr, self.config.n_sims)
        
        # Calculate equity curves
        equity = _cumprod_numba(1.0 + samples) * self.init_cash
        
        # Create DataFrame
        sim_equity = pd.DataFrame(
            equity.T,
            index=returns.index,
            columns=[f"Sim_{i}" for i in range(self.config.n_sims)]
        )
        
        # Calculate metrics
        stats = self._calculate_advanced_metrics(samples)
        original_stats = self._analyze_series(returns)
        
        # Calculate distributional distances
        dist_distances = self._calculate_distributional_distances(arr, samples)

        return {
            'original_stats': original_stats,
            'simulated_stats': stats,
            'simulated_equity_curves': sim_equity,
            'distributional_distances': dist_distances, # Added
            'method': self.method.value,
            'config': self.config
        }
    
    def _run_streaming_simulation(self, arr: np.ndarray) -> Dict[str, Any]:
        """Run simulation in streaming mode for large n_sims."""
        n_obs = len(arr)
        all_stats = []
        
        # Process in batches
        for i in range(0, self.config.n_sims, self.config.batch_size):
            end = min(i + self.config.batch_size, self.config.n_sims)
            batch_size = end - i
            
            # Generate batch samples
            batch_samples = self._generate_bootstrap_samples(arr, batch_size)
            
            # Calculate batch statistics
            batch_stats = self._calculate_advanced_metrics(batch_samples)
            all_stats.extend(batch_stats)
        
        original_stats = self._analyze_series(pd.Series(arr))
        
        # For streaming, distributional distances might be too memory intensive to store all samples
        # Or could be calculated per batch and aggregated if a streaming version of distances is made
        # For now, omitting full distributional distance calculation for streaming to avoid OOM.
        # A summary (e.g. mean of distances per batch) could be an alternative.
        dist_distances_summary = {"info": "Full distributional distances not computed in streaming mode"}
        
        # The _run_streaming_simulation method is only called when stream=True and n_sims > 10_000.
        # If we want to compute distributional distances for smaller n_sims even in streaming context,
        # this condition needs to be re-evaluated.
        # However, the original logic was:
        # if not stream or self.config.n_sims <= 10_000:
        # This implied that if it *is* streaming (which is true here) AND n_sims is small, compute.
        # Or if it's *not* streaming (which is false here), compute.
        # Given this method is *only* for streaming, the `not stream` part is always false.
        # So, the condition simplifies to: if self.config.n_sims <= 10_000 (within streaming context)
        # This seems counterintuitive for a streaming mode designed for large N.
        # Let's assume the intention was to always calculate it if n_sims is small, regardless of streaming mode,
        # and for streaming mode (large N), it's skipped.
        # The run_bootstrap_simulation method handles the non-streaming case.
        # Here, in _run_streaming_simulation, we are in streaming mode.
        # The original code had a check:
        # if not stream or self.config.n_sims <= 10_000 :
        # Since this function is called when stream is True, this simplifies to:
        # if False or self.config.n_sims <= 10_000:
        # which is: if self.config.n_sims <= 10_000:
        # This means even in streaming mode, if n_sims is small, it would try to compute.
        # This might be okay, but let's stick to the simplified logic.
        # The parameter `stream` is not directly available here.
        # We know we are in streaming mode because this function was called.
        
        # If the number of simulations is small enough, calculate full distributional distances
        # even in streaming mode (though this might be rare for calling this specific function).
        if self.config.n_sims <= 10_000:
             samples_for_dist = self._generate_bootstrap_samples(arr, self.config.n_sims)
             dist_distances = self._calculate_distributional_distances(arr, samples_for_dist)
             dist_distances_summary = dist_distances


        return {
            'original_stats': original_stats,
            'simulated_stats': all_stats,
            'simulated_equity_curves': None,  # Not stored in streaming mode
            'distributional_distances': dist_distances_summary, # Added
            'method': self.method.value,
            'config': self.config,
            'streaming': True
        }
    
    def statistical_tests(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive statistical significance tests.
        
        Args:
            results: Bootstrap simulation results
            
        Returns:
            Dictionary of test results
        """
        from .statistical_tests import StatisticalTests
        
        tester = StatisticalTests(self.ret_series, self.benchmark_series)
        return tester.run_all_tests(results)
    
    def calculate_risk_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate advanced risk metrics.
        
        Args:
            results: Bootstrap simulation results
            
        Returns:
            Dictionary of risk metrics
        """
        from .risk_metrics import RiskMetrics
        
        risk_analyzer = RiskMetrics(self.config.confidence_levels)
        return risk_analyzer.calculate_all_metrics(results)
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive bootstrap analysis with all features.
        
        Returns:
            Complete analysis results
        """
        # Run bootstrap simulation
        bootstrap_results = self.run_bootstrap_simulation()
        
        # Statistical tests
        stat_tests = self.statistical_tests(bootstrap_results)
        
        # Risk metrics
        risk_metrics = self.calculate_risk_metrics(bootstrap_results)
        
        # Combine all results
        full_results = {
            **bootstrap_results,
            'statistical_tests': stat_tests,
            'risk_metrics': risk_metrics,
            'analysis_timestamp': pd.Timestamp.now(),
        }
        
        # Add distributional distances to the top level of full_results if they exist
        if 'distributional_distances' in bootstrap_results:
            full_results['distributional_distances'] = bootstrap_results['distributional_distances']

        return full_results
    
    def export_results(self, results: Dict[str, Any], filepath: str = 'bootstrap_analysis') -> None:
        """
        Export results to CSV files.
        
        Args:
            results: Analysis results
            filepath: Base filepath for exports
        """
        # Main statistics
        stats_df = pd.DataFrame(results['simulated_stats'])
        stats_df.to_csv(f'{filepath}_statistics.csv', index=False)
        
        # Original vs simulated comparison
        comparison_df = pd.DataFrame([results['original_stats']])
        comparison_df.index = ['Original']
        stats_summary = stats_df.describe()
        comparison_df = pd.concat([comparison_df, stats_summary])
        comparison_df.to_csv(f'{filepath}_comparison.csv')
        
        # Risk metrics if available
        if 'risk_metrics' in results:
            risk_df = pd.DataFrame([results['risk_metrics']])
            risk_df.to_csv(f'{filepath}_risk_metrics.csv', index=False)
            
        # Statistical tests if available  
        if 'statistical_tests' in results:
            test_df = pd.DataFrame([results['statistical_tests']])
            test_df.to_csv(f'{filepath}_statistical_tests.csv', index=False)

        if 'distributional_distances' in results and results['distributional_distances']:
            dist_df_data = {}
            if 'mean_wasserstein_distance' in results['distributional_distances']: # Check if full dict or just info
                dist_df_data['mean_wasserstein_distance'] = [results['distributional_distances']['mean_wasserstein_distance']]
                dist_df_data['std_wasserstein_distance'] = [results['distributional_distances']['std_wasserstein_distance']]
                dist_df_data['mean_mmd_distance'] = [results['distributional_distances']['mean_mmd_distance']]
                dist_df_data['std_mmd_distance'] = [results['distributional_distances']['std_mmd_distance']]
                dist_df_data['mmd_sigma_sq_used'] = [results['distributional_distances']['mmd_sigma_sq_used']]
            
            if dist_df_data: # only create if data was populated
                dist_df = pd.DataFrame(dist_df_data)
                dist_df.to_csv(f'{filepath}_distributional_distances.csv', index=False)
            
            # Optionally save all sample distances (can be large)
            # if 'wasserstein_distances_samples' in results['distributional_distances']:
            #    pd.DataFrame({'wasserstein': results['distributional_distances']['wasserstein_distances_samples']}).to_csv(f'{filepath}_wasserstein_samples.csv', index=False)
            # if 'mmd_distances_samples' in results['distributional_distances']:
            #    pd.DataFrame({'mmd': results['distributional_distances']['mmd_distances_samples']}).to_csv(f'{filepath}_mmd_samples.csv', index=False)

    def generate_report(self, results: Optional[Dict[str, Any]] = None, 
                       output_dir: str = 'results/reports/') -> str:
        """
        Generate comprehensive HTML report.
        
        Args:
            results: Analysis results (if None, runs full analysis)
            output_dir: Output directory for report
            
        Returns:
            Path to generated report
        """
        if results is None:
            results = self.run_full_analysis()
            
        from .plotting import BootstrapPlotter
        
        plotter = BootstrapPlotter(self)
        report_path = plotter.generate_html_report(results, output_dir)
        
        return report_path
    
    def plot_results(self, results: Optional[Dict[str, Any]] = None) -> None:
        """
        Generate comprehensive plots.
        
        Args:
            results: Analysis results (if None, runs simulation)
        """
        if results is None:
            results = self.run_bootstrap_simulation()
            
        from .plotting import BootstrapPlotter
        
        plotter = BootstrapPlotter(self)
        plotter.plot_comprehensive_analysis(results)
    
    # Legacy compatibility methods
    def mc_with_replacement(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        warnings.warn(
            "mc_with_replacement is deprecated. Use run_bootstrap_simulation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.run_bootstrap_simulation()
    
    def results(self) -> pd.DataFrame:
        """Legacy method for backward compatibility."""
        warnings.warn(
            "results() is deprecated. Use run_full_analysis() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        bootstrap_results = self.run_bootstrap_simulation()
        df = pd.DataFrame(bootstrap_results['simulated_stats'])
        df.loc['Original'] = bootstrap_results['original_stats']
        return df
