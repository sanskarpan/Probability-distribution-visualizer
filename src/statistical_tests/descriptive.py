"""Descriptive statistics utilities."""

from typing import Dict, Optional, List, Tuple
import numpy as np
from scipy import stats


def describe(data: np.ndarray, percentiles: Optional[List[float]] = None) -> Dict:
    """
    Comprehensive descriptive statistics.

    Args:
        data: Input data
        percentiles: List of percentiles to compute (default: [25, 50, 75])

    Returns:
        Dictionary with descriptive statistics
    """
    if percentiles is None:
        percentiles = [25, 50, 75]

    data = np.asarray(data).flatten()

    # Basic statistics
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    var = np.var(data, ddof=1)

    # Median and MAD
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    # Range
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val

    # Quartiles and IQR
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1

    # Shape measures
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    # Standard error
    se = std / np.sqrt(n)

    # Coefficient of variation
    cv = (std / mean) * 100 if mean != 0 else np.inf

    # Percentiles
    percentile_values = {
        f'p{int(p)}': np.percentile(data, p)
        for p in percentiles
    }

    return {
        'count': n,
        'mean': float(mean),
        'std': float(std),
        'var': float(var),
        'se': float(se),
        'cv': float(cv),
        'min': float(min_val),
        'max': float(max_val),
        'range': float(range_val),
        'median': float(median),
        'mad': float(mad),
        'q1': float(q1),
        'q3': float(q3),
        'iqr': float(iqr),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        **{k: float(v) for k, v in percentile_values.items()}
    }


def quantile_summary(
    data: np.ndarray,
    n_quantiles: int = 4
) -> Dict[str, np.ndarray]:
    """
    Compute quantile summary.

    Args:
        data: Input data
        n_quantiles: Number of quantiles (4 for quartiles, 10 for deciles, etc.)

    Returns:
        Dictionary with quantile information
    """
    data = np.asarray(data).flatten()

    # Compute quantile boundaries
    quantiles = np.linspace(0, 100, n_quantiles + 1)
    boundaries = np.percentile(data, quantiles)

    # Assign data points to quantiles
    quantile_labels = np.searchsorted(boundaries[1:-1], data)

    # Count per quantile
    counts = np.bincount(quantile_labels, minlength=n_quantiles)

    return {
        'n_quantiles': n_quantiles,
        'boundaries': boundaries,
        'labels': quantile_labels,
        'counts': counts,
        'proportions': counts / len(data)
    }


def outlier_detection(
    data: np.ndarray,
    method: str = 'iqr',
    threshold: float = 1.5
) -> Dict[str, np.ndarray]:
    """
    Detect outliers using various methods.

    Args:
        data: Input data
        method: 'iqr', 'zscore', or 'mad'
        threshold: Threshold for outlier detection
                  - IQR: typically 1.5 or 3.0
                  - Z-score: typically 2.5 or 3.0
                  - MAD: typically 2.5 or 3.0

    Returns:
        Dictionary with outlier information
    """
    data = np.asarray(data).flatten()

    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outliers = (data < lower_bound) | (data > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
        lower_bound = np.mean(data) - threshold * np.std(data)
        upper_bound = np.mean(data) + threshold * np.std(data)

    elif method == 'mad':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
        outliers = np.abs(modified_z_scores) > threshold
        lower_bound = median - threshold * mad / 0.6745
        upper_bound = median + threshold * mad / 0.6745

    else:
        raise ValueError(f"Unknown method: {method}")

    outlier_indices = np.where(outliers)[0]
    outlier_values = data[outliers]

    return {
        'method': method,
        'threshold': threshold,
        'n_outliers': int(np.sum(outliers)),
        'outlier_proportion': float(np.mean(outliers)),
        'outlier_indices': outlier_indices,
        'outlier_values': outlier_values,
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'is_outlier': outliers
    }


def correlation_matrix(
    data: np.ndarray,
    method: str = 'pearson',
    return_pvalues: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute correlation matrix.

    Args:
        data: 2D array where each column is a variable
        method: 'pearson', 'spearman', or 'kendall'
        return_pvalues: Whether to return p-values

    Returns:
        Dictionary with correlation matrix and optionally p-values
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_vars = data.shape[1]
    corr_matrix = np.zeros((n_vars, n_vars))
    p_matrix = np.zeros((n_vars, n_vars)) if return_pvalues else None

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                corr_matrix[i, j] = 1.0
                if p_matrix is not None:
                    p_matrix[i, j] = 0.0
            else:
                if method == 'pearson':
                    corr, pval = stats.pearsonr(data[:, i], data[:, j])
                elif method == 'spearman':
                    corr, pval = stats.spearmanr(data[:, i], data[:, j])
                elif method == 'kendall':
                    corr, pval = stats.kendalltau(data[:, i], data[:, j])
                else:
                    raise ValueError(f"Unknown method: {method}")

                corr_matrix[i, j] = corr
                if p_matrix is not None:
                    p_matrix[i, j] = pval

    result: Dict[str, np.ndarray] = {
        'correlation_matrix': corr_matrix,
        'method': method  # type: ignore[dict-item]
    }

    if p_matrix is not None:
        result['p_values'] = p_matrix

    return result
