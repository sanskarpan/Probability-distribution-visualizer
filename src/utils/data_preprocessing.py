"""Data preprocessing utilities."""

from typing import Optional, Union, Tuple
import numpy as np
from scipy import stats


def standardize(
    data: np.ndarray,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Standardize data to zero mean and unit variance (Z-score normalization).

    Args:
        data: Input data
        return_params: Whether to return standardization parameters

    Returns:
        Standardized data, optionally with parameters (mean, std)
    """
    data = np.asarray(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    if std == 0 or np.isnan(std):
        standardized = data - mean
    else:
        standardized = (data - mean) / std

    if return_params:
        return standardized, {'mean': mean, 'std': std}
    return standardized


def normalize(
    data: np.ndarray,
    method: str = 'minmax',
    feature_range: Tuple[float, float] = (0, 1),
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Normalize data to specified range.

    Args:
        data: Input data
        method: 'minmax' or 'maxabs'
        feature_range: Target range (min, max)
        return_params: Whether to return normalization parameters

    Returns:
        Normalized data, optionally with parameters
    """
    data = np.asarray(data)

    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        range_val = max_val - min_val

        if range_val == 0:
            normalized = np.full_like(data, (feature_range[0] + feature_range[1]) / 2)
        else:
            # Scale to [0, 1]
            normalized = (data - min_val) / range_val
            # Scale to feature_range
            normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]

        params = {'min': min_val, 'max': max_val, 'feature_range': feature_range}

    elif method == 'maxabs':
        max_abs = np.max(np.abs(data))

        if max_abs == 0:
            normalized = data
        else:
            normalized = data / max_abs

        params = {'max_abs': max_abs}

    else:
        raise ValueError(f"Unknown method: {method}")

    if return_params:
        return normalized, params
    return normalized


def remove_outliers(
    data: np.ndarray,
    method: str = 'iqr',
    threshold: float = 1.5,
    return_mask: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Remove outliers from data.

    Args:
        data: Input data
        method: 'iqr', 'zscore', or 'mad'
        threshold: Threshold for outlier detection
        return_mask: Whether to return boolean mask of inliers

    Returns:
        Data with outliers removed, optionally with inlier mask
    """
    data = np.asarray(data)

    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        mask = (data >= lower_bound) & (data <= upper_bound)

    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        mask = z_scores <= threshold

    elif method == 'mad':
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            mask = np.ones(len(data), dtype=bool)
        else:
            modified_z_scores = 0.6745 * np.abs(data - median) / mad
            mask = modified_z_scores <= threshold

    else:
        raise ValueError(f"Unknown method: {method}")

    cleaned_data = data[mask]

    if return_mask:
        return cleaned_data, mask
    return cleaned_data


def handle_missing(
    data: np.ndarray,
    method: str = 'mean',
    fill_value: Optional[float] = None
) -> np.ndarray:
    """
    Handle missing values (NaN) in data.

    Args:
        data: Input data with potential NaN values
        method: 'mean', 'median', 'mode', 'forward_fill', 'backward_fill', or 'constant'
        fill_value: Value to use for 'constant' method

    Returns:
        Data with missing values filled
    """
    data = np.asarray(data).copy()
    mask = np.isnan(data)

    if not np.any(mask):
        return data  # No missing values

    if method == 'mean':
        fill = np.nanmean(data)
    elif method == 'median':
        fill = np.nanmedian(data)
    elif method == 'mode':
        # For continuous data, use median as approximation
        fill = np.nanmedian(data)
    elif method == 'constant':
        if fill_value is None:
            raise ValueError("fill_value must be provided for 'constant' method")
        fill = fill_value
    elif method == 'forward_fill':
        # Forward fill
        for i in range(len(data)):
            if np.isnan(data[i]) and i > 0:
                data[i] = data[i-1]
        # Fill any remaining NaNs at the start
        first_valid = np.where(~np.isnan(data))[0]
        if len(first_valid) > 0:
            data[:first_valid[0]] = data[first_valid[0]]
        return data
    elif method == 'backward_fill':
        # Backward fill
        for i in range(len(data)-1, -1, -1):
            if np.isnan(data[i]) and i < len(data)-1:
                data[i] = data[i+1]
        # Fill any remaining NaNs at the end
        last_valid = np.where(~np.isnan(data))[0]
        if len(last_valid) > 0:
            data[last_valid[-1]+1:] = data[last_valid[-1]]
        return data
    else:
        raise ValueError(f"Unknown method: {method}")

    data[mask] = fill
    return data


def bin_data(
    data: np.ndarray,
    n_bins: Optional[int] = None,
    bins: Optional[np.ndarray] = None,
    method: str = 'equal_width',
    return_bins: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Bin continuous data into discrete bins.

    Args:
        data: Input data
        n_bins: Number of bins (if bins not provided)
        bins: Explicit bin edges
        method: 'equal_width', 'equal_frequency', or 'custom'
        return_bins: Whether to return bin edges

    Returns:
        Bin indices for each data point, optionally with bin edges
    """
    data = np.asarray(data)

    if n_bins is None and bins is None:
        n_bins = int(np.sqrt(len(data)))  # Sturges' rule approximation

    if bins is None:
        assert n_bins is not None
        if method == 'equal_width':
            bins = np.linspace(np.min(data), np.max(data), n_bins + 1)
        elif method == 'equal_frequency':
            bins = np.percentile(data, np.linspace(0, 100, n_bins + 1))
        else:
            raise ValueError(f"Unknown method: {method}")

    # Assign data to bins
    bin_indices = np.digitize(data, bins) - 1

    # Handle edge cases
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    if return_bins:
        return bin_indices, bins
    return bin_indices


def log_transform(
    data: np.ndarray,
    shift: float = 0.0,
    base: str = 'e'
) -> np.ndarray:
    """
    Apply logarithmic transformation to data.

    Args:
        data: Input data
        shift: Value to add before taking log (for handling zeros/negatives)
        base: 'e', '10', or '2'

    Returns:
        Log-transformed data
    """
    data = np.asarray(data) + shift

    if np.any(data <= 0):
        raise ValueError("Cannot take log of non-positive values. Use shift parameter.")

    if base == 'e':
        return np.log(data)
    elif base == '10':
        return np.log10(data)
    elif base == '2':
        return np.log2(data)
    else:
        raise ValueError(f"Unknown base: {base}")


def box_cox_transform(
    data: np.ndarray,
    lambda_param: Optional[float] = None
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Apply Box-Cox power transformation.

    Args:
        data: Input data (must be positive)
        lambda_param: Transformation parameter (estimated if None)

    Returns:
        Transformed data and lambda parameter (if estimated)
    """
    data = np.asarray(data)

    if np.any(data <= 0):
        raise ValueError("Box-Cox requires positive data")

    if lambda_param is None:
        # Estimate optimal lambda
        transformed, lambda_param = stats.boxcox(data)
        return transformed, lambda_param
    else:
        # Use provided lambda
        if lambda_param == 0:
            return np.log(data)
        else:
            return (data**lambda_param - 1) / lambda_param
