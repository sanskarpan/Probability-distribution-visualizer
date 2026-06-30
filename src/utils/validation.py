"""Input validation utilities."""

from typing import Union, Tuple, Optional
import numpy as np


def validate_probability(
    p: Union[float, np.ndarray],
    name: str = "probability"
) -> Union[float, np.ndarray]:
    """
    Validate that value(s) are valid probabilities in [0, 1].

    Args:
        p: Probability value(s)
        name: Name of parameter for error messages

    Returns:
        Validated probability

    Raises:
        ValueError: If probability is not in [0, 1]
    """
    p = np.asarray(p)

    if np.any(p < 0) or np.any(p > 1):
        raise ValueError(f"{name} must be in [0, 1], got {p}")

    return p.item() if p.ndim == 0 else p


def validate_positive(
    value: Union[float, np.ndarray],
    name: str = "value"
) -> Union[float, np.ndarray]:
    """
    Validate that value(s) are strictly positive.

    Args:
        value: Value(s) to validate
        name: Name of parameter for error messages

    Returns:
        Validated value

    Raises:
        ValueError: If value is not positive
    """
    value = np.asarray(value)

    if np.any(value <= 0):
        raise ValueError(f"{name} must be positive, got {value}")

    return value.item() if value.ndim == 0 else value


def validate_nonnegative(
    value: Union[float, np.ndarray],
    name: str = "value"
) -> Union[float, np.ndarray]:
    """
    Validate that value(s) are non-negative (>= 0).

    Args:
        value: Value(s) to validate
        name: Name of parameter for error messages

    Returns:
        Validated value

    Raises:
        ValueError: If value is negative
    """
    value = np.asarray(value)

    if np.any(value < 0):
        raise ValueError(f"{name} must be non-negative, got {value}")

    return value.item() if value.ndim == 0 else value


def validate_in_range(
    value: Union[float, np.ndarray],
    lower: float,
    upper: float,
    name: str = "value",
    inclusive: str = 'both'
) -> Union[float, np.ndarray]:
    """
    Validate that value(s) are in specified range.

    Args:
        value: Value(s) to validate
        lower: Lower bound
        upper: Upper bound
        name: Name of parameter for error messages
        inclusive: 'both', 'lower', 'upper', or 'neither'

    Returns:
        Validated value

    Raises:
        ValueError: If value is not in range
    """
    value = np.asarray(value)

    if inclusive == 'both':
        condition = (value >= lower) & (value <= upper)
        range_str = f"[{lower}, {upper}]"
    elif inclusive == 'lower':
        condition = (value >= lower) & (value < upper)
        range_str = f"[{lower}, {upper})"
    elif inclusive == 'upper':
        condition = (value > lower) & (value <= upper)
        range_str = f"({lower}, {upper}]"
    elif inclusive == 'neither':
        condition = (value > lower) & (value < upper)
        range_str = f"({lower}, {upper})"
    else:
        raise ValueError(f"Unknown inclusive option: {inclusive}")

    if not np.all(condition):
        raise ValueError(f"{name} must be in range {range_str}, got {value}")

    return value.item() if value.ndim == 0 else value


def validate_array(
    array: np.ndarray,
    shape: Optional[Tuple] = None,
    ndim: Optional[int] = None,
    dtype: Optional[type] = None,
    min_length: Optional[int] = None,
    name: str = "array"
) -> np.ndarray:
    """
    Validate array properties.

    Args:
        array: Array to validate
        shape: Expected shape (None to skip check)
        ndim: Expected number of dimensions (None to skip check)
        dtype: Expected data type (None to skip check)
        min_length: Minimum length for first dimension (None to skip check)
        name: Name of parameter for error messages

    Returns:
        Validated array

    Raises:
        ValueError: If array does not meet requirements
    """
    array = np.asarray(array)

    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")

    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {array.ndim}")

    if dtype is not None and array.dtype != dtype:
        try:
            array = array.astype(dtype)
        except (ValueError, TypeError):
            raise ValueError(f"{name} must have dtype {dtype}, got {array.dtype}")

    if min_length is not None and array.shape[0] < min_length:
        raise ValueError(f"{name} must have at least {min_length} elements, got {array.shape[0]}")

    return array


def validate_integer(
    value: Union[int, float],
    name: str = "value"
) -> int:
    """
    Validate that value is an integer.

    Args:
        value: Value to validate
        name: Name of parameter for error messages

    Returns:
        Validated integer

    Raises:
        ValueError: If value is not an integer
    """
    if not isinstance(value, (int, np.integer)):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ValueError(f"{name} must be an integer, got {value} (type: {type(value)})")

    return int(value)


def validate_covariance_matrix(
    cov: np.ndarray,
    name: str = "covariance"
) -> np.ndarray:
    """
    Validate that matrix is a valid covariance matrix.

    Args:
        cov: Covariance matrix
        name: Name of parameter for error messages

    Returns:
        Validated covariance matrix

    Raises:
        ValueError: If matrix is not valid covariance
    """
    cov = np.asarray(cov)

    # Check square
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{name} must be square, got shape {cov.shape}")

    # Check symmetric
    if not np.allclose(cov, cov.T):
        raise ValueError(f"{name} must be symmetric")

    # Check positive semi-definite
    eigenvalues = np.linalg.eigvalsh(cov)
    if np.any(eigenvalues < -1e-10):  # Small tolerance for numerical errors
        raise ValueError(f"{name} must be positive semi-definite")

    return cov


def validate_correlation_matrix(
    corr: np.ndarray,
    name: str = "correlation"
) -> np.ndarray:
    """
    Validate that matrix is a valid correlation matrix.

    Args:
        corr: Correlation matrix
        name: Name of parameter for error messages

    Returns:
        Validated correlation matrix

    Raises:
        ValueError: If matrix is not valid correlation
    """
    corr = np.asarray(corr)

    # First validate as covariance matrix
    validate_covariance_matrix(corr, name)

    # Check diagonal is 1
    if not np.allclose(np.diag(corr), 1.0):
        raise ValueError(f"{name} must have 1s on diagonal")

    # Check all elements in [-1, 1]
    if np.any(corr < -1) or np.any(corr > 1):
        raise ValueError(f"{name} elements must be in [-1, 1]")

    return corr
