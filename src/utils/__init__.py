"""Utility functions for data preprocessing, validation, and visualization."""

from .data_preprocessing import (
    standardize,
    normalize,
    remove_outliers,
    handle_missing,
    bin_data,
)
from .validation import (
    validate_probability,
    validate_positive,
    validate_nonnegative,
    validate_in_range,
    validate_array,
)
from .plotting import (
    plot_distribution_comparison,
    plot_qq,
    plot_histogram_with_fit,
    plot_correlation_heatmap,
)

from .logger import (
    setup_logger,
    get_logger,
    log_error,
    log_execution_time,
    set_correlation_id,
    get_correlation_id,
)

__all__ = [
    # Data preprocessing
    "standardize",
    "normalize",
    "remove_outliers",
    "handle_missing",
    "bin_data",
    # Validation
    "validate_probability",
    "validate_positive",
    "validate_nonnegative",
    "validate_in_range",
    "validate_array",
    # Plotting
    "plot_distribution_comparison",
    "plot_qq",
    "plot_histogram_with_fit",
    "plot_correlation_heatmap",
    # Logging
    "setup_logger",
    "get_logger",
    "log_error",
    "log_execution_time",
    "set_correlation_id",
    "get_correlation_id",
]
