"""Utility functions for data preprocessing, validation, and visualization."""

from .data_preprocessing import (
    bin_data,
    box_cox_transform,
    handle_missing,
    log_transform,
    normalize,
    remove_outliers,
    standardize,
)
from .logger import (
    get_correlation_id,
    get_logger,
    log_error,
    log_execution_time,
    set_correlation_id,
    setup_logger,
)
from .plotting import (
    plot_correlation_heatmap,
    plot_distribution_comparison,
    plot_histogram_with_fit,
    plot_probability_bands,
    plot_qq,
)
from .validation import (
    validate_array,
    validate_correlation_matrix,
    validate_covariance_matrix,
    validate_in_range,
    validate_integer,
    validate_nonnegative,
    validate_positive,
    validate_probability,
)

__all__ = [
    # Data preprocessing
    "standardize",
    "normalize",
    "remove_outliers",
    "handle_missing",
    "bin_data",
    "log_transform",
    "box_cox_transform",
    # Validation
    "validate_probability",
    "validate_positive",
    "validate_nonnegative",
    "validate_in_range",
    "validate_array",
    "validate_integer",
    "validate_covariance_matrix",
    "validate_correlation_matrix",
    # Plotting
    "plot_distribution_comparison",
    "plot_qq",
    "plot_histogram_with_fit",
    "plot_correlation_heatmap",
    "plot_probability_bands",
    # Logging
    "setup_logger",
    "get_logger",
    "log_error",
    "log_execution_time",
    "set_correlation_id",
    "get_correlation_id",
]
