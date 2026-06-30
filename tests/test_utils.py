"""Tests for utility modules: validation and data_preprocessing."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.validation import (
    validate_probability,
    validate_positive,
    validate_nonnegative,
    validate_in_range,
    validate_array,
    validate_integer,
    validate_covariance_matrix,
    validate_correlation_matrix,
)
from utils.data_preprocessing import (
    standardize,
    normalize,
    remove_outliers,
    handle_missing,
    bin_data,
    log_transform,
    box_cox_transform,
)


# ---------------------------------------------------------------------------
# validation.py
# ---------------------------------------------------------------------------


class TestValidateProbability:
    def test_valid_scalar(self):
        result = validate_probability(0.5)
        assert result == 0.5

    def test_valid_zero(self):
        result = validate_probability(0.0)
        assert result == 0.0

    def test_valid_one(self):
        result = validate_probability(1.0)
        assert result == 1.0

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_probability(-0.1)

    def test_invalid_above_one(self):
        with pytest.raises(ValueError, match="must be in"):
            validate_probability(1.5)

    def test_valid_array(self):
        result = validate_probability(np.array([0.1, 0.5, 0.9]))
        assert np.allclose(result, [0.1, 0.5, 0.9])

    def test_invalid_array(self):
        with pytest.raises(ValueError):
            validate_probability(np.array([0.1, 1.2]))

    def test_custom_name(self):
        with pytest.raises(ValueError, match="alpha"):
            validate_probability(1.5, name="alpha")


class TestValidatePositive:
    def test_valid_scalar(self):
        result = validate_positive(3.0)
        assert result == 3.0

    def test_valid_tiny(self):
        result = validate_positive(1e-10)
        assert result == 1e-10

    def test_invalid_zero(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(0.0)

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(-5.0)

    def test_valid_array(self):
        result = validate_positive(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_invalid_array(self):
        with pytest.raises(ValueError):
            validate_positive(np.array([1.0, 0.0, 3.0]))

    def test_custom_name(self):
        with pytest.raises(ValueError, match="sigma"):
            validate_positive(0.0, name="sigma")


class TestValidateNonnegative:
    def test_valid_positive(self):
        result = validate_nonnegative(5.0)
        assert result == 5.0

    def test_valid_zero(self):
        result = validate_nonnegative(0.0)
        assert result == 0.0

    def test_invalid_negative(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_nonnegative(-1.0)

    def test_valid_array(self):
        result = validate_nonnegative(np.array([0.0, 2.0, 5.0]))
        assert np.allclose(result, [0.0, 2.0, 5.0])

    def test_invalid_array(self):
        with pytest.raises(ValueError):
            validate_nonnegative(np.array([0.0, -1.0]))


class TestValidateInRange:
    def test_inclusive_both_valid(self):
        result = validate_in_range(5.0, 0, 10)
        assert result == 5.0

    def test_inclusive_both_lower_bound_valid(self):
        result = validate_in_range(0.0, 0, 10)
        assert result == 0.0

    def test_inclusive_both_upper_bound_valid(self):
        result = validate_in_range(10.0, 0, 10)
        assert result == 10.0

    def test_inclusive_both_invalid(self):
        with pytest.raises(ValueError, match="must be in range"):
            validate_in_range(15.0, 0, 10)

    def test_inclusive_lower_valid(self):
        result = validate_in_range(5.0, 0, 10, inclusive='lower')
        assert result == 5.0

    def test_inclusive_lower_upper_excluded(self):
        with pytest.raises(ValueError):
            validate_in_range(10.0, 0, 10, inclusive='lower')

    def test_inclusive_upper_valid(self):
        result = validate_in_range(5.0, 0, 10, inclusive='upper')
        assert result == 5.0

    def test_inclusive_upper_lower_excluded(self):
        with pytest.raises(ValueError):
            validate_in_range(0.0, 0, 10, inclusive='upper')

    def test_inclusive_neither_valid(self):
        result = validate_in_range(5.0, 0, 10, inclusive='neither')
        assert result == 5.0

    def test_inclusive_neither_bounds_excluded(self):
        with pytest.raises(ValueError):
            validate_in_range(0.0, 0, 10, inclusive='neither')
        with pytest.raises(ValueError):
            validate_in_range(10.0, 0, 10, inclusive='neither')

    def test_unknown_inclusive_option(self):
        with pytest.raises(ValueError, match="Unknown inclusive"):
            validate_in_range(5.0, 0, 10, inclusive='bad')

    def test_array_input(self):
        result = validate_in_range(np.array([1, 2, 3]), 0, 5)
        assert np.allclose(result, [1, 2, 3])

    def test_array_input_invalid(self):
        with pytest.raises(ValueError):
            validate_in_range(np.array([1, 10]), 0, 5)

    def test_custom_name(self):
        with pytest.raises(ValueError, match="x"):
            validate_in_range(100, 0, 10, name="x")


class TestValidateArray:
    def test_valid_no_constraints(self):
        arr = np.array([1, 2, 3])
        result = validate_array(arr)
        assert np.allclose(result, arr)

    def test_valid_shape(self):
        arr = np.array([[1, 2], [3, 4]])
        result = validate_array(arr, shape=(2, 2))
        assert result.shape == (2, 2)

    def test_invalid_shape(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must have shape"):
            validate_array(arr, shape=(2, 3))

    def test_valid_ndim(self):
        arr = np.array([[1, 2], [3, 4]])
        result = validate_array(arr, ndim=2)
        assert result.ndim == 2

    def test_invalid_ndim(self):
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must have.*dimensions"):
            validate_array(arr, ndim=2)

    def test_dtype_cast(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = validate_array(arr, dtype=np.float64)
        assert result.dtype == np.float64

    def test_dtype_cast_failure(self):
        arr = np.array(['a', 'b'])
        with pytest.raises(ValueError, match="must have dtype"):
            validate_array(arr, dtype=np.float64)

    def test_min_length_valid(self):
        arr = np.array([1, 2, 3])
        result = validate_array(arr, min_length=2)
        assert len(result) == 3

    def test_min_length_invalid(self):
        arr = np.array([1, 2])
        with pytest.raises(ValueError, match="must have at least"):
            validate_array(arr, min_length=3)

    def test_list_input(self):
        result = validate_array([1, 2, 3], shape=(3,))
        assert result.shape == (3,)


class TestValidateInteger:
    def test_valid_int(self):
        assert validate_integer(5) == 5

    def test_valid_numpy_int(self):
        assert validate_integer(np.int32(10)) == 10

    def test_valid_float_whole(self):
        assert validate_integer(5.0) == 5

    def test_invalid_float(self):
        with pytest.raises(ValueError, match="must be an integer"):
            validate_integer(3.5)

    def test_invalid_string(self):
        with pytest.raises(ValueError, match="must be an integer"):
            validate_integer("5")

    def test_custom_name(self):
        with pytest.raises(ValueError, match="n"):
            validate_integer(2.5, name="n")


class TestValidateCovarianceMatrix:
    def test_valid_cov_matrix(self):
        cov = np.array([[2.0, 0.5], [0.5, 1.0]])
        result = validate_covariance_matrix(cov)
        assert np.allclose(result, cov)

    def test_diagonal_matrix(self):
        cov = np.array([[1.0, 0.0], [0.0, 2.0]])
        result = validate_covariance_matrix(cov)
        assert np.allclose(result, cov)

    def test_not_square(self):
        cov = np.array([[1.0, 0.5]])
        with pytest.raises(ValueError, match="must be square"):
            validate_covariance_matrix(cov)

    def test_not_symmetric(self):
        cov = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="must be symmetric"):
            validate_covariance_matrix(cov)

    def test_not_positive_semidefinite(self):
        cov = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueError, match="positive semi-definite"):
            validate_covariance_matrix(cov)

    def test_1d_input(self):
        cov = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="must be square"):
            validate_covariance_matrix(cov)

    def test_larger_valid_matrix(self):
        cov = np.array([[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 2]])
        result = validate_covariance_matrix(cov)
        assert result.shape == (3, 3)


class TestValidateCorrelationMatrix:
    def test_valid_corr_matrix(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = validate_correlation_matrix(corr)
        assert np.allclose(result, corr)

    def test_diagonal_not_one(self):
        corr = np.array([[0.5, 0.5], [0.5, 0.5]])
        with pytest.raises(ValueError, match="1s on diagonal"):
            validate_correlation_matrix(corr)

    def test_values_out_of_bounds(self):
        corr = np.array([[1.0, 1.5], [1.5, 1.0]])
        with pytest.raises(ValueError, match="positive semi-definite"):
            validate_correlation_matrix(corr)

    def test_not_symmetric(self):
        corr = np.array([[1.0, 0.5], [0.3, 1.0]])
        with pytest.raises(ValueError, match="must be symmetric"):
            validate_correlation_matrix(corr)

    def test_not_psd(self):
        corr = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueError):
            validate_correlation_matrix(corr)

    def test_identity_matrix(self):
        corr = np.eye(4)
        result = validate_correlation_matrix(corr)
        assert result.shape == (4, 4)

    def test_valid_negative_corr(self):
        corr = np.array([[1.0, -0.8], [-0.8, 1.0]])
        result = validate_correlation_matrix(corr)
        assert np.allclose(result, corr)


# ---------------------------------------------------------------------------
# data_preprocessing.py
# ---------------------------------------------------------------------------


class TestStandardize:
    def test_standardize_basic(self):
        np.random.seed(42)
        data = np.random.normal(10, 5, 1000)
        result = standardize(data)

        assert np.isclose(np.mean(result), 0.0, atol=1e-10)
        assert np.isclose(np.std(result, ddof=1), 1.0, atol=1e-10)

    def test_standardize_return_params(self):
        np.random.seed(42)
        data = np.random.normal(10, 5, 100)
        result, params = standardize(data, return_params=True)

        assert 'mean' in params
        assert 'std' in params
        assert np.isclose(params['mean'], np.mean(data))
        assert np.isclose(params['std'], np.std(data, ddof=1))

    def test_standardize_constant_data(self):
        data = np.array([5.0, 5.0, 5.0])
        result = standardize(data)

        assert np.allclose(result, [0.0, 0.0, 0.0])

    def test_standardize_single_value(self):
        import warnings
        data = np.array([7.0, 7.0])
        result = standardize(data)
        assert np.allclose(result, [0.0, 0.0])

    def test_standardize_2d_array(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        result = standardize(data)
        assert result.shape == data.shape


class TestNormalize:
    def test_minmax_basic(self):
        data = np.array([0, 5, 10])
        result = normalize(data, method='minmax')

        assert np.isclose(np.min(result), 0.0)
        assert np.isclose(np.max(result), 1.0)

    def test_minmax_custom_range(self):
        data = np.array([0, 5, 10])
        result = normalize(data, method='minmax', feature_range=(-1, 1))

        assert abs(np.min(result) - (-1)) < 1e-10
        assert abs(np.max(result) - 1) < 1e-10

    def test_minmax_return_params(self):
        data = np.array([0, 5, 10])
        result, params = normalize(data, method='minmax', return_params=True)

        assert 'min' in params
        assert 'max' in params
        assert 'feature_range' in params

    def test_minmax_constant_data(self):
        data = np.array([3.0, 3.0, 3.0])
        result = normalize(data, method='minmax', feature_range=(0, 1))

        assert np.allclose(result, [0.5, 0.5, 0.5])

    def test_maxabs_basic(self):
        data = np.array([-4, 1, 2])
        result = normalize(data, method='maxabs')

        assert np.max(np.abs(result)) == 1.0

    def test_maxabs_return_params(self):
        data = np.array([-4, 1, 2])
        result, params = normalize(data, method='maxabs', return_params=True)

        assert 'max_abs' in params
        assert params['max_abs'] == 4

    def test_maxabs_all_zeros(self):
        data = np.array([0.0, 0.0, 0.0])
        result = normalize(data, method='maxabs')

        assert np.allclose(result, [0.0, 0.0, 0.0])

    def test_unknown_method(self):
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown method"):
            normalize(data, method='unknown')


class TestRemoveOutliers:
    def test_iqr_no_outliers(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        cleaned = remove_outliers(data, method='iqr', threshold=1.5)

        assert len(cleaned) >= len(data) * 0.95

    def test_iqr_with_outliers(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 200])
        cleaned = remove_outliers(data, method='iqr', threshold=1.5)

        assert len(cleaned) < len(data)

    def test_iqr_return_mask(self):
        data = np.array([1, 2, 3, 100])
        cleaned, mask = remove_outliers(data, method='iqr', return_mask=True)

        assert len(cleaned) < len(data)
        assert len(mask) == len(data)
        assert mask.dtype == bool

    def test_zscore_no_outliers(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        cleaned = remove_outliers(data, method='zscore', threshold=3.0)

        assert len(cleaned) >= len(data) * 0.95

    def test_zscore_with_outliers(self):
        data = np.array([1.0, 1.1, 1.0, 0.9, 1.0, 100.0, 200.0])
        cleaned = remove_outliers(data, method='zscore', threshold=1.0)

        assert len(cleaned) < len(data)

    def test_mad_no_outliers(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        cleaned = remove_outliers(data, method='mad', threshold=3.0)

        assert len(cleaned) >= len(data) * 0.90

    def test_mad_constant_data(self):
        data = np.array([5.0, 5.0, 5.0, 5.0])
        cleaned = remove_outliers(data, method='mad', threshold=2.5)

        assert len(cleaned) == len(data)

    def test_unknown_method(self):
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown method"):
            remove_outliers(data, method='unknown')


class TestHandleMissing:
    def test_no_missing(self):
        data = np.array([1.0, 2.0, 3.0])
        result = handle_missing(data, method='mean')

        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_mean_fill(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = handle_missing(data, method='mean')

        assert not np.any(np.isnan(result))
        assert np.isclose(result[1], 3.0)
        assert np.isclose(result[3], 3.0)

    def test_median_fill(self):
        data = np.array([1.0, np.nan, 100.0, np.nan, 2.0])
        result = handle_missing(data, method='median')

        assert not np.any(np.isnan(result))

    def test_mode_fill(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = handle_missing(data, method='mode')

        assert not np.any(np.isnan(result))

    def test_constant_fill(self):
        data = np.array([1.0, np.nan, 3.0])
        result = handle_missing(data, method='constant', fill_value=-999)

        assert result[1] == -999
        assert not np.any(np.isnan(result))

    def test_constant_no_fill_value(self):
        data = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="fill_value must be provided"):
            handle_missing(data, method='constant')

    def test_forward_fill(self):
        data = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        result = handle_missing(data, method='forward_fill')

        assert np.allclose(result, [1.0, 1.0, 1.0, 4.0, 4.0])

    def test_forward_fill_all_nan_start(self):
        data = np.array([np.nan, np.nan, 3.0, np.nan, 5.0])
        result = handle_missing(data, method='forward_fill')

        assert np.allclose(result, [3.0, 3.0, 3.0, 3.0, 5.0])

    def test_backward_fill(self):
        data = np.array([1.0, np.nan, np.nan, 4.0, np.nan])
        result = handle_missing(data, method='backward_fill')

        expected = np.array([1.0, 4.0, 4.0, 4.0, 4.0])
        assert np.allclose(result, expected)

    def test_backward_fill_all_nan_end(self):
        data = np.array([1.0, np.nan, 4.0, np.nan, np.nan])
        result = handle_missing(data, method='backward_fill')

        assert np.allclose(result, [1.0, 4.0, 4.0, 4.0, 4.0])

    def test_unknown_method(self):
        data = np.array([1.0, np.nan])
        with pytest.raises(ValueError, match="Unknown method"):
            handle_missing(data, method='unknown')

    def test_all_nan_forward(self):
        data = np.array([np.nan, np.nan, np.nan])
        result = handle_missing(data, method='forward_fill')
        assert np.all(np.isnan(result))


class TestBinData:
    def test_equal_width_basic(self):
        np.random.seed(42)
        data = np.random.uniform(0, 10, 100)
        binned = bin_data(data, n_bins=5, method='equal_width')

        assert len(binned) == 100
        assert np.min(binned) >= 0
        assert np.max(binned) <= 4

    def test_equal_frequency(self):
        np.random.seed(42)
        data = np.random.uniform(0, 10, 500)
        binned = bin_data(data, n_bins=10, method='equal_frequency')

        assert len(binned) == 500
        counts = np.bincount(binned, minlength=10)
        assert all(c > 30 for c in counts)

    def test_return_bins(self):
        data = np.array([1, 2, 3, 4, 5])
        binned, bins_edges = bin_data(data, n_bins=2, method='equal_width', return_bins=True)

        assert len(bins_edges) == 3
        assert len(binned) == 5

    def test_custom_bins(self):
        data = np.array([1, 5, 10, 15, 20])
        custom_edges = np.array([0, 10, 20])
        binned = bin_data(data, bins=custom_edges)

        assert np.min(binned) >= 0
        assert np.max(binned) <= 1

    def test_default_bins(self):
        np.random.seed(42)
        data = np.random.uniform(0, 10, 100)
        binned = bin_data(data)

        assert len(binned) == 100

    def test_unknown_method(self):
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown method"):
            bin_data(data, n_bins=2, method='unknown')

    def test_small_data(self):
        data = np.array([1, 2])
        binned = bin_data(data, n_bins=2)
        assert len(binned) == 2


class TestLogTransform:
    def test_natural_log(self):
        data = np.array([1, 10, 100])
        result = log_transform(data, base='e')

        assert np.allclose(result, np.log(data))

    def test_log10(self):
        data = np.array([1, 10, 100])
        result = log_transform(data, base='10')

        assert np.allclose(result, np.log10(data))

    def test_log2(self):
        data = np.array([1, 2, 4, 8])
        result = log_transform(data, base='2')

        assert np.allclose(result, np.log2(data))

    def test_with_shift(self):
        data = np.array([0, 1, 2])
        result = log_transform(data, shift=1, base='e')

        assert np.allclose(result, np.log(data + 1))

    def test_non_positive_error(self):
        data = np.array([-1, 0, 1])
        with pytest.raises(ValueError, match="Cannot take log"):
            log_transform(data)

    def test_unknown_base(self):
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Unknown base"):
            log_transform(data, base='5')

    def test_shift_handles_zeros(self):
        data = np.array([0, 1, 2])
        result = log_transform(data, shift=0.5)
        assert not np.any(np.isinf(result))


class TestBoxCoxTransform:
    def test_estimate_lambda(self):
        np.random.seed(42)
        data = np.random.gamma(2, 2, 200)
        data = data[data > 0]
        result, lam = box_cox_transform(data)

        assert isinstance(result, np.ndarray)
        assert isinstance(lam, float)
        assert len(result) == len(data)

    def test_fixed_lambda_zero(self):
        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, 200)
        result = box_cox_transform(data, lambda_param=0)

        assert np.allclose(result, np.log(data))

    def test_fixed_lambda_nonzero(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = box_cox_transform(data, lambda_param=0.5)

        expected = (data**0.5 - 1) / 0.5
        assert np.allclose(result, expected)

    def test_non_positive_error(self):
        data = np.array([-1, 2, 3])
        with pytest.raises(ValueError, match="requires positive"):
            box_cox_transform(data)

    def test_with_zero_values(self):
        data = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="requires positive"):
            box_cox_transform(data)

    def test_fixed_lambda_negative_transform(self):
        data = np.array([1.0, 2.0, 3.0])
        result = box_cox_transform(data, lambda_param=-1)

        expected = (data ** -1 - 1) / -1
        assert np.allclose(result, expected)