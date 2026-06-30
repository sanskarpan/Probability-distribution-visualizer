"""Tests for plotting utilities."""

import matplotlib
matplotlib.use('Agg')

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.plotting import (
    plot_distribution_comparison,
    plot_qq,
    plot_histogram_with_fit,
    plot_correlation_heatmap,
    plot_probability_bands,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


# ---------------------------------------------------------------------------
# plot_distribution_comparison
# ---------------------------------------------------------------------------


class TestPlotDistributionComparison:
    def test_returns_figure(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_distribution_comparison(data, ['norm'])
        assert isinstance(fig, plt.Figure)

    def test_two_subplots(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_distribution_comparison(data, ['norm'])
        axes = fig.get_axes()
        assert len(axes) == 2

    def test_with_multiple_distributions(self):
        np.random.seed(42)
        data = np.random.normal(5, 2, 300)
        fig = plot_distribution_comparison(data, ['norm', 'expon', 'gamma'])
        assert isinstance(fig, plt.Figure)

    def test_with_fitted_params(self):
        np.random.seed(42)
        data = np.random.normal(5, 2, 200)
        fitted_params = {
            'norm': (5.0, 2.0),
            'expon': (0, 5.0),
        }
        fig = plot_distribution_comparison(data, ['norm', 'expon'], fitted_params=fitted_params)
        assert isinstance(fig, plt.Figure)

    def test_with_partial_fitted_params(self):
        np.random.seed(42)
        data = np.random.normal(5, 2, 200)
        fitted_params = {'norm': (5.0, 2.0)}
        fig = plot_distribution_comparison(data, ['norm', 'expon'], fitted_params=fitted_params)
        assert isinstance(fig, plt.Figure)

    def test_custom_bins_and_figsize(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_distribution_comparison(data, ['norm'], bins=50, figsize=(8, 4))
        assert isinstance(fig, plt.Figure)

    def test_with_exponential_data(self):
        np.random.seed(42)
        data = np.random.exponential(2, 200)
        fig = plot_distribution_comparison(data, ['expon', 'norm'])
        assert isinstance(fig, plt.Figure)

    def test_with_uniform_data(self):
        np.random.seed(42)
        data = np.random.uniform(-3, 3, 200)
        fig = plot_distribution_comparison(data, ['uniform', 'norm'])
        assert isinstance(fig, plt.Figure)

    def test_first_axis_has_histogram(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_distribution_comparison(data, ['norm'])
        ax1 = fig.get_axes()[0]
        assert len(ax1.patches) > 0

    def test_second_axis_title_is_cdf(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_distribution_comparison(data, ['norm'])
        ax2 = fig.get_axes()[1]
        assert 'CDF' in ax2.get_title()

    def test_single_distribution(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_distribution_comparison(data, ['norm'])
        assert isinstance(fig, plt.Figure)

    def test_small_data(self):
        np.random.seed(42)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fig = plot_distribution_comparison(data, ['norm', 'uniform'])
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_qq
# ---------------------------------------------------------------------------


class TestPlotQQ:
    def test_returns_figure(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_qq(data)
        assert isinstance(fig, plt.Figure)

    def test_single_axes(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_qq(data)
        assert len(fig.get_axes()) == 1

    def test_default_dist_is_norm(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_qq(data)
        ax = fig.get_axes()[0]
        assert 'norm' in ax.get_title().lower()

    def test_with_explicit_params(self):
        np.random.seed(42)
        data = np.random.normal(5, 2, 200)
        fig = plot_qq(data, dist='norm', params=(5, 2))
        assert isinstance(fig, plt.Figure)

    def test_without_params_fits_automatically(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_qq(data)
        assert isinstance(fig, plt.Figure)

    def test_with_exponential_distribution(self):
        np.random.seed(42)
        data = np.random.exponential(2, 200)
        fig = plot_qq(data, dist='expon')
        assert isinstance(fig, plt.Figure)

    def test_with_uniform_distribution(self):
        np.random.seed(42)
        data = np.random.uniform(0, 10, 200)
        fig = plot_qq(data, dist='uniform')
        assert isinstance(fig, plt.Figure)

    def test_custom_figsize(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_qq(data, figsize=(6, 6))
        assert isinstance(fig, plt.Figure)

    def test_has_reference_line(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_qq(data)
        ax = fig.get_axes()[0]
        assert len(ax.lines) > 0

    def test_with_gamma_distribution_and_params(self):
        np.random.seed(42)
        data = np.random.gamma(2, scale=3, size=200)
        fig = plot_qq(data, dist='gamma', params=(2, 0, 3))
        assert isinstance(fig, plt.Figure)

    def test_small_data(self):
        np.random.seed(42)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fig = plot_qq(data)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_histogram_with_fit
# ---------------------------------------------------------------------------


class TestPlotHistogramWithFit:
    def test_returns_figure(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_histogram_with_fit(data)
        assert isinstance(fig, plt.Figure)

    def test_single_axes(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_histogram_with_fit(data)
        assert len(fig.get_axes()) == 1

    def test_default_dist_is_norm(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_histogram_with_fit(data)
        ax = fig.get_axes()[0]
        assert 'norm' in ax.get_title().lower()

    def test_with_explicit_params(self):
        np.random.seed(42)
        data = np.random.normal(5, 2, 200)
        fig = plot_histogram_with_fit(data, dist='norm', params=(5, 2))
        assert isinstance(fig, plt.Figure)

    def test_without_params_fits_automatically(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_histogram_with_fit(data)
        assert isinstance(fig, plt.Figure)

    def test_with_exponential_distribution(self):
        np.random.seed(42)
        data = np.random.exponential(2, 200)
        fig = plot_histogram_with_fit(data, dist='expon')
        assert isinstance(fig, plt.Figure)

    def test_histogram_has_bars(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_histogram_with_fit(data)
        ax = fig.get_axes()[0]
        assert len(ax.patches) > 0

    def test_has_fitted_line(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_histogram_with_fit(data)
        ax = fig.get_axes()[0]
        assert len(ax.lines) > 0

    def test_custom_bins_and_figsize(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 200)
        fig = plot_histogram_with_fit(data, bins=20, figsize=(8, 5))
        assert isinstance(fig, plt.Figure)

    def test_with_gamma_distribution_and_params(self):
        np.random.seed(42)
        data = np.random.gamma(2, scale=3, size=200)
        fig = plot_histogram_with_fit(data, dist='gamma', params=(2, 0, 3))
        assert isinstance(fig, plt.Figure)

    def test_small_data(self):
        np.random.seed(42)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fig = plot_histogram_with_fit(data)
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# plot_correlation_heatmap
# ---------------------------------------------------------------------------


class TestPlotCorrelationHeatmap:
    def test_returns_figure_pearson(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 4))
        fig = plot_correlation_heatmap(data, method='pearson')
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_spearman(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 4))
        fig = plot_correlation_heatmap(data, method='spearman')
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_kendall(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 4))
        fig = plot_correlation_heatmap(data, method='kendall')
        assert isinstance(fig, plt.Figure)

    def test_has_axes(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        fig = plot_correlation_heatmap(data)
        assert len(fig.get_axes()) >= 1

    def test_title_contains_pearson(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        fig = plot_correlation_heatmap(data, method='pearson')
        ax = fig.get_axes()[0]
        assert 'pearson' in ax.get_title().lower()

    def test_title_contains_spearman(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        fig = plot_correlation_heatmap(data, method='spearman')
        ax = fig.get_axes()[0]
        assert 'spearman' in ax.get_title().lower()

    def test_title_contains_kendall(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        fig = plot_correlation_heatmap(data, method='kendall')
        ax = fig.get_axes()[0]
        assert 'kendall' in ax.get_title().lower()

    def test_with_custom_labels(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        labels = ['Height', 'Weight', 'Age']
        fig = plot_correlation_heatmap(data, labels=labels)
        ax = fig.get_axes()[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == labels

    def test_default_labels(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        fig = plot_correlation_heatmap(data)
        ax = fig.get_axes()[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert tick_labels == ['Var 1', 'Var 2', 'Var 3']

    def test_with_annotations_enabled(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        fig = plot_correlation_heatmap(data, annot=True)
        ax = fig.get_axes()[0]
        assert len(ax.texts) > 0

    def test_with_annotations_disabled(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        fig = plot_correlation_heatmap(data, annot=False)
        ax = fig.get_axes()[0]
        assert len(ax.texts) == 0

    def test_custom_figsize_and_cmap(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        fig = plot_correlation_heatmap(data, figsize=(6, 5), cmap='RdBu')
        assert isinstance(fig, plt.Figure)

    def test_highly_correlated_data(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = 0.9 * x + np.random.normal(0, 0.1, 200)
        data = np.column_stack([x, y])
        fig = plot_correlation_heatmap(data, annot=False)
        ax = fig.get_axes()[0]
        image_data = ax.get_images()[0].get_array()
        assert image_data[0, 1] > 0.8

    def test_negative_correlation(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = -0.8 * x + np.random.normal(0, 0.2, 200)
        data = np.column_stack([x, y])
        fig = plot_correlation_heatmap(data, annot=False)
        ax = fig.get_axes()[0]
        image_data = ax.get_images()[0].get_array()
        assert image_data[0, 1] < -0.6

    def test_spearman_vs_pearson_different(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = x**3
        data = np.column_stack([x, y])
        fig_p = plot_correlation_heatmap(data, method='pearson', annot=False)
        fig_s = plot_correlation_heatmap(data, method='spearman', annot=False)
        pearson_val = fig_p.get_axes()[0].get_images()[0].get_array()[0, 1]
        spearman_val = fig_s.get_axes()[0].get_images()[0].get_array()[0, 1]
        assert pearson_val != pytest.approx(spearman_val, abs=0.01)

    def test_invalid_method_raises(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 3))
        with pytest.raises(ValueError, match="Unknown method"):
            plot_correlation_heatmap(data, method='invalid_method')

    def test_spearman_with_ties(self):
        x = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        y = np.array([2, 2, 4, 4, 6, 6, 8, 8, 10, 10])
        data = np.column_stack([x, y])
        fig = plot_correlation_heatmap(data, method='spearman')
        assert isinstance(fig, plt.Figure)

    def test_kendall_with_ties(self):
        x = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        y = np.array([2, 2, 4, 4, 6, 6, 8, 8, 10, 10])
        data = np.column_stack([x, y])
        fig = plot_correlation_heatmap(data, method='kendall')
        assert isinstance(fig, plt.Figure)

    def test_2d_data_two_variables_independent(self):
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = np.random.normal(0, 1, 200)
        data = np.column_stack([x, y])
        fig = plot_correlation_heatmap(data, annot=False)
        ax = fig.get_axes()[0]
        image_data = ax.get_images()[0].get_array()
        assert abs(image_data[0, 1]) < 0.3

    def test_preserves_number_of_variables(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 5))
        fig = plot_correlation_heatmap(data)
        ax = fig.get_axes()[0]
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(tick_labels) == 5


# ---------------------------------------------------------------------------
# plot_probability_bands
# ---------------------------------------------------------------------------


class TestPlotProbabilityBands:
    def test_returns_figure_default(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data)
        assert isinstance(fig, plt.Figure)

    def test_single_axes(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data)
        assert len(fig.get_axes()) == 1

    def test_default_title(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data)
        ax = fig.get_axes()[0]
        assert 'Probability Bands' in ax.get_title()

    def test_with_explicit_params(self):
        np.random.seed(42)
        data = np.random.normal(5, 2, 100)
        fig = plot_probability_bands(data, dist='norm', params=(5, 2))
        assert isinstance(fig, plt.Figure)

    def test_without_params_fits_automatically(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data)
        assert isinstance(fig, plt.Figure)

    def test_default_confidence_levels(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data)
        ax = fig.get_axes()[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert '68.0% CI' in legend_texts
        assert '95.0% CI' in legend_texts
        assert '99.7% CI' in legend_texts

    def test_custom_confidence_levels(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data, confidence_levels=[0.50, 0.90])
        ax = fig.get_axes()[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert '50.0% CI' in legend_texts
        assert '90.0% CI' in legend_texts

    def test_single_confidence_level(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data, confidence_levels=[0.95])
        assert isinstance(fig, plt.Figure)

    def test_empty_confidence_levels(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data, confidence_levels=[])
        assert isinstance(fig, plt.Figure)

    def test_has_data_line(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data)
        ax = fig.get_axes()[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'Data' in legend_texts

    def test_has_mean_line(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data)
        ax = fig.get_axes()[0]
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'Mean' in legend_texts

    def test_custom_figsize(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data, figsize=(10, 5))
        assert isinstance(fig, plt.Figure)

    def test_with_exponential_distribution(self):
        np.random.seed(42)
        data = np.random.exponential(2, 100)
        fig = plot_probability_bands(data, dist='expon')
        assert isinstance(fig, plt.Figure)

    def test_with_gamma_distribution_and_params(self):
        np.random.seed(42)
        data = np.random.gamma(2, scale=3, size=100)
        fig = plot_probability_bands(data, dist='gamma', params=(2, 0, 3))
        assert isinstance(fig, plt.Figure)

    def test_small_data(self):
        np.random.seed(42)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        fig = plot_probability_bands(data)
        assert isinstance(fig, plt.Figure)

    def test_bands_have_fill_between(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_probability_bands(data)
        ax = fig.get_axes()[0]
        assert len(ax.collections) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_distribution_comparison_empty_distributions_list(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        fig = plot_distribution_comparison(data, [])
        assert isinstance(fig, plt.Figure)

    def test_distribution_comparison_single_data_point(self):
        data = np.array([5.0])
        with pytest.warns(RuntimeWarning):
            fig = plot_distribution_comparison(data, ['norm'])
        assert isinstance(fig, plt.Figure)

    def test_qq_empty_data_raises(self):
        data = np.array([])
        with pytest.raises(Exception):
            plot_qq(data)

    def test_qq_single_data_point(self):
        data = np.array([5.0])
        with pytest.warns(RuntimeWarning):
            fig = plot_qq(data)
        assert isinstance(fig, plt.Figure)

    def test_histogram_empty_data_raises(self):
        data = np.array([])
        with pytest.raises(Exception):
            plot_histogram_with_fit(data)

    def test_histogram_single_data_point(self):
        data = np.array([5.0])
        with pytest.warns(RuntimeWarning):
            fig = plot_histogram_with_fit(data)
        assert isinstance(fig, plt.Figure)

    def test_probability_bands_single_value_data(self):
        data = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        with pytest.warns(RuntimeWarning):
            fig = plot_probability_bands(data)
        assert isinstance(fig, plt.Figure)

    def test_probability_bands_with_2d_data(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 1))
        fig = plot_probability_bands(data)
        assert isinstance(fig, plt.Figure)

    def test_qq_with_lognormal_distribution(self):
        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, 200)
        fig = plot_qq(data, dist='lognorm')
        assert isinstance(fig, plt.Figure)

    def test_histogram_with_lognormal_distribution(self):
        np.random.seed(42)
        data = np.random.lognormal(0, 0.5, 200)
        fig = plot_histogram_with_fit(data, dist='lognorm')
        assert isinstance(fig, plt.Figure)

    def test_correlation_heatmap_empty_2d_data_raises(self):
        data = np.empty((0, 3))
        with pytest.raises(Exception):
            plot_correlation_heatmap(data)

    def test_correlation_heatmap_single_row_2d(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (1, 4))
        with pytest.warns(RuntimeWarning):
            fig = plot_correlation_heatmap(data)
        assert isinstance(fig, plt.Figure)

    def test_correlation_heatmap_1d_data_raises(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        with pytest.raises(Exception):
            plot_correlation_heatmap(data)

    def test_correlation_heatmap_single_column_2d_raises(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 1))
        with pytest.raises(Exception):
            plot_correlation_heatmap(data)

    def test_correlation_heatmap_1d_list_input_raises(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 50).tolist()
        with pytest.raises(Exception):
            plot_correlation_heatmap(data)

    def test_all_functions_return_figures(self):
        np.random.seed(42)
        data_1d = np.random.normal(0, 1, 100)
        data_2d = np.random.normal(0, 1, (100, 3))

        assert isinstance(plot_distribution_comparison(data_1d, ['norm']), plt.Figure)
        assert isinstance(plot_qq(data_1d), plt.Figure)
        assert isinstance(plot_histogram_with_fit(data_1d), plt.Figure)
        assert isinstance(plot_correlation_heatmap(data_2d), plt.Figure)
        assert isinstance(plot_probability_bands(data_1d), plt.Figure)