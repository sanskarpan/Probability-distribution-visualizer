"""Tests for multivariate distributions."""

import pytest
import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distributions.multivariate import (
    MultivariateDistribution,
    MultivariateNormalDistribution,
    DirichletDistribution,
    MultivariateStudentT,
    WishartDistribution,
    plot_bivariate_normal,
    plot_dirichlet_simplex,
)


class TestMultivariateNormal:
    """Test Multivariate Normal distribution."""

    def test_creation(self):
        """Test distribution creation."""
        mean = np.array([0, 0])
        cov = np.array([[1, 0.5], [0.5, 1]])
        dist = MultivariateNormalDistribution(mean, cov)
        assert dist.name == "Multivariate Normal"
        assert dist.dimension == 2

    def test_invalid_covariance(self):
        """Test that non-positive definite covariance raises error."""
        mean = np.array([0, 0])
        # Non-positive definite matrix
        cov = np.array([[1, 2], [2, 1]])
        with pytest.raises(ValueError):
            MultivariateNormalDistribution(mean, cov)

    def test_pdf(self):
        """Test PDF calculation."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        dist = MultivariateNormalDistribution(mean, cov)

        # PDF at mean should be maximum
        pdf_at_mean = dist.pdf(np.array([[0, 0]]))
        assert pdf_at_mean > 0

        # PDF should decrease away from mean
        pdf_away = dist.pdf(np.array([[2, 2]]))
        assert pdf_away < pdf_at_mean

    def test_logpdf(self):
        """Test log PDF calculation."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        dist = MultivariateNormalDistribution(mean, cov)
        logpdf_vals = dist.logpdf(np.array([[0, 0], [1, 1]]))
        assert len(logpdf_vals) == 2
        assert logpdf_vals[0] > logpdf_vals[1]

    def test_mean(self):
        """Test mean calculation."""
        mean = np.array([1, 2, 3])
        cov = np.eye(3)
        dist = MultivariateNormalDistribution(mean, cov)
        assert np.allclose(dist.mean(), mean)

    def test_covariance(self):
        """Test covariance calculation."""
        mean = np.array([0, 0])
        cov = np.array([[1, 0.5], [0.5, 2]])
        dist = MultivariateNormalDistribution(mean, cov)
        assert np.allclose(dist.cov(), cov)

    def test_marginal(self):
        """Test marginal distribution."""
        mean = np.array([1, 2, 3])
        cov = np.array([[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 1.5]])
        dist = MultivariateNormalDistribution(mean, cov)

        # Get marginal for first two dimensions
        marginal = dist.marginal([0, 1])
        assert marginal.dimension == 2
        assert np.allclose(marginal.mean(), mean[:2])
        assert np.allclose(marginal.cov(), cov[:2, :2])

    def test_conditional(self):
        """Test conditional distribution."""
        mean = np.array([0, 0])
        cov = np.array([[1, 0.8], [0.8, 1]])
        dist = MultivariateNormalDistribution(mean, cov)

        # Conditional distribution X1 | X2 = 1
        cond_dist = dist.conditional([1], np.array([1]))
        assert isinstance(cond_dist, MultivariateNormalDistribution)
        assert cond_dist.dimension == 1

    def test_conditional_complex(self):
        """Test conditional with multiple conditioning variables."""
        mean = np.array([0, 1, 2])
        cov = np.array([[2, 0.5, 0.3], [0.5, 1, 0.2], [0.3, 0.2, 1.5]])
        dist = MultivariateNormalDistribution(mean, cov)
        cond_dist = dist.conditional([0, 2], np.array([1.0, 3.0]))
        assert isinstance(cond_dist, MultivariateNormalDistribution)
        assert cond_dist.dimension == 1

    def test_mahalanobis_2d(self):
        """Test Mahalanobis with 2D input."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        dist = MultivariateNormalDistribution(mean, cov)
        x = np.array([[1, 0], [0, 1], [1, 1]])
        distances = dist.mahalanobis(x)
        assert len(distances) == 3
        assert np.allclose(distances[0], 1.0)
        assert np.allclose(distances[2], np.sqrt(2))

    def test_random_samples(self):
        """Test random sample generation."""
        mean = np.array([1, 2])
        cov = np.eye(2)
        dist = MultivariateNormalDistribution(mean, cov)

        samples = dist.rvs(size=1000, random_state=42)
        assert samples.shape == (1000, 2)

        # Check approximate mean
        sample_mean = np.mean(samples, axis=0)
        assert np.allclose(sample_mean, mean, atol=0.2)

    def test_logpdf_actual_values(self):
        """Test log PDF with known values for standard bivariate normal."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        dist = MultivariateNormalDistribution(mean, cov)
        x = np.array([[0, 0], [1, 0]])
        logpdf_vals = dist.logpdf(x)
        expected_origin = -np.log(2 * np.pi)
        assert np.isclose(logpdf_vals[0], expected_origin)
        assert np.isclose(logpdf_vals[1], expected_origin - 0.5)

    def test_conditional_actual_values(self):
        """Test conditional distribution returns correct parameters."""
        mean = np.array([0, 0])
        cov = np.array([[1, 0.8], [0.8, 1]])
        dist = MultivariateNormalDistribution(mean, cov)
        cond_dist = dist.conditional([1], np.array([1.0]))
        assert np.isclose(cond_dist.mean(), 0.8)
        assert np.isclose(cond_dist.cov(), 0.36)

    def test_marginal_single_index(self):
        """Test marginal with a single index."""
        mean = np.array([1, 2, 3])
        cov = np.array([[1, 0.5, 0.2], [0.5, 2, 0.3], [0.2, 0.3, 1.5]])
        dist = MultivariateNormalDistribution(mean, cov)
        marginal = dist.marginal([2])
        assert marginal.dimension == 1
        assert np.isclose(marginal.mean()[0], 3.0)
        assert np.isclose(marginal.cov()[0, 0], 1.5)

    def test_mean_not_1d_error(self):
        """Test that 2D mean raises ValueError."""
        mean = np.array([[0, 0]])
        cov = np.eye(2)
        with pytest.raises(ValueError, match="1-dimensional"):
            MultivariateNormalDistribution(mean, cov)

    def test_cov_not_2d_error(self):
        """Test that 1D cov raises ValueError."""
        mean = np.array([0, 0])
        cov = np.array([1, 0])
        with pytest.raises(ValueError, match="2-dimensional"):
            MultivariateNormalDistribution(mean, cov)

    def test_cov_not_square_error(self):
        """Test that non-square cov raises ValueError."""
        mean = np.array([0, 0])
        cov = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3]])
        with pytest.raises(ValueError, match="square"):
            MultivariateNormalDistribution(mean, cov)

    def test_mean_cov_dimension_mismatch(self):
        """Test that mismatched mean/cov dimensions raise ValueError."""
        mean = np.array([0, 0, 0])
        cov = np.eye(2)
        with pytest.raises(ValueError, match="dimensions must match"):
            MultivariateNormalDistribution(mean, cov)

    def test_repr(self):
        """Test __repr__."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        dist = MultivariateNormalDistribution(mean, cov)
        assert repr(dist) == "MultivariateNormal(dimension=2)"


class TestDirichlet:
    """Test Dirichlet distribution."""

    def test_creation(self):
        """Test distribution creation."""
        alpha = np.array([1, 2, 3])
        dist = DirichletDistribution(alpha)
        assert dist.name == "Dirichlet"
        assert dist.dimension == 3

    def test_invalid_alpha(self):
        """Test that negative alpha raises error."""
        alpha = np.array([1, -2, 3])
        with pytest.raises(ValueError):
            DirichletDistribution(alpha)

    def test_mean(self):
        """Test mean calculation."""
        alpha = np.array([2, 3, 5])
        dist = DirichletDistribution(alpha)
        expected_mean = alpha / np.sum(alpha)
        assert np.allclose(dist.mean(), expected_mean)

    def test_pdf_2d(self):
        """Test PDF with 2D input."""
        alpha = np.array([2, 2, 2])
        dist = DirichletDistribution(alpha)
        x = np.array([[0.3, 0.3, 0.4], [0.5, 0.3, 0.2]])
        pdf_vals = dist.pdf(x)
        assert len(pdf_vals) == 2
        assert np.all(pdf_vals > 0)

    def test_logpdf(self):
        """Test log PDF."""
        alpha = np.array([2, 2, 2])
        dist = DirichletDistribution(alpha)
        x = np.array([[0.3, 0.3, 0.4], [0.4, 0.3, 0.3]])
        logpdf_vals = dist.logpdf(x)
        assert len(logpdf_vals) == 2
        assert np.all(np.isfinite(logpdf_vals))

    def test_var(self):
        """Test variance per component."""
        alpha = np.array([2, 3, 5])
        dist = DirichletDistribution(alpha)
        var_vals = dist.var()
        assert len(var_vals) == 3
        assert np.all(var_vals > 0)

    def test_cov(self):
        """Test covariance matrix."""
        alpha = np.array([2, 3, 5])
        dist = DirichletDistribution(alpha)
        cov_mat = dist.cov()
        assert cov_mat.shape == (3, 3)
        assert np.allclose(np.diag(cov_mat), dist.var())

    def test_mode(self):
        """Test mode when alpha > 1."""
        alpha = np.array([3, 4, 5])
        dist = DirichletDistribution(alpha)
        mode_vals = dist.mode()
        expected = (alpha - 1) / (np.sum(alpha) - len(alpha))
        assert np.allclose(mode_vals, expected)

    def test_entropy(self):
        """Test differential entropy."""
        alpha = np.array([2, 2, 2])
        dist = DirichletDistribution(alpha)
        ent = dist.entropy()
        assert np.isfinite(ent)

    def test_random_samples(self):
        """Test random sample generation."""
        alpha = np.array([1, 1, 1])
        dist = DirichletDistribution(alpha)

        samples = dist.rvs(size=1000, random_state=42)
        assert samples.shape == (1000, 3)

        # Each row should sum to 1
        assert np.allclose(np.sum(samples, axis=1), 1.0)

        # All values should be in [0, 1]
        assert np.all(samples >= 0)
        assert np.all(samples <= 1)

    def test_concentration(self):
        """Test that higher alpha values lead to lower variance."""
        alpha_low = np.array([1, 1, 1])
        alpha_high = np.array([10, 10, 10])

        dist_low = DirichletDistribution(alpha_low)
        dist_high = DirichletDistribution(alpha_high)

        samples_low = dist_low.rvs(size=1000, random_state=42)
        samples_high = dist_high.rvs(size=1000, random_state=43)

        var_low = np.var(samples_low[:, 0])
        var_high = np.var(samples_high[:, 0])

        assert var_low > var_high

    def test_mode_error_alpha_leq_1(self):
        """Test that mode raises ValueError when any alpha <= 1."""
        alpha = np.array([2, 1, 3])
        dist = DirichletDistribution(alpha)
        with pytest.raises(ValueError, match="only defined"):
            dist.mode()

    def test_repr(self):
        """Test __repr__."""
        alpha = np.array([2, 3, 5])
        dist = DirichletDistribution(alpha)
        assert repr(dist) == "Dirichlet(dimension=3, alpha=[2 3 5])"


class TestMultivariateStudentT:
    """Test Multivariate Student-t distribution."""

    def test_creation(self):
        """Test distribution creation."""
        loc = np.array([0, 0])
        shape = np.eye(2)
        df = 5
        dist = MultivariateStudentT(df, loc, shape)
        assert dist.name == "Multivariate Student-t"
        assert dist.dimension == 2

    def test_invalid_df(self):
        """Test that invalid degrees of freedom raises error."""
        loc = np.array([0, 0])
        shape = np.eye(2)
        with pytest.raises(ValueError):
            MultivariateStudentT(df=0, loc=loc, shape=shape)

    def test_mean(self):
        """Test mean calculation."""
        loc = np.array([1, 2])
        shape = np.eye(2)
        dist = MultivariateStudentT(df=10, loc=loc, shape=shape)
        assert np.allclose(dist.mean(), loc)

    def test_mean_df_leq_1(self):
        """Test that mean raises ValueError for df <= 1."""
        loc = np.array([1, 2])
        shape = np.eye(2)
        dist = MultivariateStudentT(df=0.5, loc=loc, shape=shape)
        with pytest.raises(ValueError):
            dist.mean()

    def test_cov(self):
        """Test covariance matrix."""
        loc = np.array([0, 0])
        shape = np.array([[2, 0.5], [0.5, 1]])
        df = 10
        dist = MultivariateStudentT(df, loc, shape)
        expected_cov = shape * (df / (df - 2))
        assert np.allclose(dist.cov(), expected_cov)

    def test_rvs_with_random_state(self):
        """Test rvs with explicit random_state."""
        loc = np.array([0, 0])
        shape = np.eye(2)
        dist = MultivariateStudentT(df=5, loc=loc, shape=shape)
        samples = dist.rvs(size=50, random_state=42)
        assert samples.shape == (50, 2)

    def test_pdf(self):
        """Test PDF calculation."""
        loc = np.array([0, 0])
        shape = np.eye(2)
        dist = MultivariateStudentT(df=5, loc=loc, shape=shape)
        pdf_val = dist.pdf(np.array([[0, 0]]))
        assert pdf_val[0] > 0
        assert np.isfinite(pdf_val[0])

    def test_random_samples(self):
        """Test random sample generation."""
        loc = np.array([0, 0])
        shape = np.eye(2)
        dist = MultivariateStudentT(df=5, loc=loc, shape=shape)

        samples = dist.rvs(size=100, random_state=42)
        assert samples.shape == (100, 2)

    def test_heavier_tails_than_normal(self):
        """Test that Student-t has heavier tails than Normal."""
        mean = np.array([0, 0])
        cov = np.eye(2)

        mvn = MultivariateNormalDistribution(mean, cov)
        mvt = MultivariateStudentT(df=3, loc=mean, shape=cov)

        # Generate samples
        samples_n = mvn.rvs(size=10000, random_state=42)
        samples_t = mvt.rvs(size=10000, random_state=43)

        # Student-t should have more extreme values
        extreme_n = np.sum(np.linalg.norm(samples_n, axis=1) > 3)
        extreme_t = np.sum(np.linalg.norm(samples_t, axis=1) > 3)

        assert extreme_t > extreme_n

    def test_pdf_actual_values(self):
        """Test PDF returns expected value at origin for bivariate t(df=1)."""
        loc = np.array([0, 0])
        shape = np.eye(2)
        dist = MultivariateStudentT(df=1, loc=loc, shape=shape)
        pdf_val = dist.pdf(np.array([[0, 0]]))
        expected = 1 / (2 * np.pi)
        assert np.isclose(pdf_val[0], expected)

    def test_rvs_without_random_state(self):
        """Test rvs with random_state=None."""
        loc = np.array([0, 0])
        shape = np.eye(2)
        dist = MultivariateStudentT(df=5, loc=loc, shape=shape)
        samples = dist.rvs(size=30, random_state=None)
        assert samples.shape == (30, 2)

    def test_cov_error_df_leq_2(self):
        """Test that cov raises ValueError for df <= 2."""
        loc = np.array([0, 0])
        shape = np.eye(2)
        dist = MultivariateStudentT(df=2, loc=loc, shape=shape)
        with pytest.raises(ValueError, match="only defined"):
            dist.cov()

    def test_shape_not_square_error(self):
        """Test that non-square shape raises ValueError."""
        loc = np.array([0, 0])
        shape = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3]])
        with pytest.raises(ValueError, match="square"):
            MultivariateStudentT(df=5, loc=loc, shape=shape)

    def test_loc_not_1d_error(self):
        """Test that 2D loc raises ValueError."""
        loc = np.array([[0, 0]])
        shape = np.eye(2)
        with pytest.raises(ValueError, match="1-dimensional"):
            MultivariateStudentT(df=5, loc=loc, shape=shape)

    def test_repr(self):
        """Test __repr__."""
        loc = np.array([0, 0])
        shape = np.eye(2)
        dist = MultivariateStudentT(df=5, loc=loc, shape=shape)
        assert repr(dist) == "MultivariateStudentT(dimension=2, df=5)"


class TestWishart:
    """Test Wishart distribution."""

    def test_creation(self):
        """Test distribution creation."""
        scale = np.eye(2)
        df = 5
        dist = WishartDistribution(df, scale)
        assert dist.name == "Wishart"
        assert dist.dimension == 2

    def test_invalid_df(self):
        """Test that df < dimension raises error."""
        scale = np.eye(3)
        with pytest.raises(ValueError):
            WishartDistribution(df=2, scale=scale)

    def test_mean(self):
        """Test mean calculation."""
        scale = 2 * np.eye(2)
        df = 5
        dist = WishartDistribution(df, scale)
        expected_mean = df * scale
        assert np.allclose(dist.mean(), expected_mean)

    def test_pdf(self):
        """Test PDF calculation."""
        scale = np.eye(2)
        df = 5
        dist = WishartDistribution(df, scale)
        x = np.array([[2, 0.5], [0.5, 2]])
        pdf_val = dist.pdf(x)
        assert pdf_val > 0

    def test_logpdf(self):
        """Test log PDF calculation."""
        scale = np.eye(2)
        df = 5
        dist = WishartDistribution(df, scale)
        x = np.array([[2, 0.5], [0.5, 2]])
        logpdf_val = dist.logpdf(x)
        assert np.isfinite(logpdf_val)

    def test_mode(self):
        """Test mode matrix."""
        scale = np.eye(2)
        df = 6
        dist = WishartDistribution(df, scale)
        mode_val = dist.mode()
        assert mode_val.shape == (2, 2)
        assert np.all(np.diag(mode_val) > 0)

    def test_mode_invalid_df(self):
        """Test that mode raises for df < dimension + 1."""
        scale = np.eye(2)
        df = 2
        dist = WishartDistribution(df, scale)
        with pytest.raises(ValueError, match="only defined"):
            dist.mode()

    def test_random_samples(self):
        """Test random sample generation."""
        scale = np.eye(2)
        df = 5
        dist = WishartDistribution(df, scale)

        samples = dist.rvs(size=10, random_state=42)
        assert samples.shape == (10, 2, 2)

        # All samples should be symmetric positive definite
        for sample in samples:
            assert np.allclose(sample, sample.T)  # Symmetric
            eigenvalues = np.linalg.eigvalsh(sample)
            assert np.all(eigenvalues > 0)  # Positive definite

    def test_mode_actual_values(self):
        """Test mode returns expected values."""
        scale = np.eye(2)
        df = 5
        dist = WishartDistribution(df, scale)
        mode_val = dist.mode()
        expected = (df - dist.dimension - 1) * scale
        assert np.allclose(mode_val, expected)

    def test_rvs_with_random_state(self):
        """Test rvs reproducibility with fixed random_state."""
        scale = np.eye(2)
        df = 5
        dist = WishartDistribution(df, scale)
        samples1 = dist.rvs(size=5, random_state=42)
        samples2 = dist.rvs(size=5, random_state=42)
        assert np.allclose(samples1, samples2)

    def test_repr(self):
        """Test __repr__."""
        scale = np.eye(2)
        df = 5
        dist = WishartDistribution(df, scale)
        assert repr(dist) == "Wishart(dimension=2, df=5)"


def test_mahalanobis_distance():
    """Test Mahalanobis distance calculation."""
    mean = np.array([0, 0])
    cov = np.eye(2)
    dist = MultivariateNormalDistribution(mean, cov)

    x = np.array([[1, 0]])
    distance = dist.mahalanobis(x)

    # For identity covariance, Mahalanobis = Euclidean
    expected = np.sqrt(1**2 + 0**2)
    assert np.allclose(distance, expected)


class TestPlotFunctions:
    """Test multivariate plotting functions."""

    def test_plot_bivariate_normal(self):
        """Test that plot_bivariate_normal returns figure and axes."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        dist = MultivariateNormalDistribution(mean, cov)
        fig, axes = plot_bivariate_normal(dist, num_points=20, num_contours=5)
        assert fig is not None
        assert axes is not None
        plt.close('all')

    def test_plot_bivariate_normal_dimension_error(self):
        """Test that plot_bivariate_normal raises for dim != 2."""
        mean = np.array([0, 0, 0])
        cov = np.eye(3)
        dist = MultivariateNormalDistribution(mean, cov)
        with pytest.raises(ValueError, match="only plot bivariate"):
            plot_bivariate_normal(dist)
        plt.close('all')

    def test_plot_dirichlet_simplex(self):
        """Test that plot_dirichlet_simplex returns figure and axes."""
        alpha = np.array([2, 2, 2])
        dist = DirichletDistribution(alpha)
        fig, ax = plot_dirichlet_simplex(dist, num_samples=50)
        assert fig is not None
        assert ax is not None
        plt.close('all')

    def test_plot_dirichlet_simplex_dimension_error(self):
        """Test that plot_dirichlet_simplex raises for dim != 3."""
        alpha = np.array([1, 1])
        dist = DirichletDistribution(alpha)
        with pytest.raises(ValueError, match="only plot 3"):
            plot_dirichlet_simplex(dist)
        plt.close('all')


class TestMultivariateDistributionBase:
    """Test base class abstract methods."""

    def test_pdf_not_implemented(self):
        """Test base pdf raises NotImplementedError."""
        dist = MultivariateDistribution("test", 2)
        with pytest.raises(NotImplementedError):
            dist.pdf(np.array([0, 0]))

    def test_rvs_not_implemented(self):
        """Test base rvs raises NotImplementedError."""
        dist = MultivariateDistribution("test", 2)
        with pytest.raises(NotImplementedError):
            dist.rvs()

    def test_mean_not_implemented(self):
        """Test base mean raises NotImplementedError."""
        dist = MultivariateDistribution("test", 2)
        with pytest.raises(NotImplementedError):
            dist.mean()

    def test_cov_not_implemented(self):
        """Test base cov raises NotImplementedError."""
        dist = MultivariateDistribution("test", 2)
        with pytest.raises(NotImplementedError):
            dist.cov()
