"""Mixture distributions - combinations of multiple distributions."""

from typing import List, Tuple, Optional, Union
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


class MixtureDistribution:
    """General mixture distribution."""

    def __init__(self,
                 components: List,
                 weights: Union[List[float], np.ndarray]):
        """
        Initialize mixture distribution.

        Args:
            components: List of component distributions
            weights: Mixing weights (must sum to 1)
        """
        weights = np.asarray(weights, dtype=float)

        if len(components) != len(weights):
            raise ValueError("Number of components must match number of weights")

        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1")

        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")

        self.components = components
        self.weights = weights
        self.n_components = len(components)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate probability density function.

        Args:
            x: Input values

        Returns:
            PDF values
        """
        x = np.atleast_1d(x)
        pdf_vals = np.zeros_like(x, dtype=float)

        for i, (component, weight) in enumerate(zip(self.components, self.weights)):
            pdf_vals += weight * component.pdf(x)

        return pdf_vals

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate cumulative distribution function.

        Args:
            x: Input values

        Returns:
            CDF values
        """
        x = np.atleast_1d(x)
        cdf_vals = np.zeros_like(x, dtype=float)

        for i, (component, weight) in enumerate(zip(self.components, self.weights)):
            cdf_vals += weight * component.cdf(x)

        return cdf_vals

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples.

        Args:
            size: Number of samples
            random_state: Random seed

        Returns:
            Random samples
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Sample component indices according to weights
        component_indices = np.random.choice(
            self.n_components,
            size=size,
            p=self.weights
        )

        # Sample from each selected component
        samples = np.zeros(size)
        for i in range(self.n_components):
            mask = component_indices == i
            n_samples = np.sum(mask)

            if n_samples > 0:
                comp_samples = self.components[i].rvs(size=n_samples)
                samples[mask] = comp_samples

        return samples

    def mean(self) -> float:
        """
        Calculate mean of mixture.

        Returns:
            Mean value
        """
        mean = 0.0
        for component, weight in zip(self.components, self.weights):
            mean += weight * component.mean()

        return mean

    def var(self) -> float:
        """
        Calculate variance of mixture.

        Returns:
            Variance value
        """
        # Var(X) = E[Var(X|Z)] + Var(E[X|Z])
        # where Z is the component indicator

        # E[Var(X|Z)]
        var_within = 0.0
        for component, weight in zip(self.components, self.weights):
            var_within += weight * component.var()

        # Var(E[X|Z])
        mixture_mean = self.mean()
        var_between = 0.0
        for component, weight in zip(self.components, self.weights):
            diff = component.mean() - mixture_mean
            var_between += weight * diff**2

        return var_within + var_between

    def fit_em(self,
              data: np.ndarray,
              n_components: int,
              max_iter: int = 100,
              tol: float = 1e-4) -> Tuple[np.ndarray, List, List[float]]:
        """
        Fit mixture model using Expectation-Maximization.

        Args:
            data: Observed data
            n_components: Number of mixture components
            max_iter: Maximum EM iterations
            tol: Convergence tolerance

        Returns:
            Tuple of (responsibilities, components, weights)
        """
        n = len(data)

        # Initialize parameters randomly
        weights = np.ones(n_components) / n_components
        means = np.random.choice(data, size=n_components, replace=False)
        stds = np.ones(n_components) * np.std(data)

        for iteration in range(max_iter):
            # E-step: Calculate responsibilities
            responsibilities = np.zeros((n, n_components))

            for k in range(n_components):
                component = stats.norm(loc=means[k], scale=stds[k])
                responsibilities[:, k] = weights[k] * component.pdf(data)

            # Normalize responsibilities
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)

            # M-step: Update parameters
            nk = responsibilities.sum(axis=0)
            new_weights = nk / n

            new_means = np.zeros(n_components)
            new_stds = np.zeros(n_components)

            for k in range(n_components):
                new_means[k] = np.sum(responsibilities[:, k] * data) / nk[k]
                diff_sq = (data - new_means[k])**2
                new_stds[k] = np.sqrt(np.sum(responsibilities[:, k] * diff_sq) / nk[k])

            # Check convergence
            if (np.max(np.abs(new_means - means)) < tol and
                np.max(np.abs(new_stds - stds)) < tol):
                break

            weights = new_weights
            means = new_means
            stds = new_stds

        # Create component distributions
        components = [stats.norm(loc=m, scale=s) for m, s in zip(means, stds)]

        return responsibilities, components, weights.tolist()

    def __repr__(self) -> str:
        return f"MixtureDistribution(n_components={self.n_components})"


class GaussianMixtureModel:
    """Gaussian Mixture Model using sklearn."""

    def __init__(self,
                 n_components: int = 2,
                 covariance_type: str = 'full',
                 max_iter: int = 100):
        """
        Initialize Gaussian Mixture Model.

        Args:
            n_components: Number of mixture components
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            max_iter: Maximum EM iterations
        """
        self.n_components = n_components
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=42
        )
        self.fitted = False

    def fit(self, data: np.ndarray) -> 'GaussianMixtureModel':
        """
        Fit GMM to data.

        Args:
            data: Training data (n x d)

        Returns:
            Self
        """
        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.gmm.fit(data)
        self.fitted = True

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict component labels.

        Args:
            data: Data to predict

        Returns:
            Component labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        return self.gmm.predict(data)

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate log-likelihood of samples.

        Args:
            data: Data to score

        Returns:
            Log-likelihood values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        return self.gmm.score_samples(data)

    def pdf(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate probability density.

        Args:
            data: Data points

        Returns:
            PDF values
        """
        log_pdf = self.score_samples(data)
        return np.exp(log_pdf)

    def rvs(self, size: int = 1) -> np.ndarray:
        """
        Generate random samples.

        Args:
            size: Number of samples

        Returns:
            Samples
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        samples, _ = self.gmm.sample(size)
        return samples.squeeze()

    def bic(self, data: np.ndarray) -> float:
        """
        Calculate Bayesian Information Criterion.

        Args:
            data: Data

        Returns:
            BIC value
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        return self.gmm.bic(data)

    def aic(self, data: np.ndarray) -> float:
        """
        Calculate Akaike Information Criterion.

        Args:
            data: Data

        Returns:
            AIC value
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        return self.gmm.aic(data)

    def get_parameters(self) -> dict:
        """
        Get fitted parameters.

        Returns:
            Dictionary with means, covariances, and weights
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        return {
            'means': self.gmm.means_,
            'covariances': self.gmm.covariances_,
            'weights': self.gmm.weights_
        }


class BayesianGMM:
    """Bayesian Gaussian Mixture Model with automatic component selection."""

    def __init__(self,
                 max_components: int = 10,
                 weight_concentration_prior: float = 1.0,
                 max_iter: int = 200,
                 tol: float = 1e-4):
        """
        Initialize Bayesian GMM.

        Args:
            max_components: Maximum number of components
            weight_concentration_prior: Dirichlet concentration prior
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
        """
        self.max_components = max_components
        self.bgmm = BayesianGaussianMixture(
            n_components=max_components,
            weight_concentration_prior=weight_concentration_prior,
            max_iter=max_iter,
            tol=tol,
            random_state=42
        )
        self.fitted = False

    def fit(self, data: np.ndarray) -> 'BayesianGMM':
        """
        Fit Bayesian GMM to data.

        Args:
            data: Training data

        Returns:
            Self
        """
        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self.bgmm.fit(data)
        self.fitted = True

        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict component labels."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        data = np.atleast_2d(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        return self.bgmm.predict(data)

    def get_active_components(self) -> int:
        """
        Get number of active components (with non-negligible weight).

        Returns:
            Number of active components
        """
        if not self.fitted:
            raise ValueError("Model must be fitted first")

        # Components with weight > 0.01 are considered active
        return np.sum(self.bgmm.weights_ > 0.01)


def select_optimal_components(data: np.ndarray,
                              max_components: int = 10) -> Tuple[int, dict]:
    """
    Select optimal number of components using BIC.

    Args:
        data: Training data
        max_components: Maximum components to try

    Returns:
        Tuple of (optimal_n_components, results_dict)
    """
    data = np.atleast_2d(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    bic_scores: List[float] = []
    aic_scores: List[float] = []

    for n in range(1, max_components + 1):
        gmm = GaussianMixtureModel(n_components=n)
        gmm.fit(data)

        bic_scores.append(gmm.bic(data))
        aic_scores.append(gmm.aic(data))

    # Lower BIC is better
    optimal_n: int = int(np.argmin(bic_scores) + 1)

    results = {
        'optimal_components': optimal_n,
        'bic_scores': bic_scores,
        'aic_scores': aic_scores,
        'components_range': list(range(1, max_components + 1))
    }

    return optimal_n, results
