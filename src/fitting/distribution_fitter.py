"""Distribution fitting to empirical data."""

from typing import Dict, List, Tuple, Optional, Type
import logging
import numpy as np
from scipy import stats, optimize
from scipy.special import gammaln
import warnings

logger = logging.getLogger(__name__)


class DistributionFitter:
    """Fit probability distributions to empirical data."""

    def __init__(self, data: np.ndarray):
        """
        Initialize distribution fitter.

        Args:
            data: Empirical data to fit
        """
        self.data = np.asarray(data).flatten()
        self.n = len(self.data)

        if self.n < 2:
            raise ValueError("Need at least 2 data points")

    def fit_all(self, distributions: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Fit multiple distributions and rank by goodness of fit.

        Args:
            distributions: List of distribution names to try.
                          If None, tries all supported distributions.

        Returns:
            Dictionary of results sorted by fit quality (AIC)
        """
        if distributions is None:
            distributions = [
                'norm', 'expon', 'gamma', 'beta', 'lognorm',
                'weibull_min', 't', 'chi2', 'uniform', 'cauchy'
            ]

        results = {}

        logger.info(
            "Starting distribution fitting for %d candidate(s) on %d data points",
            len(distributions),
            self.n,
        )

        for dist_name in distributions:
            try:
                result = self.fit_distribution(dist_name)
                results[dist_name] = result
            except Exception as e:
                logger.warning("Failed to fit distribution '%s': %s", dist_name, e)
                warnings.warn(f"Failed to fit {dist_name}: {e}")
                continue

        # Sort by AIC (lower is better)
        sorted_results = dict(sorted(results.items(),
                                    key=lambda x: x[1]['aic']))

        if sorted_results:
            best = next(iter(sorted_results))
            best_aic = sorted_results[best]["aic"]
            logger.info(
                "Distribution fitting complete: best='%s' (AIC=%.2f), %d/%d succeeded",
                best,
                best_aic,
                len(sorted_results),
                len(distributions),
            )
        else:
            logger.warning("Distribution fitting complete: no distributions fit successfully")

        return sorted_results

    def fit_distribution(self, dist_name: str) -> Dict:
        """
        Fit a specific distribution using Maximum Likelihood Estimation.

        Args:
            dist_name: Name of scipy.stats distribution

        Returns:
            Dictionary with fit results
        """
        dist = getattr(stats, dist_name)

        # Fit using MLE
        params = dist.fit(self.data)

        # Calculate likelihood
        log_likelihood = np.sum(dist.logpdf(self.data, *params))

        # Calculate information criteria
        k = len(params)  # Number of parameters
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(self.n) - 2 * log_likelihood

        # Perform goodness-of-fit tests
        ks_statistic, ks_pvalue = stats.kstest(self.data,
                                                lambda x: dist.cdf(x, *params))

        logger.debug(
            "Fitted '%s': params=%s, AIC=%.2f, BIC=%.2f, KS_p=%.4f",
            dist_name,
            params,
            aic,
            bic,
            ks_pvalue,
        )

        return {
            'distribution': dist_name,
            'parameters': params,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'fitted_dist': dist(*params),
        }

    def fit_normal(self) -> Tuple[float, float]:
        """
        Fit normal distribution using MLE.

        Returns:
            Tuple of (mu, sigma)
        """
        mu = np.mean(self.data)
        sigma = np.std(self.data, ddof=1)
        return mu, sigma

    def fit_exponential(self) -> float:
        """
        Fit exponential distribution using MLE.

        Returns:
            Lambda parameter
        """
        lambda_param = 1.0 / np.mean(self.data)
        return lambda_param

    def fit_gamma_mle(self) -> Tuple[float, float]:
        """
        Fit gamma distribution using Maximum Likelihood Estimation.

        Returns:
            Tuple of (shape, scale)
        """
        # Use scipy's built-in MLE
        shape, loc, scale = stats.gamma.fit(self.data, floc=0)
        return shape, scale

    def fit_gamma_mom(self) -> Tuple[float, float]:
        """
        Fit gamma distribution using Method of Moments.

        Returns:
            Tuple of (shape, scale)
        """
        mean = np.mean(self.data)
        var = np.var(self.data, ddof=1)

        # shape = mean^2 / var
        # scale = var / mean
        shape = mean**2 / var
        scale = var / mean

        return shape, scale

    def fit_beta(self) -> Tuple[float, float]:
        """
        Fit beta distribution (data must be in [0, 1]).

        Returns:
            Tuple of (alpha, beta)
        """
        if np.any(self.data < 0) or np.any(self.data > 1):
            raise ValueError("Beta distribution requires data in [0, 1]")

        # Method of moments
        mean = np.mean(self.data)
        var = np.var(self.data, ddof=1)

        # Solve for alpha and beta
        common = mean * (1 - mean) / var - 1
        alpha = mean * common
        beta = (1 - mean) * common

        return alpha, beta

    def fit_weibull(self) -> Tuple[float, float]:
        """
        Fit Weibull distribution using MLE.

        Returns:
            Tuple of (shape, scale)
        """
        shape, loc, scale = stats.weibull_min.fit(self.data, floc=0)
        return shape, scale

    def fit_lognormal(self) -> Tuple[float, float]:
        """
        Fit lognormal distribution.

        Returns:
            Tuple of (mu, sigma) of underlying normal
        """
        if np.any(self.data <= 0):
            raise ValueError("Lognormal requires positive data")

        log_data = np.log(self.data)
        mu = np.mean(log_data)
        sigma = np.std(log_data, ddof=1)

        return mu, sigma

    def qq_plot_data(self, dist_name: str, params: Tuple) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data for Q-Q plot.

        Args:
            dist_name: Distribution name
            params: Distribution parameters

        Returns:
            Tuple of (theoretical_quantiles, sample_quantiles)
        """
        dist = getattr(stats, dist_name)

        # Sort data
        sorted_data = np.sort(self.data)

        # Calculate empirical quantiles
        p = (np.arange(self.n) + 0.5) / self.n

        # Calculate theoretical quantiles
        theoretical_quantiles = dist.ppf(p, *params)

        return theoretical_quantiles, sorted_data

    def calculate_residuals(self, dist_name: str, params: Tuple) -> np.ndarray:
        """
        Calculate standardized residuals.

        Args:
            dist_name: Distribution name
            params: Distribution parameters

        Returns:
            Standardized residuals
        """
        dist = getattr(stats, dist_name)

        # Calculate CDF values for data
        cdf_vals = dist.cdf(self.data, *params)

        # Transform to standard normal
        residuals = stats.norm.ppf(cdf_vals)

        return residuals


class BayesianEstimator:
    """Bayesian parameter estimation."""

    def __init__(self, data: np.ndarray):
        """
        Initialize Bayesian estimator.

        Args:
            data: Observed data
        """
        self.data = np.asarray(data).flatten()
        self.n = len(self.data)

    def estimate_normal_mean(self,
                            prior_mean: float,
                            prior_var: float,
                            known_variance: float) -> Tuple[float, float]:
        """
        Bayesian estimation of normal mean with known variance.

        Args:
            prior_mean: Prior mean
            prior_var: Prior variance
            known_variance: Known data variance

        Returns:
            Tuple of (posterior_mean, posterior_variance)
        """
        # Conjugate prior: Normal
        sample_mean = np.mean(self.data)

        # Posterior parameters
        posterior_var = 1.0 / (1.0/prior_var + self.n/known_variance)
        posterior_mean = posterior_var * (prior_mean/prior_var +
                                         self.n*sample_mean/known_variance)

        return posterior_mean, posterior_var

    def estimate_normal_variance(self,
                                prior_shape: float,
                                prior_scale: float,
                                known_mean: float) -> Tuple[float, float]:
        """
        Bayesian estimation of normal variance with known mean.

        Args:
            prior_shape: Prior shape (Inverse-Gamma)
            prior_scale: Prior scale (Inverse-Gamma)
            known_mean: Known mean

        Returns:
            Tuple of (posterior_shape, posterior_scale)
        """
        # Conjugate prior: Inverse-Gamma
        ss = np.sum((self.data - known_mean)**2)

        # Posterior parameters
        posterior_shape = prior_shape + self.n / 2
        posterior_scale = prior_scale + ss / 2

        return posterior_shape, posterior_scale

    def estimate_poisson_rate(self,
                             prior_shape: float,
                             prior_rate: float) -> Tuple[float, float]:
        """
        Bayesian estimation of Poisson rate parameter.

        Args:
            prior_shape: Prior shape (Gamma)
            prior_rate: Prior rate (Gamma)

        Returns:
            Tuple of (posterior_shape, posterior_rate)
        """
        # Conjugate prior: Gamma
        total = np.sum(self.data)

        # Posterior parameters
        posterior_shape = prior_shape + total
        posterior_rate = prior_rate + self.n

        return posterior_shape, posterior_rate

    def estimate_bernoulli_p(self,
                            prior_alpha: float,
                            prior_beta: float) -> Tuple[float, float]:
        """
        Bayesian estimation of Bernoulli success probability.

        Args:
            prior_alpha: Prior alpha (Beta)
            prior_beta: Prior beta (Beta)

        Returns:
            Tuple of (posterior_alpha, posterior_beta)
        """
        # Conjugate prior: Beta
        successes = np.sum(self.data)
        failures = self.n - successes

        # Posterior parameters
        posterior_alpha = prior_alpha + successes
        posterior_beta = prior_beta + failures

        return posterior_alpha, posterior_beta


class GoodnessOfFit:
    """Goodness of fit tests."""

    @staticmethod
    def chi_square_test(observed: np.ndarray,
                       expected: np.ndarray,
                       df: Optional[int] = None) -> Tuple[float, float]:
        """
        Chi-square goodness of fit test.

        Args:
            observed: Observed frequencies
            expected: Expected frequencies
            df: Degrees of freedom (if None, calculated automatically)

        Returns:
            Tuple of (chi2_statistic, p_value)
        """
        chi2_stat, p_value = stats.chisquare(observed, expected)

        return chi2_stat, p_value

    @staticmethod
    def kolmogorov_smirnov_test(data: np.ndarray,
                                cdf_function) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test.

        Args:
            data: Sample data
            cdf_function: Theoretical CDF function

        Returns:
            Tuple of (ks_statistic, p_value)
        """
        ks_stat, p_value = stats.kstest(data, cdf_function)

        return ks_stat, p_value

    @staticmethod
    def anderson_darling_test(data: np.ndarray,
                             dist: str = 'norm') -> Dict:
        """
        Anderson-Darling test.

        Args:
            data: Sample data
            dist: Distribution name ('norm', 'expon', 'logistic', 'gumbel', 'extreme1')

        Returns:
            Dictionary with test results
        """
        result = stats.anderson(data, dist=dist, method='interpolate')

        return {
            'statistic': result.statistic,
            'pvalue': result.pvalue,
        }

    @staticmethod
    def shapiro_wilk_test(data: np.ndarray) -> Tuple[float, float]:
        """
        Shapiro-Wilk test for normality.

        Args:
            data: Sample data

        Returns:
            Tuple of (w_statistic, p_value)
        """
        if len(data) > 5000:
            warnings.warn("Shapiro-Wilk test may be unreliable for large samples")

        w_stat, p_value = stats.shapiro(data)

        return w_stat, p_value

    @staticmethod
    def jarque_bera_test(data: np.ndarray) -> Tuple[float, float]:
        """
        Jarque-Bera test for normality.

        Args:
            data: Sample data

        Returns:
            Tuple of (jb_statistic, p_value)
        """
        jb_stat, p_value = stats.jarque_bera(data)

        return jb_stat, p_value
