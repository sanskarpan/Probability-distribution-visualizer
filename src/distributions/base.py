"""Base distribution class for all probability distributions."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class Distribution(ABC):
    """Abstract base class for probability distributions."""

    def __init__(self, name: str, is_discrete: bool = False):
        """
        Initialize the distribution.

        Args:
            name: Name of the distribution
            is_discrete: Whether the distribution is discrete
        """
        self.name = name
        self.is_discrete = is_discrete
        self._dist = None
        logger.debug("Distribution '%s' created (discrete=%s)", name, is_discrete)

    @abstractmethod
    def _create_distribution(self, **params):
        """
        Create the scipy distribution object.

        Args:
            **params: Distribution parameters

        Returns:
            scipy distribution object
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current distribution parameters.

        Returns:
            Dictionary of parameter names and values
        """
        pass

    @abstractmethod
    def set_parameters(self, **params):
        """
        Set distribution parameters.

        Args:
            **params: Parameter names and values
        """
        pass

    @abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get valid parameter ranges.

        Returns:
            Dictionary of parameter names and (min, max) tuples
        """
        pass

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate probability density function (or PMF for discrete).

        Args:
            x: Input values

        Returns:
            PDF/PMF values
        """
        if self._dist is None:
            logger.error("Attempted to call pdf() on uninitialized distribution '%s'", self.name)
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        if self.is_discrete:
            return self._dist.pmf(x)
        return self._dist.pdf(x)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate cumulative distribution function.

        Args:
            x: Input values

        Returns:
            CDF values
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.cdf(x)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate percent point function (inverse CDF).

        Args:
            q: Probabilities (between 0 and 1)

        Returns:
            Quantile values
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.ppf(q)

    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the distribution.

        Args:
            size: Number of samples to generate
            random_state: Random seed for reproducibility

        Returns:
            Array of random samples
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.rvs(size=size, random_state=random_state)

    def mean(self) -> float:
        """
        Calculate the mean of the distribution.

        Returns:
            Mean value
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.mean()

    def var(self) -> float:
        """
        Calculate the variance of the distribution.

        Returns:
            Variance value
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.var()

    def std(self) -> float:
        """
        Calculate the standard deviation of the distribution.

        Returns:
            Standard deviation value
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.std()

    def median(self) -> float:
        """
        Calculate the median of the distribution.

        Returns:
            Median value
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.median()

    def mode(self) -> float:
        """
        Calculate the mode of the distribution.

        Returns:
            Mode value (may not be available for all distributions)
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        support = self.get_support()
        lower, upper = support[0], support[1]

        if np.isinf(lower) or np.isinf(upper):
            try:
                return float(self._dist.mode())
            except Exception:
                bounds = (-1e6, 1e6)
                if not np.isinf(lower):
                    bounds = (lower, min(upper, bounds[1]) if not np.isinf(upper) else bounds[1])
                elif not np.isinf(upper):
                    bounds = (max(lower, bounds[0]), upper)
                lower_b, upper_b = bounds

                if self.is_discrete:
                    x = np.arange(int(lower_b), int(upper_b) + 1)
                else:
                    x = np.linspace(lower_b, upper_b, 10000)

                pdf_values = self.pdf(x)
                return float(x[np.argmax(pdf_values)])

        if self.is_discrete:
            x = np.arange(lower, upper + 1)
        else:
            x = np.linspace(lower, upper, 10000)

        pdf_values = self.pdf(x)
        return float(x[np.argmax(pdf_values)])

    def skewness(self) -> float:
        """
        Calculate the skewness of the distribution.

        Returns:
            Skewness value
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        stats_result = self._dist.stats(moments='s')
        return float(stats_result)

    def kurtosis(self) -> float:
        """
        Calculate the excess kurtosis of the distribution.

        Returns:
            Excess kurtosis value
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        stats_result = self._dist.stats(moments='k')
        return float(stats_result)

    def entropy(self) -> float:
        """
        Calculate the differential entropy of the distribution.

        Returns:
            Entropy value
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.entropy()

    def get_support(self) -> Tuple[float, float]:
        """
        Get the support (valid range) of the distribution.

        Returns:
            Tuple of (min, max) values
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.support()

    def interval(self, alpha: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval.

        Args:
            alpha: Confidence level (e.g., 0.95 for 95% confidence)

        Returns:
            Tuple of (lower, upper) bounds
        """
        if self._dist is None:
            raise ValueError("Distribution not initialized. Call set_parameters first.")

        return self._dist.interval(alpha)

    def get_statistics(self) -> Dict[str, Optional[float]]:
        """
        Get comprehensive statistics for the distribution.

        Returns:
            Dictionary of statistic names and values
        """
        try:
            stats_dict: Dict[str, Optional[float]] = {
                "mean": self.mean(),
                "variance": self.var(),
                "std_dev": self.std(),
                "median": self.median(),
            }

            try:
                stats_dict["mode"] = self.mode()
            except Exception:
                stats_dict["mode"] = None

            try:
                stats_dict["skewness"] = self.skewness()
            except Exception:
                stats_dict["skewness"] = None

            try:
                stats_dict["kurtosis"] = self.kurtosis()
            except Exception:
                stats_dict["kurtosis"] = None

            try:
                stats_dict["entropy"] = self.entropy()
            except Exception:
                stats_dict["entropy"] = None

            return stats_dict
        except Exception as e:
            logger.error("Error calculating statistics for '%s': %s", self.name, e)
            raise ValueError(f"Error calculating statistics: {e}")

    def __repr__(self) -> str:
        """String representation of the distribution."""
        params = self.get_parameters()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.name}({param_str})"
