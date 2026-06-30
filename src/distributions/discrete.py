"""Discrete probability distributions."""

from typing import Dict, Any, Tuple
from scipy import stats
from .base import Distribution


class BinomialDistribution(Distribution):
    """Binomial distribution."""

    def __init__(self, n: int = 10, p: float = 0.5):
        """
        Initialize Binomial distribution.

        Args:
            n: Number of trials (must be positive integer)
            p: Probability of success (must be between 0 and 1)
        """
        super().__init__("Binomial", is_discrete=True)
        self.n = n
        self.p = p
        self.set_parameters(n=n, p=p)

    def _create_distribution(self, **params):
        """Create scipy binomial distribution."""
        return stats.binom(n=params['n'], p=params['p'])

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {"n": self.n, "p": self.p}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.n = int(params.get('n', self.n))
        self.p = params.get('p', self.p)

        if self.n <= 0:
            raise ValueError("n must be positive")
        if not 0 <= self.p <= 1:
            raise ValueError("p must be between 0 and 1")

        self._dist = self._create_distribution(n=self.n, p=self.p)

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "n": (1, 100),
            "p": (0.0, 1.0),
        }


class PoissonDistribution(Distribution):
    """Poisson distribution."""

    def __init__(self, lambda_param: float = 3.0):
        """
        Initialize Poisson distribution.

        Args:
            lambda_param: Rate parameter (must be > 0)
        """
        super().__init__("Poisson", is_discrete=True)
        self.lambda_param = lambda_param
        self.set_parameters(lambda_param=lambda_param)

    def _create_distribution(self, **params):
        """Create scipy poisson distribution."""
        return stats.poisson(mu=params['lambda_param'])

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {"lambda": self.lambda_param}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.lambda_param = params.get('lambda_param', params.get('lambda', self.lambda_param))

        if self.lambda_param <= 0:
            raise ValueError("lambda must be positive")

        self._dist = self._create_distribution(lambda_param=self.lambda_param)

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return {"lambda": (0.1, 20.0)}


class GeometricDistribution(Distribution):
    """Geometric distribution."""

    def __init__(self, p: float = 0.5):
        """
        Initialize Geometric distribution.

        Args:
            p: Probability of success (must be between 0 and 1)
        """
        super().__init__("Geometric", is_discrete=True)
        self.p = p
        self.set_parameters(p=p)

    def _create_distribution(self, **params):
        """Create scipy geometric distribution."""
        return stats.geom(p=params['p'])

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {"p": self.p}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.p = params.get('p', self.p)

        if not 0 < self.p <= 1:
            raise ValueError("p must be between 0 and 1 (exclusive of 0)")

        self._dist = self._create_distribution(p=self.p)

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return {"p": (0.01, 1.0)}


class NegativeBinomialDistribution(Distribution):
    """Negative Binomial distribution."""

    def __init__(self, r: int = 5, p: float = 0.5):
        """
        Initialize Negative Binomial distribution.

        Args:
            r: Number of successes (must be positive integer)
            p: Probability of success (must be between 0 and 1)
        """
        super().__init__("Negative Binomial", is_discrete=True)
        self.r = r
        self.p = p
        self.set_parameters(r=r, p=p)

    def _create_distribution(self, **params):
        """Create scipy negative binomial distribution."""
        return stats.nbinom(n=params['r'], p=params['p'])

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {"r": self.r, "p": self.p}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.r = int(params.get('r', self.r))
        self.p = params.get('p', self.p)

        if self.r <= 0:
            raise ValueError("r must be positive")
        if not 0 < self.p <= 1:
            raise ValueError("p must be between 0 and 1")

        self._dist = self._create_distribution(r=self.r, p=self.p)

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "r": (1, 50),
            "p": (0.01, 1.0),
        }


class HypergeometricDistribution(Distribution):
    """Hypergeometric distribution."""

    def __init__(self, M: int = 20, n: int = 7, N: int = 12):
        """
        Initialize Hypergeometric distribution.

        Args:
            M: Total population size
            n: Number of success states in population
            N: Number of draws
        """
        super().__init__("Hypergeometric", is_discrete=True)
        self.M = M
        self.n = n
        self.N = N
        self.set_parameters(M=M, n=n, N=N)

    def _create_distribution(self, **params):
        """Create scipy hypergeometric distribution."""
        return stats.hypergeom(M=params['M'], n=params['n'], N=params['N'])

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {"M": self.M, "n": self.n, "N": self.N}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.M = int(params.get('M', self.M))
        self.n = int(params.get('n', self.n))
        self.N = int(params.get('N', self.N))

        if self.M <= 0 or self.n < 0 or self.N < 0:
            raise ValueError("M must be positive, n and N must be non-negative")
        if self.n > self.M:
            raise ValueError("n cannot be greater than M")
        if self.N > self.M:
            raise ValueError("N cannot be greater than M")

        self._dist = self._create_distribution(M=self.M, n=self.n, N=self.N)

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "M": (1, 100),
            "n": (0, 100),
            "N": (1, 100),
        }


class DiscreteUniformDistribution(Distribution):
    """Discrete Uniform distribution."""

    def __init__(self, low: int = 1, high: int = 6):
        """
        Initialize Discrete Uniform distribution.

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
        """
        super().__init__("Discrete Uniform", is_discrete=True)
        self.low = low
        self.high = high
        self.set_parameters(low=low, high=high)

    def _create_distribution(self, **params):
        """Create scipy discrete uniform distribution."""
        return stats.randint(low=params['low'], high=params['high'] + 1)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters."""
        return {"low": self.low, "high": self.high}

    def set_parameters(self, **params):
        """Set distribution parameters."""
        self.low = int(params.get('low', self.low))
        self.high = int(params.get('high', self.high))

        if self.low >= self.high:
            raise ValueError("low must be less than high")

        self._dist = self._create_distribution(low=self.low, high=self.high)

    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        return {
            "low": (0, 50),
            "high": (1, 50),
        }
