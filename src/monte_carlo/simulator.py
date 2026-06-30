"""Monte Carlo simulation engine for complex probability calculations."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import logging
import time
import numpy as np
from scipy import stats
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    mean: float
    std: float
    var: float
    median: float
    quantiles: Dict[float, float]
    samples: np.ndarray
    confidence_interval: Tuple[float, float]
    convergence_data: Optional[np.ndarray] = None


class MonteCarloSimulator:
    """Advanced Monte Carlo simulation engine."""

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        self._rng = np.random.RandomState(random_seed)

    def simulate(self,
                func: Callable,
                num_samples: int = 10000,
                track_convergence: bool = False,
                confidence_level: float = 0.95,
                **kwargs) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            func: Function that generates one sample
            num_samples: Number of Monte Carlo samples
            track_convergence: Whether to track convergence
            confidence_level: Confidence level for interval
            **kwargs: Additional arguments passed to func

        Returns:
            SimulationResult object
        """
        logger.info(
            "Monte Carlo simulation starting: %d samples, convergence=%s",
            num_samples,
            track_convergence,
        )
        start_time = time.perf_counter()

        samples = np.array([func(**kwargs) for _ in range(num_samples)])

        # Calculate statistics
        mean = np.mean(samples)
        std = np.std(samples, ddof=1)
        var = np.var(samples, ddof=1)
        median = np.median(samples)

        # Calculate quantiles
        quantiles = {
            0.01: np.percentile(samples, 1),
            0.05: np.percentile(samples, 5),
            0.25: np.percentile(samples, 25),
            0.50: np.percentile(samples, 50),
            0.75: np.percentile(samples, 75),
            0.95: np.percentile(samples, 95),
            0.99: np.percentile(samples, 99),
        }

        # Confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(samples, 100 * alpha / 2)
        ci_upper = np.percentile(samples, 100 * (1 - alpha / 2))

        # Track convergence if requested
        convergence_data = None
        if track_convergence:
            convergence_data = self._calculate_convergence(samples)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Monte Carlo simulation completed: %d samples in %.2fs, mean=%.4f, std=%.4f",
            num_samples,
            elapsed,
            mean,
            std,
        )

        return SimulationResult(
            mean=mean,
            std=std,
            var=var,
            median=median,
            quantiles=quantiles,
            samples=samples,
            confidence_interval=(ci_lower, ci_upper),
            convergence_data=convergence_data
        )

    def _calculate_convergence(self, samples: np.ndarray) -> np.ndarray:
        """Calculate running mean for convergence analysis."""
        running_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
        return running_mean

    def estimate_probability(self,
                            event_func: Callable,
                            num_samples: int = 100000,
                            confidence_level: float = 0.95) -> Dict:
        """
        Estimate probability of an event using Monte Carlo.

        Args:
            event_func: Function that returns True/False for event occurrence
            num_samples: Number of Monte Carlo samples
            confidence_level: Confidence level for interval

        Returns:
            Dictionary with probability estimate and confidence interval
        """
        outcomes = np.array([event_func() for _ in range(num_samples)])
        prob_estimate = np.mean(outcomes)

        # Wilson score interval for binomial proportion
        z = stats.norm.ppf((1 + confidence_level) / 2)
        n = num_samples

        denominator = 1 + z**2 / n
        center = (prob_estimate + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((prob_estimate * (1 - prob_estimate) / n +
                             z**2 / (4 * n**2))) / denominator

        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)

        return {
            'probability': prob_estimate,
            'confidence_interval': (ci_lower, ci_upper),
            'num_samples': num_samples,
            'num_successes': int(np.sum(outcomes))
        }

    def estimate_expectation(self,
                           random_var_func: Callable,
                           num_samples: int = 10000) -> Tuple[float, float]:
        """
        Estimate expectation using Monte Carlo.

        Args:
            random_var_func: Function that generates random variable values
            num_samples: Number of Monte Carlo samples

        Returns:
            Tuple of (expectation_estimate, standard_error)
        """
        samples = np.array([random_var_func() for _ in range(num_samples)])
        expectation = np.mean(samples)
        std_error = np.std(samples, ddof=1) / np.sqrt(num_samples)

        return expectation, std_error

    def importance_sampling(self,
                          target_func: Callable,
                          proposal_sampler: Callable,
                          proposal_pdf: Callable,
                          target_pdf: Callable,
                          num_samples: int = 10000) -> Tuple[float, float]:
        """
        Importance sampling for rare event estimation.

        Args:
            target_func: Function to compute on target distribution
            proposal_sampler: Function to sample from proposal distribution
            proposal_pdf: PDF of proposal distribution
            target_pdf: PDF of target distribution
            num_samples: Number of samples

        Returns:
            Tuple of (estimate, standard_error)
        """
        # Generate samples from proposal
        samples = np.array([proposal_sampler() for _ in range(num_samples)])

        # Calculate importance weights
        weights = target_pdf(samples) / proposal_pdf(samples)

        # Calculate weighted average
        values = np.array([target_func(s) for s in samples])
        estimate = np.average(values, weights=weights)

        # Calculate effective sample size
        ess = np.sum(weights)**2 / np.sum(weights**2)

        # Standard error
        std_error = np.sqrt(np.average((values - estimate)**2,
                                       weights=weights) / ess)

        return estimate, std_error

    def stratified_sampling(self,
                          func: Callable,
                          strata_bounds: List[Tuple[float, float]],
                          num_samples_per_stratum: int = 1000) -> SimulationResult:
        """
        Stratified sampling for variance reduction.

        Args:
            func: Function to evaluate
            strata_bounds: List of (lower, upper) bounds for each stratum
            num_samples_per_stratum: Samples per stratum

        Returns:
            SimulationResult object
        """
        all_samples: list = []

        for lower, upper in strata_bounds:
            # Uniform sampling within stratum
            stratum_samples = self._rng.uniform(lower, upper,
                                               num_samples_per_stratum)
            values = np.array([func(s) for s in stratum_samples])
            all_samples.extend(values)

        samples = np.array(all_samples)

        return SimulationResult(
            mean=np.mean(samples),
            std=np.std(samples, ddof=1),
            var=np.var(samples, ddof=1),
            median=np.median(samples),
            quantiles={
                0.05: np.percentile(samples, 5),
                0.50: np.percentile(samples, 50),
                0.95: np.percentile(samples, 95),
            },
            samples=samples,
            confidence_interval=(np.percentile(samples, 2.5),
                               np.percentile(samples, 97.5))
        )

    def bootstrap(self,
                 data: np.ndarray,
                 statistic: Callable,
                 num_bootstrap: int = 10000,
                 confidence_level: float = 0.95) -> Dict:
        """
        Bootstrap resampling for estimating sampling distribution.

        Args:
            data: Original data
            statistic: Function to compute statistic (takes data, returns scalar)
            num_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for interval

        Returns:
            Dictionary with bootstrap results
        """
        n = len(data)
        bootstrap_stats = np.zeros(num_bootstrap)

        for i in range(num_bootstrap):
            # Resample with replacement
            bootstrap_sample = self._rng.choice(data, size=n, replace=True)
            bootstrap_stats[i] = statistic(bootstrap_sample)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return {
            'estimate': statistic(data),
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats, ddof=1),
            'confidence_interval': (ci_lower, ci_upper),
            'bootstrap_distribution': bootstrap_stats
        }

    def permutation_test(self,
                        group1: np.ndarray,
                        group2: np.ndarray,
                        test_statistic: Callable,
                        num_permutations: int = 10000) -> Dict:
        """
        Permutation test for hypothesis testing.

        Args:
            group1: First group data
            group2: Second group data
            test_statistic: Function that computes test statistic from (group1, group2)
            num_permutations: Number of permutations

        Returns:
            Dictionary with test results
        """
        observed_stat = test_statistic(group1, group2)

        # Pool data
        pooled = np.concatenate([group1, group2])
        n1 = len(group1)
        n_total = len(pooled)

        # Permutation distribution
        perm_stats = np.zeros(num_permutations)

        for i in range(num_permutations):
            # Shuffle and split
            shuffled = self._rng.permutation(pooled)
            perm_group1 = shuffled[:n1]
            perm_group2 = shuffled[n1:]

            perm_stats[i] = test_statistic(perm_group1, perm_group2)

        # Calculate p-value (two-tailed)
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))

        return {
            'observed_statistic': observed_stat,
            'p_value': p_value,
            'permutation_distribution': perm_stats
        }


class VarianceReduction:
    """Variance reduction techniques."""

    @staticmethod
    def antithetic_variates(sampler: Callable,
                          func: Callable,
                          num_pairs: int = 5000) -> Tuple[float, float]:
        """
        Antithetic variates for variance reduction.

        Args:
            sampler: Function that generates uniform(0,1) samples
            func: Function to evaluate
            num_pairs: Number of antithetic pairs

        Returns:
            Tuple of (estimate, standard_error)
        """
        estimates_list = []

        for _ in range(num_pairs):
            u = sampler()
            # Antithetic variate
            u_anti = 1 - u

            y1 = func(u)
            y2 = func(u_anti)

            # Average of pair
            estimates_list.append((y1 + y2) / 2)

        estimates = np.array(estimates_list)
        return np.mean(estimates), np.std(estimates, ddof=1) / np.sqrt(num_pairs)

    @staticmethod
    def control_variates(target_sampler: Callable,
                        target_func: Callable,
                        control_func: Callable,
                        control_mean: float,
                        num_samples: int = 10000) -> Tuple[float, float]:
        """
        Control variates for variance reduction.

        Args:
            target_sampler: Function to generate samples
            target_func: Function to evaluate (unknown expectation)
            control_func: Control function (known expectation)
            control_mean: Known expectation of control function
            num_samples: Number of samples

        Returns:
            Tuple of (estimate, standard_error)
        """
        samples = [target_sampler() for _ in range(num_samples)]
        y_values = np.array([target_func(s) for s in samples])
        c_values = np.array([control_func(s) for s in samples])

        # Optimal coefficient
        cov = np.cov(y_values, c_values)[0, 1]
        var_c = np.var(c_values, ddof=1)
        beta = cov / var_c if var_c > 0 else 0

        # Control variate estimate
        controlled = y_values - beta * (c_values - control_mean)

        return float(np.mean(controlled)), float(np.std(controlled, ddof=1) / np.sqrt(num_samples))


class QuasiMonteCarloSimulator:
    """Quasi-Monte Carlo using low-discrepancy sequences."""

    @staticmethod
    def halton_sequence(n: int, base: int = 2) -> np.ndarray:
        """
        Generate Halton sequence.

        Args:
            n: Number of points
            base: Base for sequence

        Returns:
            Array of Halton sequence values
        """
        sequence = np.zeros(n)

        for i in range(n):
            f = 1.0
            r = 0.0
            j = i + 1

            while j > 0:
                f = f / base
                r = r + f * (j % base)
                j = j // base

            sequence[i] = r

        return sequence

    @staticmethod
    def sobol_sequence(n: int, dim: int = 1) -> np.ndarray:
        """
        Generate Sobol sequence (requires scipy >= 1.7.0).

        Args:
            n: Number of points
            dim: Dimension

        Returns:
            Array of shape (n, dim)
        """
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=dim, scramble=True)
        return sampler.random(n)

    def integrate_qmc(self,
                     func: Callable,
                     bounds: List[Tuple[float, float]],
                     num_points: int = 10000) -> float:
        """
        Quasi-Monte Carlo integration.

        Args:
            func: Function to integrate
            bounds: Integration bounds for each dimension
            num_points: Number of QMC points

        Returns:
            Integral estimate
        """
        dim = len(bounds)

        # Generate Sobol sequence
        qmc_points = self.sobol_sequence(num_points, dim)

        # Transform to integration bounds
        for i, (lower, upper) in enumerate(bounds):
            qmc_points[:, i] = lower + (upper - lower) * qmc_points[:, i]

        # Evaluate function
        values = np.array([func(*point) for point in qmc_points])

        # Calculate volume
        volume = np.prod([upper - lower for lower, upper in bounds])

        return volume * np.mean(values)
