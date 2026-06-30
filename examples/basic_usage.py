"""Basic usage examples for probability distributions."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from distributions import NormalDistribution, ExponentialDistribution, BinomialDistribution


def example_normal_distribution():
    """Demonstrate Normal distribution."""
    print("="*60)
    print("Normal Distribution Example")
    print("="*60)

    # Create a standard normal distribution
    normal = NormalDistribution(mu=0, sigma=1)

    # Display statistics
    print(f"\nDistribution: {normal}")
    print(f"Mean: {normal.mean():.4f}")
    print(f"Variance: {normal.var():.4f}")
    print(f"Std Dev: {normal.std():.4f}")
    print(f"Median: {normal.median():.4f}")

    # Calculate probabilities
    x = np.linspace(-4, 4, 1000)
    pdf = normal.pdf(x)
    cdf = normal.cdf(x)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x, pdf, 'b-', linewidth=2)
    ax1.fill_between(x, pdf, alpha=0.3)
    ax1.axvline(normal.mean(), color='r', linestyle='--', label='Mean')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Normal Distribution PDF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, cdf, 'g-', linewidth=2)
    ax2.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(normal.median(), color='r', linestyle='--', label='Median')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Normal Distribution CDF')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('normal_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'normal_distribution.png'")


def example_exponential_distribution():
    """Demonstrate Exponential distribution."""
    print("\n" + "="*60)
    print("Exponential Distribution Example")
    print("="*60)

    # Create exponential distribution
    exp_dist = ExponentialDistribution(lambda_param=0.5)

    # Display statistics
    print(f"\nDistribution: {exp_dist}")
    print(f"Mean: {exp_dist.mean():.4f}")
    print(f"Variance: {exp_dist.var():.4f}")

    # Generate and plot samples
    samples = exp_dist.rvs(size=10000, random_state=42)

    plt.figure(figsize=(12, 5))

    # Histogram of samples
    plt.subplot(1, 2, 1)
    plt.hist(samples, bins=50, density=True, alpha=0.7, edgecolor='black')
    x = np.linspace(0, np.percentile(samples, 99), 500)
    plt.plot(x, exp_dist.pdf(x), 'r-', linewidth=2, label='Theoretical PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Exponential Distribution - Histogram vs PDF')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # CDF
    plt.subplot(1, 2, 2)
    x = np.linspace(0, 10, 500)
    plt.plot(x, exp_dist.cdf(x), 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Cumulative Probability')
    plt.title('Exponential Distribution CDF')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exponential_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'exponential_distribution.png'")


def example_binomial_distribution():
    """Demonstrate Binomial distribution."""
    print("\n" + "="*60)
    print("Binomial Distribution Example")
    print("="*60)

    # Create binomial distribution
    binomial = BinomialDistribution(n=20, p=0.3)

    # Display statistics
    print(f"\nDistribution: {binomial}")
    print(f"Mean: {binomial.mean():.4f}")
    print(f"Variance: {binomial.var():.4f}")

    # Calculate PMF
    x = np.arange(0, 21)
    pmf = binomial.pdf(x)

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(x, pmf, alpha=0.7, edgecolor='black')
    plt.axvline(binomial.mean(), color='r', linestyle='--', linewidth=2, label='Mean')
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.title(f'Binomial Distribution PMF (n={binomial.n}, p={binomial.p})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    cdf = binomial.cdf(x)
    plt.step(x, cdf, where='post', linewidth=2)
    plt.xlabel('Number of Successes')
    plt.ylabel('Cumulative Probability')
    plt.title('Binomial Distribution CDF')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('binomial_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'binomial_distribution.png'")


def example_comparison():
    """Compare multiple distributions."""
    print("\n" + "="*60)
    print("Distribution Comparison Example")
    print("="*60)

    # Create multiple normal distributions
    dists = [
        NormalDistribution(mu=0, sigma=1),
        NormalDistribution(mu=2, sigma=1.5),
        NormalDistribution(mu=-1, sigma=0.5),
    ]

    colors = ['blue', 'red', 'green']
    labels = ['N(0, 1)', 'N(2, 1.5)', 'N(-1, 0.5)']

    # Plot
    plt.figure(figsize=(12, 6))

    x = np.linspace(-5, 6, 1000)

    for dist, color, label in zip(dists, colors, labels):
        pdf = dist.pdf(x)
        plt.plot(x, pdf, color=color, linewidth=2, label=label)
        plt.axvline(dist.mean(), color=color, linestyle='--', alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Comparison of Normal Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'distribution_comparison.png'")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("PROBABILITY DISTRIBUTION VISUALIZER - EXAMPLES")
    print("="*60)

    try:
        example_normal_distribution()
        example_exponential_distribution()
        example_binomial_distribution()
        example_comparison()

        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        print("\nGenerated plots:")
        print("  - normal_distribution.png")
        print("  - exponential_distribution.png")
        print("  - binomial_distribution.png")
        print("  - distribution_comparison.png")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
