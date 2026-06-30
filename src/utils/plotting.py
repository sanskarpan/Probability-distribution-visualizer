"""Plotting utilities for probability distributions."""

from typing import Optional, List, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_distribution_comparison(
    data: np.ndarray,
    distributions: List[str],
    fitted_params: Optional[dict] = None,
    bins: int = 30,
    figsize: Tuple[float, float] = (12, 6)
) -> plt.Figure:
    """
    Compare empirical data with fitted distributions.

    Args:
        data: Empirical data
        distributions: List of distribution names
        fitted_params: Dictionary of fitted parameters for each distribution
        bins: Number of histogram bins
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Histogram
    ax1.hist(data, bins=bins, density=True, alpha=0.6, color='gray', label='Data')

    # Plot fitted distributions
    x = np.linspace(np.min(data), np.max(data), 1000)

    for dist_name in distributions:
        dist = getattr(stats, dist_name)

        if fitted_params and dist_name in fitted_params:
            params = fitted_params[dist_name]
        else:
            params = dist.fit(data)

        pdf = dist.pdf(x, *params)
        ax1.plot(x, pdf, label=f'{dist_name}', linewidth=2)

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # CDF comparison
    empirical_cdf = np.arange(1, len(sorted(data)) + 1) / len(data)
    ax2.plot(sorted(data), empirical_cdf, 'o', markersize=2, alpha=0.6, label='Empirical')

    for dist_name in distributions:
        dist = getattr(stats, dist_name)

        if fitted_params and dist_name in fitted_params:
            params = fitted_params[dist_name]
        else:
            params = dist.fit(data)

        cdf = dist.cdf(x, *params)
        ax2.plot(x, cdf, label=f'{dist_name}', linewidth=2)

    ax2.set_xlabel('Value')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('CDF Comparison')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_qq(
    data: np.ndarray,
    dist: str = 'norm',
    params: Optional[tuple] = None,
    figsize: Tuple[float, float] = (8, 8)
) -> plt.Figure:
    """
    Create Q-Q plot for distribution fit assessment.

    Args:
        data: Empirical data
        dist: Distribution name
        params: Distribution parameters (fitted if None)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    distribution = getattr(stats, dist)

    if params is None:
        params = distribution.fit(data)

    fig, ax = plt.subplots(figsize=figsize)

    # Theoretical quantiles
    sorted_data = np.sort(data)
    n = len(data)
    theoretical_quantiles = distribution.ppf(np.linspace(0.01, 0.99, n), *params)

    # Q-Q plot
    ax.scatter(theoretical_quantiles, sorted_data, alpha=0.6, s=20)

    # Reference line
    min_val = min(np.min(theoretical_quantiles), np.min(sorted_data))
    max_val = max(np.max(theoretical_quantiles), np.max(sorted_data))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')

    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(f'Q-Q Plot ({dist} distribution)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_histogram_with_fit(
    data: np.ndarray,
    dist: str = 'norm',
    params: Optional[tuple] = None,
    bins: int = 30,
    figsize: Tuple[float, float] = (10, 6)
) -> plt.Figure:
    """
    Plot histogram with fitted distribution overlay.

    Args:
        data: Empirical data
        dist: Distribution name
        params: Distribution parameters (fitted if None)
        bins: Number of histogram bins
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    distribution = getattr(stats, dist)

    if params is None:
        params = distribution.fit(data)

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    n, bins_edges, patches = ax.hist(
        data, bins=bins, density=True,
        alpha=0.6, color='skyblue', edgecolor='black',
        label='Data'
    )

    # Fitted distribution
    x = np.linspace(np.min(data), np.max(data), 1000)
    pdf = distribution.pdf(x, *params)
    ax.plot(x, pdf, 'r-', linewidth=2, label=f'Fitted {dist}')

    # Statistics text
    param_str = ', '.join([f'{p:.3f}' for p in params])
    stats_text = f'Parameters: {param_str}\n'
    stats_text += f'Mean: {np.mean(data):.3f}\n'
    stats_text += f'Std: {np.std(data):.3f}'

    ax.text(0.95, 0.95, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f'Histogram with Fitted {dist.capitalize()} Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    data: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = 'pearson',
    figsize: Tuple[float, float] = (10, 8),
    annot: bool = True,
    cmap: str = 'coolwarm'
) -> plt.Figure:
    """
    Create correlation matrix heatmap.

    Args:
        data: 2D array where each column is a variable
        labels: Variable names
        method: 'pearson', 'spearman', or 'kendall'
        figsize: Figure size
        annot: Whether to annotate cells with values
        cmap: Colormap name

    Returns:
        Matplotlib figure
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n_vars = data.shape[1]

    # Compute correlation matrix
    if method == 'pearson':
        corr_matrix = np.corrcoef(data.T)
    else:
        corr_matrix = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    if method == 'spearman':
                        corr, _ = stats.spearmanr(data[:, i], data[:, j])
                    elif method == 'kendall':
                        corr, _ = stats.kendalltau(data[:, i], data[:, j])
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    corr_matrix[i, j] = corr

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    # Labels
    if labels is None:
        labels = [f'Var {i+1}' for i in range(n_vars)]

    ax.set_xticks(np.arange(n_vars))
    ax.set_yticks(np.arange(n_vars))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)

    # Annotations
    if annot:
        for i in range(n_vars):
            for j in range(n_vars):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha='center', va='center',
                             color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black',
                             fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20)

    ax.set_title(f'{method.capitalize()} Correlation Matrix')

    plt.tight_layout()
    return fig


def plot_probability_bands(
    data: np.ndarray,
    dist: str = 'norm',
    params: Optional[tuple] = None,
    confidence_levels: List[float] = [0.68, 0.95, 0.997],
    figsize: Tuple[float, float] = (12, 6)
) -> plt.Figure:
    """
    Plot data with probability bands.

    Args:
        data: Time series or sequential data
        dist: Distribution name
        params: Distribution parameters
        confidence_levels: List of confidence levels
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    distribution = getattr(stats, dist)

    if params is None:
        params = distribution.fit(data)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot data
    x = np.arange(len(data))
    ax.plot(x, data, 'k-', linewidth=1.5, label='Data', zorder=5)

    # Mean line
    mean = distribution.mean(*params)
    ax.axhline(y=mean, color='red', linestyle='--', linewidth=2, label='Mean', zorder=4)

    # Probability bands
    colors = ['lightblue', 'lightgreen', 'lightyellow']

    for level, color in zip(confidence_levels, colors):
        alpha = (1 - level) / 2
        lower = distribution.ppf(alpha, *params)
        upper = distribution.ppf(1 - alpha, *params)

        ax.fill_between(x, lower, upper, alpha=0.3, color=color,
                        label=f'{level*100:.1f}% CI', zorder=1)

    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.set_title('Data with Probability Bands')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig
