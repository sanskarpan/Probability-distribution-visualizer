"""
Probability Distribution Visualizer - Streamlit Web App

An interactive web application for visualizing and exploring probability distributions.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.distributions import (
    NormalDistribution,
    ExponentialDistribution,
    UniformDistribution,
    BetaDistribution,
    GammaDistribution,
    ChiSquareDistribution,
    StudentTDistribution,
    WeibullDistribution,
    LognormalDistribution,
    CauchyDistribution,
    BinomialDistribution,
    PoissonDistribution,
    GeometricDistribution,
    NegativeBinomialDistribution,
    HypergeometricDistribution,
    DiscreteUniformDistribution,
)

from src.utils.logger import get_logger, set_correlation_id, log_error

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Probability Distribution Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Header
st.markdown("# 📊 Probability Distribution Visualizer")
st.caption("Interactive exploration of statistical distributions")


# Distribution registry
DISTRIBUTIONS = {
    "Continuous": {
        "Normal (Gaussian)": NormalDistribution,
        "Exponential": ExponentialDistribution,
        "Uniform": UniformDistribution,
        "Beta": BetaDistribution,
        "Gamma": GammaDistribution,
        "Chi-Square": ChiSquareDistribution,
        "Student-t": StudentTDistribution,
        "Weibull": WeibullDistribution,
        "Lognormal": LognormalDistribution,
        "Cauchy": CauchyDistribution,
    },
    "Discrete": {
        "Binomial": BinomialDistribution,
        "Poisson": PoissonDistribution,
        "Geometric": GeometricDistribution,
        "Negative Binomial": NegativeBinomialDistribution,
        "Hypergeometric": HypergeometricDistribution,
        "Discrete Uniform": DiscreteUniformDistribution,
    }
}


def create_parameter_inputs(dist_class, dist_name):
    """Create input widgets for distribution parameters."""
    try:
        dist = dist_class()
        params = dist.get_parameters()
        bounds = dist.get_parameter_bounds()

        new_params = {}
        for param_name, param_value in params.items():
            min_val, max_val = bounds.get(param_name, (0.0, 10.0))

            if isinstance(param_value, int) and param_name not in ['n', 'M', 'N', 'r']:
                new_params[param_name] = st.sidebar.slider(
                    f"{param_name}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(param_value),
                    step=0.1,
                    key=f"{dist_name}_{param_name}"
                )
            elif isinstance(param_value, int):
                new_params[param_name] = int(st.sidebar.slider(
                    f"{param_name}",
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(param_value),
                    step=1,
                    key=f"{dist_name}_{param_name}"
                ))
            else:
                new_params[param_name] = st.sidebar.slider(
                    f"{param_name}",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(param_value),
                    step=0.01,
                    key=f"{dist_name}_{param_name}"
                )

        return new_params
    except Exception as e:
        log_error(logger, f"Error creating parameter inputs for '{dist_name}'", exc=e)
        st.error(f"Error creating parameter inputs: {e}")
        return {}


@st.cache_data(show_spinner=False)
def _cached_plot_pdf_pmf(dist_class_name, dist_type, params_json, num_points=1000):
    """Cached PDF/PMF plot generation."""
    import json
    params = json.loads(params_json)
    dist_class = DISTRIBUTIONS[dist_type][dist_class_name]
    dist = dist_class(**params)
    return _build_pdf_pmf_plot(dist, num_points)


def _build_pdf_pmf_plot(dist, num_points=1000):
    """Build PDF/PMF plot figure."""
    try:
        if dist.is_discrete:
            support = dist.get_support()
            if support[1] == np.inf:
                x_max = min(int(dist.mean() + 5 * dist.std()), 100)
                x = np.arange(max(0, support[0]), x_max)
            else:
                x = np.arange(int(support[0]), min(int(support[1]) + 1, 100))

            y = dist.pdf(x)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=x,
                y=y,
                name='PMF',
                marker=dict(
                    color=y,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Probability"),
                ),
                hovertemplate='<b>x</b>: %{x}<br><b>P(X=x)</b>: %{y:.4f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"{dist.name} Distribution - PMF",
                xaxis_title="x",
                yaxis_title="Probability Mass Function P(X=x)",
                hovermode='closest',
                template='plotly_white',
                height=500
            )

        else:
            support = dist.get_support()
            if support[0] == -np.inf:
                x_min = dist.mean() - 4 * dist.std()
            else:
                x_min = support[0]

            if support[1] == np.inf:
                x_max = dist.mean() + 4 * dist.std()
            else:
                x_max = support[1]

            x = np.linspace(x_min, x_max, num_points)
            y = dist.pdf(x)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='PDF',
                fill='tozeroy',
                line=dict(color='#667eea', width=3),
                fillcolor='rgba(102, 126, 234, 0.2)',
                hovertemplate='<b>x</b>: %{x:.4f}<br><b>f(x)</b>: %{y:.4f}<extra></extra>'
            ))

            mean_val = dist.mean()
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_val:.3f}",
                annotation_position="top"
            )

            fig.update_layout(
                title=f"{dist.name} Distribution - PDF",
                xaxis_title="x",
                yaxis_title="Probability Density Function f(x)",
                hovermode='closest',
                template='plotly_white',
                height=500
            )

        return fig
    except Exception as e:
        log_error(logger, f"Error plotting PDF/PMF for '{dist.name}'", exc=e)
        raise


def plot_pdf_pmf(dist, dist_class_name, dist_type, num_points=1000):
    """Plot PDF or PMF of the distribution with caching."""
    import json
    params = dist.get_parameters()
    params_json = json.dumps(params, sort_keys=True, default=str)
    try:
        return _cached_plot_pdf_pmf(dist_class_name, dist_type, params_json, num_points)
    except Exception as e:
        st.error(f"Error plotting PDF/PMF: {e}")
        return None


@st.cache_data(show_spinner=False)
def _cached_plot_cdf(dist_class_name, dist_type, params_json, num_points=1000):
    """Cached CDF plot generation."""
    import json
    params = json.loads(params_json)
    dist_class = DISTRIBUTIONS[dist_type][dist_class_name]
    dist = dist_class(**params)
    return _build_cdf_plot(dist, num_points)


def _build_cdf_plot(dist, num_points=1000):
    """Build CDF plot figure."""
    try:
        if dist.is_discrete:
            support = dist.get_support()
            if support[1] == np.inf:
                x_max = min(int(dist.mean() + 5 * dist.std()), 100)
                x = np.arange(max(0, support[0]), x_max)
            else:
                x = np.arange(int(support[0]), min(int(support[1]) + 1, 100))

            y = dist.cdf(x)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name='CDF',
                line=dict(color='#764ba2', width=3),
                marker=dict(size=6),
                hovertemplate='<b>x</b>: %{x}<br><b>F(x)</b>: %{y:.4f}<extra></extra>'
            ))
        else:
            support = dist.get_support()
            if support[0] == -np.inf:
                x_min = dist.mean() - 4 * dist.std()
            else:
                x_min = support[0]

            if support[1] == np.inf:
                x_max = dist.mean() + 4 * dist.std()
            else:
                x_max = support[1]

            x = np.linspace(x_min, x_max, num_points)
            y = dist.cdf(x)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='CDF',
                line=dict(color='#764ba2', width=3),
                hovertemplate='<b>x</b>: %{x:.4f}<br><b>F(x)</b>: %{y:.4f}<extra></extra>'
            ))

            median_val = dist.median()
            fig.add_vline(
                x=median_val,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Median: {median_val:.3f}",
                annotation_position="top"
            )

        fig.update_layout(
            title=f"{dist.name} Distribution - CDF",
            xaxis_title="x",
            yaxis_title="Cumulative Distribution Function F(x)",
            hovermode='closest',
            template='plotly_white',
            height=500
        )

        return fig
    except Exception as e:
        log_error(logger, f"Error plotting CDF for '{dist.name}'", exc=e)
        raise


def plot_cdf(dist, dist_class_name, dist_type, num_points=1000):
    """Plot CDF of the distribution with caching."""
    import json
    params = dist.get_parameters()
    params_json = json.dumps(params, sort_keys=True, default=str)
    try:
        return _cached_plot_cdf(dist_class_name, dist_type, params_json, num_points)
    except Exception as e:
        st.error(f"Error plotting CDF: {e}")
        return None


@st.cache_data(show_spinner=False)
def _cached_get_statistics(dist_class_name, dist_type, params_json):
    """Cached statistics computation."""
    import json
    params = json.loads(params_json)
    dist_class = DISTRIBUTIONS[dist_type][dist_class_name]
    dist = dist_class(**params)
    return dist.get_statistics()


def get_statistics(dist, dist_class_name, dist_type):
    """Get distribution statistics with caching."""
    import json
    params = dist.get_parameters()
    params_json = json.dumps(params, sort_keys=True, default=str)
    try:
        return _cached_get_statistics(dist_class_name, dist_type, params_json)
    except Exception as e:
        log_error(logger, f"Error computing statistics for '{dist.name}'", exc=e)
        raise


def display_statistics(dist, dist_class_name, dist_type):
    """Display distribution statistics in cards."""
    try:
        stats = get_statistics(dist, dist_class_name, dist_type)

        def _safe_format(value, default="N/A"):
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                return default
            return f"{value:.4f}"

        cols = st.columns(4)

        with cols[0]:
            st.metric("Mean", _safe_format(stats.get('mean'), 'N/A'))

        with cols[1]:
            st.metric("Variance", _safe_format(stats.get('variance'), 'N/A'))

        with cols[2]:
            st.metric("Std Dev", _safe_format(stats.get('std_dev'), 'N/A'))

        with cols[3]:
            st.metric("Median", _safe_format(stats.get('median'), 'N/A'))

        st.markdown("### Additional Statistics")
        cols2 = st.columns(3)

        with cols2[0]:
            mode_val = stats.get('mode')
            st.metric("Mode", _safe_format(mode_val, "N/A"))

        with cols2[1]:
            skew_val = stats.get('skewness')
            st.metric("Skewness", _safe_format(skew_val, "N/A"))

        with cols2[2]:
            kurt_val = stats.get('kurtosis')
            st.metric("Excess Kurtosis", _safe_format(kurt_val, "N/A"))

    except Exception as e:
        log_error(logger, "Error displaying statistics", exc=e)
        st.error(f"Error displaying statistics: {e}")


def main():
    """Main application logic with error boundary."""
    set_correlation_id()

    try:
        _main_impl()
    except Exception as e:
        log_error(logger, "Unhandled exception in main()", exc=e)
        st.error("⚠️ An unexpected error occurred. Please refresh the page or try adjusting your parameters.")
        st.error(f"Details: {e}")


def _main_impl():
    """Internal main implementation."""
    logger.info("Page view - Distribution Visualizer loaded")

    st.sidebar.title("Distribution Settings")

    dist_type = st.sidebar.radio("Distribution Type", ["Continuous", "Discrete"])

    available_dists = list(DISTRIBUTIONS[dist_type].keys())
    selected_dist_name = st.sidebar.selectbox(
        "Select Distribution",
        available_dists,
        key="dist_select"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Parameters")

    dist_class = DISTRIBUTIONS[dist_type][selected_dist_name]

    params = create_parameter_inputs(dist_class, selected_dist_name)

    logger.info(
        "Distribution selected: %s (%s), params=%s",
        selected_dist_name,
        dist_type,
        params,
    )

    try:
        dist = dist_class(**params)
    except Exception as e:
        log_error(
            logger,
            f"Error creating distribution '{selected_dist_name}'",
            exc=e,
            extra={"distribution": selected_dist_name, "type": dist_type, "params": str(params)},
        )
        st.error(f"Error creating distribution: {e}")
        return

    st.sidebar.markdown("---")

    st.sidebar.markdown("### Visualization Options")
    show_pdf = st.sidebar.checkbox("Show PDF/PMF", value=True)
    show_cdf = st.sidebar.checkbox("Show CDF", value=True)
    show_samples = st.sidebar.checkbox("Show Random Samples", value=False)

    if show_samples:
        num_samples = st.sidebar.slider("Number of Samples", 100, 10000, 1000, step=100)
        use_fixed_seed = st.sidebar.checkbox("Fix random seed", value=True)
        if use_fixed_seed:
            random_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    st.markdown(f"## {selected_dist_name} Distribution")

    with st.expander("ℹ️ Current Parameters", expanded=False):
        if len(params) > 0:
            param_cols = st.columns(len(params))
            for i, (param_name, param_value) in enumerate(params.items()):
                with param_cols[i]:
                    st.metric(param_name, f"{param_value}")
        else:
            st.write("No parameters to display")

    st.markdown("### Distribution Statistics")
    with st.spinner("Computing statistics..."):
        display_statistics(dist, selected_dist_name, dist_type)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if show_pdf:
            with st.spinner("Generating PDF/PMF plot..."):
                fig_pdf = plot_pdf_pmf(dist, selected_dist_name, dist_type)
            if fig_pdf:
                st.plotly_chart(fig_pdf, use_container_width=True)

    with col2:
        if show_cdf:
            with st.spinner("Generating CDF plot..."):
                fig_cdf = plot_cdf(dist, selected_dist_name, dist_type)
            if fig_cdf:
                st.plotly_chart(fig_cdf, use_container_width=True)

    if show_samples:
        st.markdown("### Random Samples")

        with st.spinner("Generating random samples..."):
            try:
                if use_fixed_seed:
                    samples = dist.rvs(size=num_samples, random_state=random_seed)
                else:
                    samples = dist.rvs(size=num_samples)

                fig_samples = go.Figure()

                if dist.is_discrete:
                    unique, counts = np.unique(samples, return_counts=True)
                    fig_samples.add_trace(go.Bar(
                        x=unique,
                        y=counts / num_samples,
                        name='Sample Distribution',
                        marker=dict(color='rgba(102, 126, 234, 0.7)'),
                    ))
                else:
                    fig_samples.add_trace(go.Histogram(
                        x=samples,
                        name='Sample Distribution',
                        nbinsx=50,
                        histnorm='probability density',
                        marker=dict(color='rgba(102, 126, 234, 0.7)'),
                    ))

                    x_theory = np.linspace(samples.min(), samples.max(), 500)
                    y_theory = dist.pdf(x_theory)
                    fig_samples.add_trace(go.Scatter(
                        x=x_theory,
                        y=y_theory,
                        mode='lines',
                        name='Theoretical PDF',
                        line=dict(color='red', width=3),
                    ))

                fig_samples.update_layout(
                    title=f"Random Samples (n={num_samples})",
                    xaxis_title="Value",
                    yaxis_title="Density/Probability",
                    template='plotly_white',
                    height=500,
                    showlegend=True
                )

                st.plotly_chart(fig_samples, use_container_width=True)

                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("Sample Mean", f"{np.mean(samples):.4f}")
                with col_s2:
                    st.metric("Sample Std", f"{np.std(samples):.4f}")
                with col_s3:
                    st.metric("Sample Min", f"{np.min(samples):.4f}")
                with col_s4:
                    st.metric("Sample Max", f"{np.max(samples):.4f}")

            except Exception as e:
                log_error(logger, f"Error generating samples for '{selected_dist_name}'", exc=e)
                st.error(f"Error generating samples: {e}")

    with st.expander("📊 Quantiles", expanded=False):
        st.markdown("### Percentiles")

        percentiles = [5, 25, 50, 75, 95]
        quantile_values = []
        for p in percentiles:
            try:
                q_val = dist.ppf(p / 100)
                if np.isinf(q_val) or np.isnan(q_val):
                    q_val = float('nan')
                quantile_values.append(q_val)
            except Exception:
                quantile_values.append(float('nan'))

        quant_cols = st.columns(len(percentiles))
        for i, (p, q) in enumerate(zip(percentiles, quantile_values)):
            with quant_cols[i]:
                if np.isnan(q):
                    st.metric(f"{p}th percentile", "N/A")
                else:
                    st.metric(f"{p}th percentile", f"{q:.4f}")


if __name__ == "__main__":
    main()