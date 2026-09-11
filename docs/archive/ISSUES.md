# Issues

## Round 1: Initial Audit (69 test failures, 40% coverage)
See [IS-001] through [IS-015] for code bugs and API mismatches.

## Round 2: Production Readiness Gap Analysis

### Infrastructure Gaps (Before Round 2)

| # | Severity | Area | Gap |
|---|----------|------|-----|
| IS-016 | HIGH | Observability | No structured logging - only `print()` and `st.error()` |
| IS-017 | HIGH | Observability | No request/correlation IDs for debugging |
| IS-018 | HIGH | Observability | No performance timing for expensive operations |
| IS-019 | HIGH | Test Coverage | 0% coverage on `statistical_tests/` (3 modules, 175 lines) |
| IS-020 | HIGH | Test Coverage | 0% coverage on `utils/validation.py` (73 lines) |
| IS-021 | HIGH | Test Coverage | 0% coverage on `utils/data_preprocessing.py` (128 lines) |
| IS-022 | HIGH | Test Coverage | 0% coverage on `utils/plotting.py` (134 lines) |
| IS-023 | HIGH | Deployment Safety | No Dockerfile - can't containerize |
| IS-024 | HIGH | Deployment Safety | No CI/CD pipeline - no automated testing |
| IS-025 | HIGH | Deployment Safety | No health check endpoints |
| IS-026 | HIGH | Operational Readiness | No dependency security scanning |
| IS-027 | MEDIUM | Security | `unsafe_allow_html=True` in Streamlit app |
| IS-028 | MEDIUM | Maintainability | `sys.path.insert()` hack for imports |
| IS-029 | MEDIUM | Performance | No caching for expensive computations |
| IS-030 | MEDIUM | Reliability | No error boundary in web app (uncaught exceptions crash UI) |
| IS-031 | MEDIUM | Reliability | No loading states for heavy operations |
| IS-032 | LOW | Deployment Safety | No `.dockerignore` |
| IS-033 | LOW | Deployment Safety | No `pyproject.toml` for modern tooling |

### Edge Case Gaps (Found in Round 2)

| # | Severity | Area | Description |
|---|----------|------|-------------|
| IS-034 | MEDIUM | Mixtures | `BayesianGMM` didn't converge with default 100 iters |
| IS-035 | MEDIUM | Fitting | `anderson_darling_test` used deprecated scipy API |
| IS-036 | MEDIUM | Utils | `standardize()` didn't handle single-value arrays (NaN std) |

---

## Summary (All Rounds)

| Severity | Round 1 | Round 2 | Total | Fixed |
|----------|---------|---------|-------|-------|
| CRITICAL | 1 | 0 | 1 | 1 |
| HIGH | 6 | 11 | 17 | 17 |
| MEDIUM | 6 | 6 | 12 | 12 |
| LOW | 3 | 2 | 5 | 5 |
| **Total** | **16** | **19** | **35** | **35** |