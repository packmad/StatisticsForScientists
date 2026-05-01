from __future__ import annotations

import math
import warnings
from typing import Any, Literal, Callable, Optional, Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import (
    ConstantInputWarning,
    t,
    norm,
    levene,
    shapiro,
    anderson,
    kurtosis,
    pearsonr,
    spearmanr,
    mannwhitneyu,
)

ArrayLike1D = Sequence[float] | np.ndarray
Alternative = Literal["two-sided", "less", "greater"]
CorrelationMethod = Literal["pearson", "spearman"]
ComparisonEstimand = Literal["mean_difference", "stochastic_dominance"]


@dataclass(frozen=True)
class DescriptiveStats:
    n: int
    mean: float
    sd: float
    median: float
    minimum: float
    maximum: float
    kurtosis_fisher: float
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssumptionCheck:
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    passed: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceInterval:
    level: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectSize:
    name: str
    value: float
    interpretation: Optional[str] = None
    ci: Optional[ConfidenceInterval] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TwoGroupComparisonResult:
    estimand: ComparisonEstimand
    method: str
    alternative: Alternative
    statistic: float
    p_value: float
    estimate: float
    estimate_label: str
    ci: Optional[ConfidenceInterval]
    effect_size: Optional[EffectSize]
    n1: int
    n2: int
    group1_descriptives: DescriptiveStats
    group2_descriptives: DescriptiveStats
    df: Optional[float] = None
    assumptions: tuple[AssumptionCheck, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["assumptions"] = [a.to_dict() for a in self.assumptions]
        if self.ci is not None:
            out["ci"] = self.ci.to_dict()
        if self.effect_size is not None:
            out["effect_size"] = self.effect_size.to_dict()
        out["group1_descriptives"] = self.group1_descriptives.to_dict()
        out["group2_descriptives"] = self.group2_descriptives.to_dict()
        return out

    def summary(self, digits: int = 3) -> str:
        p = f"{self.p_value:.{digits}g}"
        est = f"{self.estimate:.{digits}f}"
        stat = f"{self.statistic:.{digits}f}"
        parts = [
            f"{self.method}: {self.estimate_label}={est}",
            f"statistic={stat}",
        ]
        if self.df is not None:
            parts.append(f"df={self.df:.{digits}f}")
        parts.append(f"p={p}")
        if self.ci is not None:
            parts.append(f"{int(self.ci.level * 100)}% CI [{self.ci.lower:.{digits}f}, {self.ci.upper:.{digits}f}]")
        if self.effect_size is not None:
            parts.append(f"{self.effect_size.name}={self.effect_size.value:.{digits}f}")
        parts.append(f"group1_n={self.group1_descriptives.n}")
        parts.append(f"group2_n={self.group2_descriptives.n}")
        return "; ".join(parts)


@dataclass(frozen=True)
class CorrelationResult:
    method: CorrelationMethod
    alternative: Alternative
    coefficient: float
    p_value: float
    n: int
    ci: Optional[ConfidenceInterval]
    x_descriptives: DescriptiveStats
    y_descriptives: DescriptiveStats
    assumptions: tuple[AssumptionCheck, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["assumptions"] = [a.to_dict() for a in self.assumptions]
        if self.ci is not None:
            out["ci"] = self.ci.to_dict()
        out["x_descriptives"] = self.x_descriptives.to_dict()
        out["y_descriptives"] = self.y_descriptives.to_dict()
        return out

    def summary(self, digits: int = 3) -> str:
        p = f"{self.p_value:.{digits}g}"
        r = f"{self.coefficient:.{digits}f}"
        parts = [f"{self.method} correlation: r={r}", f"p={p}", f"n={self.n}"]
        if self.ci is not None:
            parts.append(f"{int(self.ci.level * 100)}% CI [{self.ci.lower:.{digits}f}, {self.ci.upper:.{digits}f}]")
        parts.append(f"x_mean={self.x_descriptives.mean:.{digits}f}")
        parts.append(f"y_mean={self.y_descriptives.mean:.{digits}f}")
        return "; ".join(parts)


# ------------------------------
# Validation and descriptives
# ------------------------------


def _as_1d_float_array(data: ArrayLike1D, *, name: str = "data", allow_nan: bool = False) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape={x.shape}.")
    if x.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not allow_nan and np.isnan(x).any():
        raise ValueError(f"{name} contains NaN values. Impute or remove them explicitly before analysis.")
    if np.isinf(x).any():
        raise ValueError(f"{name} contains infinite values.")
    return x


def _require_variation(x: np.ndarray, *, name: str) -> None:
    if np.allclose(x, x[0]):
        raise ValueError(f"{name} has zero variance; the requested analysis is undefined.")


def describe(data: ArrayLike1D) -> DescriptiveStats:
    x = _as_1d_float_array(data, name="data")
    if x.size < 2:
        raise ValueError("At least 2 observations are required for descriptive statistics with sample SD.")
    notes: list[str] = []
    # SciPy's kurtosis is numerically unstable for constant or nearly constant arrays.
    # We avoid surfacing SciPy warnings by returning NaN with an explicit note instead.
    ptp = float(np.ptp(x))
    scale = max(1.0, float(abs(np.mean(x))))
    near_constant_tol = 1e-12 * scale
    if ptp <= near_constant_tol:
        kurt = float("nan")
        notes.append("Kurtosis is undefined or numerically unstable for constant/nearly-constant data; returning NaN.")
    else:
        kurt = float(kurtosis(x, fisher=True, bias=False, nan_policy="raise"))
    return DescriptiveStats(
        n=int(x.size),
        mean=float(np.mean(x)),
        sd=float(np.std(x, ddof=1)),
        median=float(np.median(x)),
        minimum=float(np.min(x)),
        maximum=float(np.max(x)),
        kurtosis_fisher=kurt,
        notes=tuple(notes),
    )


# ------------------------------
# Diagnostics (not automatic gatekeepers)
# ------------------------------


def shapiro_normality(data: ArrayLike1D, alpha: float = 0.05) -> AssumptionCheck:
    x = _as_1d_float_array(data, name="data")
    n = x.size
    if n < 3:
        raise ValueError(f"Shapiro-Wilk requires at least 3 observations, got {n}.")
    statistic, p_value = shapiro(x)
    note = (
        "Diagnostic only: normality tests should not be the sole gatekeeper for parametric inference. "
        "Interpret alongside Q-Q plots, sample size, and substantive robustness considerations."
    )
    if n > 5000:
        note += " SciPy warns that p-values may be inaccurate for n > 5000."
    return AssumptionCheck(
        test_name="Shapiro-Wilk",
        statistic=float(statistic),
        p_value=float(p_value),
        alpha=alpha,
        passed=bool(p_value >= alpha),
        note=note,
    )


def equal_variance_check(
    group1: ArrayLike1D,
    group2: ArrayLike1D,
    *,
    alpha: float = 0.05,
    center: Literal["mean", "median"] = "median",
) -> AssumptionCheck:
    x = _as_1d_float_array(group1, name="group1")
    y = _as_1d_float_array(group2, name="group2")
    statistic, p_value = levene(x, y, center=center)
    note = (
        f"Levene/Brown-Forsythe test with center='{center}'. "
        "Use as a diagnostic; Welch's t-test is typically preferred when comparing means because it does not assume equal variances."
    )
    return AssumptionCheck(
        test_name="Levene",
        statistic=float(statistic),
        p_value=float(p_value),
        alpha=alpha,
        passed=bool(p_value >= alpha),
        note=note,
    )


def anderson_darling_candidates(data: ArrayLike1D) -> list[str]:
    x = _as_1d_float_array(data, name="data")
    dists = ["norm", "expon", "logistic", "gumbel_l", "gumbel_r"]
    accepted: list[str] = []
    for dist in dists:
        # SciPy 1.17+ requires choosing a p-value calculation method explicitly.
        # 'interpolate' preserves the historical, deterministic table-based behavior.
        result = anderson(x, dist=dist, method="interpolate")
        # Historically we compared against the 5% critical value (index 2).
        # With an explicit method, SciPy returns a p-value instead, so we keep
        # the same decision rule: accept if we fail to reject at alpha=0.05.
        if float(result.pvalue) >= 0.05:
            accepted.append(dist)
    return accepted


# ------------------------------
# Effect sizes
# ------------------------------


def interpret_hedges_g(value: float) -> str:
    d = abs(value)
    if d < 0.01:
        return "negligible"
    if d < 0.20:
        return "very small"
    if d < 0.50:
        return "small"
    if d < 0.80:
        return "medium"
    if d < 1.20:
        return "large"
    if d < 2.00:
        return "very large"
    return "huge"


def hedges_g(group1: ArrayLike1D, group2: ArrayLike1D) -> EffectSize:
    x = _as_1d_float_array(group1, name="group1")
    y = _as_1d_float_array(group2, name="group2")
    if x.size < 2 or y.size < 2:
        raise ValueError("Hedges' g requires at least 2 observations per group.")

    s1 = np.var(x, ddof=1)
    s2 = np.var(y, ddof=1)
    pooled = math.sqrt(((x.size - 1) * s1 + (y.size - 1) * s2) / (x.size + y.size - 2))
    if pooled == 0:
        raise ValueError("Hedges' g is undefined because the pooled standard deviation is zero.")

    d = (np.mean(x) - np.mean(y)) / pooled
    correction = 1.0 - (3.0 / (4.0 * (x.size + y.size) - 9.0))
    g = correction * d
    return EffectSize(name="Hedges_g", value=float(g), interpretation=interpret_hedges_g(float(g)))


def _interpret_cliffs_delta(delta: float) -> str:
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def _probability_of_superiority_from_arrays(x: np.ndarray, y: np.ndarray) -> float:
    diffs = x[:, None] - y[None, :]
    wins = np.sum(diffs > 0)
    ties = np.sum(diffs == 0)
    return float(wins + 0.5 * ties) / float(diffs.size)


def _probability_of_superiority_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    confidence_level: float,
    alternative: Alternative,
    n_resamples: int = 5000,
    random_state: int = 0,
) -> ConfidenceInterval:
    rng = np.random.default_rng(random_state)
    nx = x.size
    ny = y.size
    estimates = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        xb = x[rng.integers(nx, size=nx)]
        yb = y[rng.integers(ny, size=ny)]
        estimates[i] = _probability_of_superiority_from_arrays(xb, yb)

    alpha = 1.0 - confidence_level
    if alternative == "two-sided":
        lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    elif alternative == "greater":
        lower = float(np.quantile(estimates, alpha))
        upper = 1.0
    else:
        lower = 0.0
        upper = float(np.quantile(estimates, 1.0 - alpha))

    return ConfidenceInterval(level=confidence_level, lower=float(lower), upper=float(upper))


def _bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    confidence_level: float,
    alternative: Alternative,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 5000,
    random_state: int = 0,
) -> tuple[ConfidenceInterval, int]:
    rng = np.random.default_rng(random_state)
    n = x.size
    finite_estimates: list[float] = []
    nonfinite_count = 0

    for _ in range(n_resamples):
        idx = rng.integers(n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            estimate = float(statistic_fn(x[idx], y[idx]))
        if not np.isfinite(estimate):
            nonfinite_count += 1
            continue
        finite_estimates.append(float(np.clip(estimate, -1.0, 1.0)))

    if len(finite_estimates) < 10:
        # Too few finite bootstrap draws to form a meaningful percentile interval.
        return ConfidenceInterval(level=confidence_level, lower=float("nan"), upper=float("nan")), nonfinite_count

    estimates = np.asarray(finite_estimates, dtype=float)
    alpha = 1.0 - confidence_level
    if alternative == "two-sided":
        lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    elif alternative == "greater":
        lower = float(np.quantile(estimates, alpha))
        upper = 1.0
    else:
        lower = -1.0
        upper = float(np.quantile(estimates, 1.0 - alpha))

    return ConfidenceInterval(level=confidence_level, lower=float(lower), upper=float(upper)), nonfinite_count


def cliffs_delta(group1: ArrayLike1D, group2: ArrayLike1D) -> EffectSize:
    x = _as_1d_float_array(group1, name="group1")
    y = _as_1d_float_array(group2, name="group2")
    superiority = _probability_of_superiority_from_arrays(x, y)
    delta = 2.0 * superiority - 1.0
    return EffectSize(name="Cliffs_delta", value=delta, interpretation=_interpret_cliffs_delta(delta))


# ------------------------------
# Independent-group inference
# ------------------------------


def _welch_df(x: np.ndarray, y: np.ndarray) -> float:
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    nx = x.size
    ny = y.size
    num = (vx / nx + vy / ny) ** 2
    den = ((vx / nx) ** 2) / (nx - 1) + ((vy / ny) ** 2) / (ny - 1)
    return float(num / den)


def _mean_difference_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    confidence_level: float,
    equal_var: bool,
    alternative: Alternative,
) -> tuple[ConfidenceInterval, float]:
    mean_diff = float(np.mean(x) - np.mean(y))
    nx = x.size
    ny = y.size

    if equal_var:
        sp2 = (((nx - 1) * np.var(x, ddof=1)) + ((ny - 1) * np.var(y, ddof=1))) / (nx + ny - 2)
        se = math.sqrt(sp2 * (1.0 / nx + 1.0 / ny))
        df = float(nx + ny - 2)
    else:
        se = math.sqrt(np.var(x, ddof=1) / nx + np.var(y, ddof=1) / ny)
        df = _welch_df(x, y)

    alpha = 1.0 - confidence_level
    if alternative == "two-sided":
        crit = t.ppf(1.0 - alpha / 2.0, df)
        ci = ConfidenceInterval(level=confidence_level, lower=mean_diff - crit * se, upper=mean_diff + crit * se)
    elif alternative == "greater":
        crit = t.ppf(1.0 - alpha, df)
        ci = ConfidenceInterval(level=confidence_level, lower=mean_diff - crit * se, upper=math.inf)
    else:
        crit = t.ppf(1.0 - alpha, df)
        ci = ConfidenceInterval(level=confidence_level, lower=-math.inf, upper=mean_diff + crit * se)
    return ci, df


def compare_independent_groups(
    group1: ArrayLike1D,
    group2: ArrayLike1D,
    *,
    estimand: ComparisonEstimand = "mean_difference",
    method: Optional[str] = None,
    alternative: Alternative = "two-sided",
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> TwoGroupComparisonResult:
    """
    Compare two independent groups using an explicit estimand.

    Parameters
    ----------
    estimand:
        - 'mean_difference': paper-friendly default is Welch's t-test.
        - 'stochastic_dominance': Mann-Whitney U with probability of superiority and Cliff's delta.
    method:
        For mean_difference: {'welch', 'student'}; default is 'welch'.
        For stochastic_dominance: {'mannwhitney'}; default is 'mannwhitney'.

    Notes
    -----
    This function intentionally avoids choosing the inferential target based on
    a normality pre-test. Normality and variance checks are returned as
    diagnostics, not gatekeepers.
    """
    x = _as_1d_float_array(group1, name="group1")
    y = _as_1d_float_array(group2, name="group2")
    if x.size < 2 or y.size < 2:
        raise ValueError("At least 2 observations per group are required.")
    group1_descriptives = describe(x)
    group2_descriptives = describe(y)

    if estimand == "mean_difference":
        test_method = (method or "welch").lower()
        if test_method not in {"welch", "student"}:
            raise ValueError("For estimand='mean_difference', method must be 'welch' or 'student'.")

        assumptions = (
            shapiro_normality(x, alpha=alpha),
            shapiro_normality(y, alpha=alpha),
            equal_variance_check(x, y, alpha=alpha, center="median"),
        )

        equal_var = test_method == "student"
        from scipy.stats import ttest_ind

        statistic, p_value = ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
        ci, df = _mean_difference_ci(
            x, y, confidence_level=confidence_level, equal_var=equal_var, alternative=alternative
        )
        effect = hedges_g(x, y)
        note = (
            "This analysis assumes independent observations within and between groups; paired or repeated-measures designs require different methods. "
            "Welch's t-test is the recommended default for comparing means because it remains valid under unequal variances."
            if not equal_var
            else "This analysis assumes independent observations within and between groups; paired or repeated-measures designs require different methods. Student's t-test assumes equal variances across groups."
        )
        return TwoGroupComparisonResult(
            estimand="mean_difference",
            method="Welch_t_test" if not equal_var else "Students_t_test",
            alternative=alternative,
            statistic=float(statistic),
            p_value=float(p_value),
            estimate=float(np.mean(x) - np.mean(y)),
            estimate_label="mean_difference",
            ci=ci,
            effect_size=effect,
            n1=int(x.size),
            n2=int(y.size),
            group1_descriptives=group1_descriptives,
            group2_descriptives=group2_descriptives,
            df=df,
            assumptions=assumptions,
            notes=(note,),
        )

    if estimand == "stochastic_dominance":
        test_method = (method or "mannwhitney").lower()
        if test_method != "mannwhitney":
            raise ValueError("For estimand='stochastic_dominance', method must be 'mannwhitney'.")

        statistic, p_value = mannwhitneyu(x, y, alternative=alternative, method="auto")
        superiority = _probability_of_superiority_from_arrays(x, y)
        ci = _probability_of_superiority_ci(
            x,
            y,
            confidence_level=confidence_level,
            alternative=alternative,
        )
        effect = cliffs_delta(x, y)
        effect = EffectSize(
            name=effect.name,
            value=effect.value,
            interpretation=effect.interpretation,
            ci=ConfidenceInterval(
                level=ci.level,
                lower=2.0 * ci.lower - 1.0,
                upper=2.0 * ci.upper - 1.0,
            ),
        )
        note = (
            "This analysis assumes independent observations within and between groups; paired or repeated-measures designs require different methods. "
            "Mann-Whitney U targets stochastic dominance rather than mean differences. The reported estimand is the probability of superiority, defined as P(group1 > group2) + 0.5 P(tie), with a percentile bootstrap confidence interval. Cliff's delta is reported as the corresponding standardized effect size. Separate Shapiro or equal-variance tests are not reported here because they are not the key diagnostics for this estimand; instead inspect overlap, ties, and whether a location-shift interpretation would require defensible same-shape assumptions."
        )
        return TwoGroupComparisonResult(
            estimand="stochastic_dominance",
            method="Mann_Whitney_U",
            alternative=alternative,
            statistic=float(statistic),
            p_value=float(p_value),
            estimate=superiority,
            estimate_label="probability_of_superiority",
            ci=ci,
            effect_size=effect,
            n1=int(x.size),
            n2=int(y.size),
            group1_descriptives=group1_descriptives,
            group2_descriptives=group2_descriptives,
            df=None,
            assumptions=(),
            notes=(note,),
        )

    raise ValueError("estimand must be 'mean_difference' or 'stochastic_dominance'.")


# ------------------------------
# Correlation inference
# ------------------------------


def _pearson_ci(r: float, n: int, confidence_level: float, alternative: Alternative) -> ConfidenceInterval:
    if n < 4:
        raise ValueError("Pearson confidence interval via Fisher z requires n >= 4.")
    if r >= 1.0:
        return ConfidenceInterval(level=confidence_level, lower=1.0, upper=1.0)
    if r <= -1.0:
        return ConfidenceInterval(level=confidence_level, lower=-1.0, upper=-1.0)
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    alpha = 1.0 - confidence_level
    if alternative == "two-sided":
        z_crit = float(norm.ppf(1.0 - alpha / 2.0))
        lower = float(np.tanh(z - z_crit * se))
        upper = float(np.tanh(z + z_crit * se))
        return ConfidenceInterval(level=confidence_level, lower=lower, upper=upper)
    if alternative == "greater":
        # One-sided (1-alpha) lower confidence bound.
        z_crit = float(norm.ppf(1.0 - alpha))
        lower = float(np.tanh(z - z_crit * se))
        return ConfidenceInterval(level=confidence_level, lower=lower, upper=1.0)
    # alternative == "less"
    z_crit = float(norm.ppf(1.0 - alpha))
    upper = float(np.tanh(z + z_crit * se))
    return ConfidenceInterval(level=confidence_level, lower=-1.0, upper=upper)


def correlation(
    x: ArrayLike1D,
    y: ArrayLike1D,
    *,
    method: CorrelationMethod = "pearson",
    alternative: Alternative = "two-sided",
    confidence_level: float = 0.95,
    alpha: float = 0.05,
) -> CorrelationResult:
    x_arr = _as_1d_float_array(x, name="x")
    y_arr = _as_1d_float_array(y, name="y")
    if x_arr.size != y_arr.size:
        raise ValueError(f"x and y must have equal length, got {x_arr.size} and {y_arr.size}.")
    if x_arr.size < 3:
        raise ValueError("Correlation requires at least 3 paired observations.")
    _require_variation(x_arr, name="x")
    _require_variation(y_arr, name="y")

    if method == "pearson":
        coefficient, p_value = pearsonr(x_arr, y_arr, alternative=alternative)
        ci = _pearson_ci(float(coefficient), int(x_arr.size), confidence_level, alternative)
        assumptions: tuple[AssumptionCheck, ...] = ()
        notes: tuple[str, ...] = (
            "Pearson correlation targets linear association. The key diagnostics are the paired-data scatterplot, focusing on linearity, influential outliers, and other joint-structure issues such as heteroscedasticity. Marginal normality of x and y is not the main assumption, so separate normality tests are intentionally not reported here.",
        )
    elif method == "spearman":
        coefficient, p_value = spearmanr(x_arr, y_arr, alternative=alternative)
        n_resamples = 5000
        ci, nonfinite = _bootstrap_correlation_ci(
            x_arr,
            y_arr,
            confidence_level=confidence_level,
            alternative=alternative,
            statistic_fn=lambda a, b: float(spearmanr(a, b, alternative=alternative).statistic),
            n_resamples=n_resamples,
        )
        assumptions = ()
        base_notes: list[str] = [
            "Spearman correlation targets monotonic association using ranks. Diagnostics should focus on whether the relationship is monotonic and on unusual paired observations or many ties; marginal normality tests are not relevant here. A percentile bootstrap confidence interval is reported to provide uncertainty without relying on large-sample normal approximations for rho.",
        ]
        if nonfinite > 0:
            base_notes.append(
                f"Bootstrap CI note: dropped {nonfinite} of {n_resamples} resamples with non-finite Spearman estimates (typically due to ties/degenerate resamples)."
            )
        notes = tuple(base_notes)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'.")

    return CorrelationResult(
        method=method,
        alternative=alternative,
        coefficient=float(coefficient),
        p_value=float(p_value),
        n=int(x_arr.size),
        ci=ci,
        x_descriptives=describe(x_arr),
        y_descriptives=describe(y_arr),
        assumptions=assumptions,
        notes=notes,
    )


# ------------------------------
# Interpretation helpers
# ------------------------------


def interpret_correlation_coefficient(r: float) -> str:
    ar = abs(r)
    if ar < 0.10:
        return "negligible"
    if ar < 0.30:
        return "weak"
    if ar < 0.50:
        return "moderate"
    if ar < 0.70:
        return "strong"
    if ar < 0.90:
        return "very strong"
    return "near perfect"


def interpret_two_group(result: TwoGroupComparisonResult, *, alpha: float = 0.05) -> str:
    direction = "higher" if result.estimate > 0 else "lower" if result.estimate < 0 else "equal"

    parts: list[str] = []
    if result.estimand == "mean_difference":
        parts.append(f"The estimated mean difference (group 1 - group 2) is {result.estimate:.3f} units.")
        parts.append(f"Group 1 scored {direction} than group 2 by {abs(result.estimate):.3f} units.")
        if result.ci is not None and math.isfinite(result.ci.lower) and math.isfinite(result.ci.upper):
            contains_zero = result.ci.lower <= 0 <= result.ci.upper
            parts.append(
                f"The {int(result.ci.level * 100)}% confidence interval "
                f"[{result.ci.lower:.3f}, {result.ci.upper:.3f}] "
                + (
                    "includes zero, consistent with no meaningful difference."
                    if contains_zero
                    else "excludes zero, reinforcing the finding."
                )
            )
        parts.append(
            f"The exact inferential result is {apa_pvalue(result.p_value)}"
            + (
                f", which would usually be described as statistically significant at alpha = {alpha}."
                if result.p_value < alpha
                else f", which would not usually be described as statistically significant at alpha = {alpha}."
            )
        )
    elif result.estimand == "stochastic_dominance":
        if result.estimate > 0.5:
            parts.append(
                f"A randomly chosen value from group 1 exceeds one from group 2 about {result.estimate:.3f} of the time, counting ties as one-half."
            )
        elif result.estimate < 0.5:
            parts.append(
                f"A randomly chosen value from group 1 exceeds one from group 2 about {result.estimate:.3f} of the time, so group 2 tends to have larger values."
            )
        else:
            parts.append(
                "The probability of superiority is 0.500, so neither group shows stochastic dominance over the other."
            )

        if result.ci is not None and math.isfinite(result.ci.lower) and math.isfinite(result.ci.upper):
            contains_half = result.ci.lower <= 0.5 <= result.ci.upper
            parts.append(
                f"The {int(result.ci.level * 100)}% confidence interval "
                f"[{result.ci.lower:.3f}, {result.ci.upper:.3f}] "
                + (
                    "includes 0.500, consistent with no stochastic dominance."
                    if contains_half
                    else "excludes 0.500, reinforcing the dominance pattern."
                )
            )
        parts.append(
            f"The exact inferential result is {apa_pvalue(result.p_value)}"
            + (
                f", which would usually be described as statistically significant at alpha = {alpha}."
                if result.p_value < alpha
                else f", which would not usually be described as statistically significant at alpha = {alpha}."
            )
        )

    if result.effect_size is not None and result.effect_size.interpretation is not None:
        parts.append(
            f"The effect size ({result.effect_size.name.replace('_', ' ')} = {result.effect_size.value:.3f}) "
            f"is {result.effect_size.interpretation}."
        )

    return " ".join(parts)


def interpret_correlation(result: CorrelationResult, *, alpha: float = 0.05) -> str:
    strength = interpret_correlation_coefficient(result.coefficient)
    direction = "positive" if result.coefficient > 0 else "negative" if result.coefficient < 0 else "zero"
    symbol = "r" if result.method == "pearson" else "rho"

    parts: list[str] = []
    parts.append(
        f"The {result.method} correlation is {direction} and {strength} ({symbol} = {result.coefficient:.3f})."
    )

    if result.ci is not None:
        parts.append(
            f"The {int(result.ci.level * 100)}% confidence interval "
            f"[{result.ci.lower:.3f}, {result.ci.upper:.3f}] "
            + (
                "includes zero, consistent with no association."
                if result.ci.lower <= 0 <= result.ci.upper
                else "excludes zero, reinforcing the association."
            )
        )
    parts.append(
        f"The exact inferential result is {apa_pvalue(result.p_value)}"
        + (
            f", which would usually be described as statistically significant at alpha = {alpha}."
            if result.p_value < alpha
            else f", which would not usually be described as statistically significant at alpha = {alpha}."
        )
    )

    return " ".join(parts)


# ------------------------------
# Reporting helpers
# ------------------------------


def apa_pvalue(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def report_two_group(
    result: TwoGroupComparisonResult,
    digits: int = 2,
    *,
    include_interpretation: bool = True,
    alpha: float = 0.05,
) -> str:
    group1_context = (
        f"Group 1 descriptives: n = {result.group1_descriptives.n}, "
        f"mean = {result.group1_descriptives.mean:.{digits}f}, SD = {result.group1_descriptives.sd:.{digits}f}, "
        f"median = {result.group1_descriptives.median:.{digits}f}, range [{result.group1_descriptives.minimum:.{digits}f}, {result.group1_descriptives.maximum:.{digits}f}]"
    )
    group2_context = (
        f"Group 2 descriptives: n = {result.group2_descriptives.n}, "
        f"mean = {result.group2_descriptives.mean:.{digits}f}, SD = {result.group2_descriptives.sd:.{digits}f}, "
        f"median = {result.group2_descriptives.median:.{digits}f}, range [{result.group2_descriptives.minimum:.{digits}f}, {result.group2_descriptives.maximum:.{digits}f}]"
    )
    if result.estimand == "mean_difference":
        assert result.ci is not None
        ci_str = f"({int(result.ci.level * 100)}% CI [{result.ci.lower:.{digits}f}, {result.ci.upper:.{digits}f}]), "
        stat_name = "t"
        df = f"{result.df:.{digits}f}" if result.df is not None else "?"
        eff = ""
        if result.effect_size is not None:
            eff = f", {result.effect_size.name.replace('_', ' ')} = {result.effect_size.value:.{digits}f}"
        text = (
            f"{result.method.replace('_', ' ')} showed a mean difference of {result.estimate:.{digits}f} "
            f"{ci_str}"
            f"{stat_name}({df}) = {result.statistic:.{digits}f}, {apa_pvalue(result.p_value)}{eff}."
        )
    else:
        ci_str = ""
        if result.ci is not None:
            ci_str = f" ({int(result.ci.level * 100)}% CI [{result.ci.lower:.{digits}f}, {result.ci.upper:.{digits}f}])"
        eff = ""
        if result.effect_size is not None:
            eff = f", Cliff's delta = {result.effect_size.value:.{digits}f}"
            if result.effect_size.ci is not None:
                eff += (
                    f" ({int(result.effect_size.ci.level * 100)}% CI "
                    f"[{result.effect_size.ci.lower:.{digits}f}, {result.effect_size.ci.upper:.{digits}f}])"
                )
        text = (
            f"{result.method.replace('_', ' ')} estimated a probability of superiority of "
            f"{result.estimate:.{digits}f}{ci_str}, U = {result.statistic:.{digits}f}, {apa_pvalue(result.p_value)}{eff}."
        )
    text += f"\n{group1_context}.\n{group2_context}."

    if include_interpretation:
        text += "\n\nInterpretation: " + interpret_two_group(result, alpha=alpha)
    return text


def report_correlation(
    result: CorrelationResult,
    digits: int = 2,
    *,
    include_interpretation: bool = True,
    alpha: float = 0.05,
) -> str:
    symbol = "r" if result.method == "pearson" else "rho"
    ci = ""
    if result.ci is not None:
        ci = f", {int(result.ci.level * 100)}% CI [{result.ci.lower:.{digits}f}, {result.ci.upper:.{digits}f}]"
    text = (
        f"{result.method.capitalize()} correlation was {symbol} = {result.coefficient:.{digits}f}{ci}, "
        f"{apa_pvalue(result.p_value)}, n = {result.n}."
    )
    text += (
        f"\nX descriptives: n = {result.x_descriptives.n}, mean = {result.x_descriptives.mean:.{digits}f}, "
        f"SD = {result.x_descriptives.sd:.{digits}f}, median = {result.x_descriptives.median:.{digits}f}, "
        f"range [{result.x_descriptives.minimum:.{digits}f}, {result.x_descriptives.maximum:.{digits}f}]."
    )
    text += (
        f"\nY descriptives: n = {result.y_descriptives.n}, mean = {result.y_descriptives.mean:.{digits}f}, "
        f"SD = {result.y_descriptives.sd:.{digits}f}, median = {result.y_descriptives.median:.{digits}f}, "
        f"range [{result.y_descriptives.minimum:.{digits}f}, {result.y_descriptives.maximum:.{digits}f}]."
    )

    if include_interpretation:
        text += "\n\nInterpretation: " + interpret_correlation(result, alpha=alpha)
    return text


__all__ = [
    "AssumptionCheck",
    "ConfidenceInterval",
    "CorrelationResult",
    "DescriptiveStats",
    "EffectSize",
    "TwoGroupComparisonResult",
    "anderson_darling_candidates",
    "apa_pvalue",
    "cliffs_delta",
    "compare_independent_groups",
    "correlation",
    "describe",
    "equal_variance_check",
    "hedges_g",
    "interpret_correlation",
    "interpret_correlation_coefficient",
    "interpret_two_group",
    "report_correlation",
    "report_two_group",
    "shapiro_normality",
]
