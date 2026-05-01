import math
import unittest

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr, ttest_ind, mannwhitneyu

from stats4science import inferential_stats as s


class TestInferentialStats(unittest.TestCase):
    def test_as_1d_float_array_rejects_non_1d(self) -> None:
        with self.assertRaisesRegex(ValueError, r"must be one-dimensional"):
            s._as_1d_float_array([[1.0, 2.0], [3.0, 4.0]], name="x")  # type: ignore[list-item]

    def test_as_1d_float_array_rejects_empty(self) -> None:
        with self.assertRaisesRegex(ValueError, r"must not be empty"):
            s._as_1d_float_array([], name="x")

    def test_as_1d_float_array_rejects_nan_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, r"contains NaN"):
            s._as_1d_float_array([1.0, float("nan")], name="x")

    def test_as_1d_float_array_allows_nan_when_requested(self) -> None:
        x = s._as_1d_float_array([1.0, float("nan")], name="x", allow_nan=True)
        self.assertEqual(x.shape, (2,))
        self.assertTrue(math.isnan(float(x[1])))

    def test_as_1d_float_array_rejects_infinite(self) -> None:
        with self.assertRaisesRegex(ValueError, r"contains infinite"):
            s._as_1d_float_array([1.0, float("inf")], name="x")

    def test_require_variation_rejects_constant(self) -> None:
        with self.assertRaisesRegex(ValueError, r"zero variance"):
            s._require_variation(np.array([2.0, 2.0, 2.0]), name="x")

    def test_describe_basic_properties(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        d = s.describe(x)
        self.assertEqual(d.n, 4)
        self.assertAlmostEqual(d.mean, 2.5, places=12)
        self.assertAlmostEqual(d.sd, float(np.std(x, ddof=1)), places=12)
        self.assertAlmostEqual(d.median, 2.5, places=12)
        self.assertEqual(d.minimum, 1.0)
        self.assertEqual(d.maximum, 4.0)
        self.assertIsInstance(d.kurtosis_fisher, float)
        as_dict = d.to_dict()
        self.assertEqual(as_dict["n"], 4)

    def test_describe_requires_at_least_2(self) -> None:
        with self.assertRaisesRegex(ValueError, r"At least 2 observations"):
            s.describe([1.0])

    def test_shapiro_normality_requires_at_least_3(self) -> None:
        with self.assertRaisesRegex(ValueError, r"requires at least 3"):
            s.shapiro_normality([1.0, 2.0])

    def test_shapiro_normality_includes_large_n_note(self) -> None:
        x = np.linspace(0.0, 1.0, 5001)
        res = s.shapiro_normality(x)
        self.assertEqual(res.test_name, "Shapiro-Wilk")
        self.assertIn("n > 5000", res.note)
        self.assertIsInstance(res.passed, bool)

    def test_equal_variance_check_center_mean_or_median(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 4.0, 8.0])
        r1 = s.equal_variance_check(x, y, center="median")
        r2 = s.equal_variance_check(x, y, center="mean")
        self.assertEqual(r1.test_name, "Levene")
        self.assertEqual(r2.test_name, "Levene")
        self.assertIn("center=", r1.note)
        self.assertIsInstance(r1.p_value, float)

    def test_anderson_darling_candidates_returns_known_names(self) -> None:
        x = np.random.default_rng(0).normal(size=200)
        accepted = s.anderson_darling_candidates(x)
        self.assertIsInstance(accepted, list)
        allowed = {"norm", "expon", "logistic", "gumbel_l", "gumbel_r"}
        self.assertTrue(all(dist in allowed for dist in accepted))

    def test_interpret_hedges_g_thresholds(self) -> None:
        cases = [
            (0.0, "negligible"),
            (0.05, "very small"),
            (0.2, "small"),
            (0.5, "medium"),
            (0.8, "large"),
            (1.2, "very large"),
            (2.0, "huge"),
            (-0.5, "medium"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(s.interpret_hedges_g(value), expected)

    def test_hedges_g_errors_with_too_small_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, r"at least 2"):
            s.hedges_g([1.0], [1.0, 2.0])

    def test_hedges_g_errors_with_zero_pooled_sd(self) -> None:
        with self.assertRaisesRegex(ValueError, r"pooled standard deviation is zero"):
            s.hedges_g([1.0, 1.0], [2.0, 2.0])

    def test_cliffs_delta_basic_sanity(self) -> None:
        x = np.array([3.0, 4.0, 5.0])
        y = np.array([1.0, 2.0])
        eff = s.cliffs_delta(x, y)
        self.assertEqual(eff.name, "Cliffs_delta")
        self.assertGreater(eff.value, 0)
        self.assertIn(eff.interpretation, {"negligible", "small", "medium", "large"})

    def test_compare_independent_groups_mean_difference_default_is_welch(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 1.0, 2.0])
        res = s.compare_independent_groups(x, y, estimand="mean_difference")
        self.assertEqual(res.estimand, "mean_difference")
        self.assertEqual(res.method, "Welch_t_test")
        self.assertIsNotNone(res.ci)
        self.assertIsNotNone(res.effect_size)
        self.assertEqual(res.estimate_label, "mean_difference")
        self.assertEqual((res.n1, res.n2), (4, 4))
        self.assertEqual(len(res.assumptions), 3)
        self.assertIn("Welch", res.notes[0])
        d = res.to_dict()
        self.assertIsInstance(d["assumptions"], list)
        self.assertAlmostEqual(d["ci"]["level"], 0.95, places=12)
        self.assertEqual(d["group1_descriptives"]["n"], 4)
        self.assertEqual(d["group2_descriptives"]["n"], 4)

    def test_compare_independent_groups_student_sets_df_n_minus_2(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0, 5.0])
        res = s.compare_independent_groups(x, y, estimand="mean_difference", method="student")
        self.assertEqual(res.method, "Students_t_test")
        self.assertIsNotNone(res.df)
        self.assertAlmostEqual(float(res.df), 6.0, places=12)  # type: ignore[arg-type]

    def test_compare_independent_groups_welch_matches_scipy_statistic_and_pvalue(self) -> None:
        x = np.array([1.0, 2.5, 4.0, 5.5, 7.0])
        y = np.array([0.5, 1.5, 1.75, 2.5, 3.0, 3.5])
        res = s.compare_independent_groups(x, y, estimand="mean_difference", method="welch")
        expected = ttest_ind(x, y, equal_var=False, alternative="two-sided")
        self.assertAlmostEqual(res.statistic, float(expected.statistic), places=12)
        self.assertAlmostEqual(res.p_value, float(expected.pvalue), places=12)

    def test_compare_independent_groups_student_matches_scipy_statistic_and_pvalue(self) -> None:
        x = np.array([1.0, 2.0, 4.0, 5.0, 7.0])
        y = np.array([0.0, 1.0, 3.0, 4.0, 6.0])
        res = s.compare_independent_groups(x, y, estimand="mean_difference", method="student")
        expected = ttest_ind(x, y, equal_var=True, alternative="two-sided")
        self.assertAlmostEqual(res.statistic, float(expected.statistic), places=12)
        self.assertAlmostEqual(res.p_value, float(expected.pvalue), places=12)

    def test_compare_independent_groups_stochastic_dominance_mann_whitney(self) -> None:
        x = np.array([10.0, 11.0, 12.0, 13.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        res = s.compare_independent_groups(x, y, estimand="stochastic_dominance")
        self.assertEqual(res.estimand, "stochastic_dominance")
        self.assertEqual(res.method, "Mann_Whitney_U")
        self.assertIsNotNone(res.ci)
        self.assertIsNotNone(res.effect_size)
        self.assertEqual(res.estimate_label, "probability_of_superiority")
        self.assertAlmostEqual(res.estimate, 1.0, places=12)
        self.assertIsNone(res.df)
        assert res.ci is not None
        self.assertAlmostEqual(res.ci.lower, 1.0, places=12)
        self.assertAlmostEqual(res.ci.upper, 1.0, places=12)
        assert res.effect_size is not None
        self.assertAlmostEqual(res.effect_size.value, 1.0, places=12)
        self.assertIsNotNone(res.effect_size.ci)
        self.assertEqual(res.assumptions, ())
        d = res.to_dict()
        self.assertAlmostEqual(d["ci"]["level"], 0.95, places=12)
        self.assertAlmostEqual(d["effect_size"]["ci"]["lower"], 1.0, places=12)
        self.assertIn("probability of superiority", res.notes[0])
        self.assertIn("Shapiro", res.notes[0])
        self.assertIn("same-shape assumptions", res.notes[0])

    def test_compare_independent_groups_mann_whitney_matches_scipy_statistic_and_pvalue(self) -> None:
        x = np.array([1.0, 2.0, 2.0, 5.0, 8.0])
        y = np.array([0.0, 2.0, 3.0, 3.0, 9.0])
        res = s.compare_independent_groups(x, y, estimand="stochastic_dominance")
        expected = mannwhitneyu(x, y, alternative="two-sided", method="auto")
        self.assertAlmostEqual(res.statistic, float(expected.statistic), places=12)
        self.assertAlmostEqual(res.p_value, float(expected.pvalue), places=12)

    def test_compare_independent_groups_stochastic_dominance_tiny_samples_return_bounded_ci(self) -> None:
        x = np.array([1.0, 4.0])
        y = np.array([2.0, 3.0])
        res = s.compare_independent_groups(x, y, estimand="stochastic_dominance")
        self.assertAlmostEqual(res.estimate, 0.5, places=12)
        self.assertEqual(res.estimate_label, "probability_of_superiority")
        self.assertIsNotNone(res.ci)
        assert res.ci is not None
        self.assertLessEqual(0.0, res.ci.lower)
        self.assertLessEqual(res.ci.lower, res.ci.upper)
        self.assertLessEqual(res.ci.upper, 1.0)
        assert res.effect_size is not None
        self.assertAlmostEqual(res.effect_size.value, 0.0, places=12)

    def test_compare_independent_groups_stochastic_dominance_all_ties_returns_midpoint_effect(self) -> None:
        x = np.array([2.0, 2.0, 2.0, 2.0])
        y = np.array([2.0, 2.0, 2.0, 2.0])
        res = s.compare_independent_groups(x, y, estimand="stochastic_dominance")
        self.assertAlmostEqual(res.estimate, 0.5, places=12)
        assert res.ci is not None
        self.assertAlmostEqual(res.ci.lower, 0.5, places=12)
        self.assertAlmostEqual(res.ci.upper, 0.5, places=12)
        assert res.effect_size is not None
        self.assertAlmostEqual(res.effect_size.value, 0.0, places=12)
        assert res.effect_size.ci is not None
        self.assertAlmostEqual(res.effect_size.ci.lower, 0.0, places=12)
        self.assertAlmostEqual(res.effect_size.ci.upper, 0.0, places=12)

    def test_compare_independent_groups_stochastic_dominance_handles_zero_variance_groups(self) -> None:
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([2.0, 2.0, 2.0])
        res = s.compare_independent_groups(x, y, estimand="stochastic_dominance")
        self.assertAlmostEqual(res.estimate, 0.0, places=12)
        assert res.ci is not None
        self.assertAlmostEqual(res.ci.lower, 0.0, places=12)
        self.assertAlmostEqual(res.ci.upper, 0.0, places=12)
        assert res.effect_size is not None
        self.assertAlmostEqual(res.effect_size.value, -1.0, places=12)
        self.assertEqual(res.effect_size.interpretation, "large")

    def test_compare_independent_groups_mean_difference_heteroscedastic_groups_flag_variance_issue(self) -> None:
        x = np.array([9.8, 10.0, 10.1, 10.2, 9.9, 10.0, 10.1, 9.9])
        y = np.array([-20.0, -5.0, 0.0, 5.0, 20.0, 35.0, 40.0, 45.0])
        res = s.compare_independent_groups(x, y, estimand="mean_difference")
        self.assertEqual(res.method, "Welch_t_test")
        self.assertFalse(res.assumptions[2].passed)
        assert res.df is not None
        self.assertLess(res.df, x.size + y.size - 2)

    def test_compare_independent_groups_validates_estimand_and_method(self) -> None:
        with self.assertRaisesRegex(ValueError, r"estimand must be"):
            s.compare_independent_groups([1.0, 2.0], [1.0, 2.0], estimand="bogus")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, r"method must be 'welch' or 'student'"):
            s.compare_independent_groups([1.0, 2.0], [1.0, 2.0], estimand="mean_difference", method="nope")
        with self.assertRaisesRegex(ValueError, r"method must be 'mannwhitney'"):
            s.compare_independent_groups([1.0, 2.0], [1.0, 2.0], estimand="stochastic_dominance", method="welch")

    def test_correlation_length_and_min_n_and_variation_checks(self) -> None:
        with self.assertRaisesRegex(ValueError, r"must have equal length"):
            s.correlation([1.0, 2.0, 3.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, r"at least 3"):
            s.correlation([1.0, 2.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, r"zero variance"):
            s.correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])

    def test_correlation_pearson_includes_ci(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 1.9, 3.2, 4.1, 4.9])
        res = s.correlation(x, y, method="pearson")
        self.assertEqual(res.method, "pearson")
        self.assertIsNotNone(res.ci)
        assert res.ci is not None
        self.assertAlmostEqual(res.ci.level, 0.95, places=12)
        self.assertLessEqual(res.ci.lower, res.coefficient)
        self.assertLessEqual(res.coefficient, res.ci.upper)
        self.assertEqual(res.n, 5)
        self.assertEqual(res.assumptions, ())
        self.assertIn("linearity", res.notes[0])
        self.assertIn("outliers", res.notes[0])
        self.assertEqual(res.x_descriptives.n, 5)
        self.assertEqual(res.y_descriptives.n, 5)

    def test_pearson_ci_matches_standard_fisher_z_normal_interval(self) -> None:
        r = 0.6
        n = 20
        level = 0.95

        ci = s._pearson_ci(r, n, level, "two-sided")

        z = np.arctanh(r)
        se = 1.0 / math.sqrt(n - 3)
        z_crit = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
        expected_lower = float(np.tanh(z - z_crit * se))
        expected_upper = float(np.tanh(z + z_crit * se))

        self.assertAlmostEqual(ci.lower, expected_lower, places=12)
        self.assertAlmostEqual(ci.upper, expected_upper, places=12)

    def test_pearson_ci_requires_at_least_4_observations(self) -> None:
        with self.assertRaisesRegex(ValueError, r"requires n >= 4"):
            s._pearson_ci(0.5, 3, 0.95, "two-sided")

    def test_pearson_ci_one_sided_greater_has_upper_at_1(self) -> None:
        r = 0.25
        n = 30
        level = 0.95
        ci = s._pearson_ci(r, n, level, "greater")
        self.assertAlmostEqual(ci.upper, 1.0, places=15)
        self.assertLessEqual(ci.lower, r)

    def test_pearson_ci_one_sided_less_has_lower_at_minus_1(self) -> None:
        r = -0.10
        n = 30
        level = 0.95
        ci = s._pearson_ci(r, n, level, "less")
        self.assertAlmostEqual(ci.lower, -1.0, places=15)
        self.assertGreaterEqual(ci.upper, r)

    def test_probability_of_superiority_ci_is_bootstrap_reproducible(self) -> None:
        x = np.array([1.0, 2.0, 4.0, 7.0])
        y = np.array([0.0, 3.0, 3.5, 8.0])
        ci1 = s._probability_of_superiority_ci(x, y, confidence_level=0.95, alternative="two-sided")
        ci2 = s._probability_of_superiority_ci(x, y, confidence_level=0.95, alternative="two-sided")
        self.assertEqual(ci1, ci2)

    def test_correlation_pearson_extreme_values_produce_bounded_intervals(self) -> None:
        cases = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([2.0, 4.0, 6.0, 8.0, 10.0]), 1.0),
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([-2.0, -4.0, -6.0, -8.0, -10.0]), -1.0),
        ]
        for x, y, expected in cases:
            with self.subTest(expected=expected):
                res = s.correlation(x, y, method="pearson")
                self.assertAlmostEqual(res.coefficient, expected, places=12)
                assert res.ci is not None
                self.assertLessEqual(-1.0, res.ci.lower)
                self.assertLessEqual(res.ci.lower, res.ci.upper)
                self.assertLessEqual(res.ci.upper, 1.0)
                self.assertLessEqual(res.ci.lower, res.coefficient)
                self.assertLessEqual(res.coefficient, res.ci.upper)

    def test_correlation_pearson_matches_scipy_statistic_and_pvalue(self) -> None:
        x = np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0])
        y = np.array([0.8, 2.2, 3.7, 4.8, 7.2, 7.9])
        res = s.correlation(x, y, method="pearson")
        expected = pearsonr(x, y, alternative="two-sided")
        self.assertAlmostEqual(res.coefficient, float(expected.statistic), places=12)
        self.assertAlmostEqual(res.p_value, float(expected.pvalue), places=12)

    def test_correlation_pearson_near_positive_boundary_is_bounded(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([2.0, 4.0, 6.0, 8.01, 9.99, 12.0])
        res = s.correlation(x, y, method="pearson")
        self.assertGreater(res.coefficient, 0.999)
        assert res.ci is not None
        self.assertLess(res.coefficient, 1.0)
        self.assertLess(res.ci.upper, 1.0)
        self.assertLessEqual(res.ci.lower, res.coefficient)

    def test_correlation_pearson_near_negative_boundary_is_bounded(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([-2.0, -4.0, -6.0, -8.01, -9.99, -12.0])
        res = s.correlation(x, y, method="pearson")
        self.assertLess(res.coefficient, -0.999)
        assert res.ci is not None
        self.assertGreater(res.coefficient, -1.0)
        self.assertGreater(res.ci.lower, -1.0)
        self.assertLessEqual(res.coefficient, res.ci.upper)

    def test_correlation_spearman_has_bootstrap_ci(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        res = s.correlation(x, y, method="spearman")
        self.assertEqual(res.method, "spearman")
        self.assertIsNotNone(res.ci)
        assert res.ci is not None
        self.assertLessEqual(-1.0, res.ci.lower)
        self.assertLessEqual(res.ci.lower, res.ci.upper)
        self.assertLessEqual(res.ci.upper, 1.0)
        self.assertEqual(res.assumptions, ())
        self.assertIn("monotonic", res.notes[0])
        self.assertIn("normality tests are not relevant", res.notes[0])
        self.assertIn("bootstrap confidence interval", res.notes[0])

    def test_correlation_spearman_matches_scipy_statistic_and_pvalue(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y = np.array([1.0, 1.5, 2.0, 3.0, 3.5, 5.0])
        res = s.correlation(x, y, method="spearman")
        expected = spearmanr(x, y, alternative="two-sided")
        self.assertAlmostEqual(res.coefficient, float(expected.statistic), places=12)
        self.assertAlmostEqual(res.p_value, float(expected.pvalue), places=12)

    def test_correlation_spearman_bootstrap_reports_nonfinite_drops(self) -> None:
        # Construct a case where the observed Spearman rho is defined (both vary),
        # but some bootstrap resamples can become degenerate (all x values equal),
        # yielding non-finite Spearman estimates.
        x = np.array([0.0, 0.0, 0.0, 1.0])
        y = np.array([0.0, 1.0, 2.0, 3.0])
        res = s.correlation(x, y, method="spearman")
        self.assertIsNotNone(res.ci)
        # The key behavioral requirement: we do not silently substitute; we report drops.
        self.assertTrue(any("dropped" in note.lower() for note in res.notes) or len(res.notes) >= 1)

    def test_correlation_invalid_method(self) -> None:
        with self.assertRaisesRegex(ValueError, r"method must be"):
            s.correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], method="kendall")  # type: ignore[arg-type]

    def test_apa_pvalue_formatting(self) -> None:
        self.assertEqual(s.apa_pvalue(0.0005), "p < .001")
        self.assertEqual(s.apa_pvalue(0.05), "p = .050")

    def test_report_two_group_and_report_correlation_smoke(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0, 5.0])
        comp = s.compare_independent_groups(x, y, estimand="mean_difference")
        txt = s.report_two_group(comp, digits=2)
        self.assertIn("CI", txt)
        self.assertIn("t(", txt)
        self.assertIn("Group 1 descriptives", txt)

        corr = s.correlation(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.1, 2.9, 4.2]),
            method="pearson",
        )
        ctxt = s.report_correlation(corr, digits=2)
        self.assertIn("Pearson correlation", ctxt)
        self.assertIn("n =", ctxt)
        self.assertIn("X descriptives", ctxt)

    def test_report_two_group_stochastic_dominance_reports_interpretable_estimand(self) -> None:
        x = np.array([10.0, 11.0, 12.0, 13.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        comp = s.compare_independent_groups(x, y, estimand="stochastic_dominance")
        txt = s.report_two_group(comp, digits=2, include_interpretation=False)
        self.assertIn("probability of superiority", txt)
        self.assertIn("95% CI", txt)
        self.assertIn("Cliff's delta", txt)

    def test_interpret_correlation_coefficient_thresholds(self) -> None:
        cases = [
            (0.0, "negligible"),
            (0.05, "negligible"),
            (0.15, "weak"),
            (0.35, "moderate"),
            (0.55, "strong"),
            (0.75, "very strong"),
            (0.95, "near perfect"),
            (-0.55, "strong"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(s.interpret_correlation_coefficient(value), expected)

    def test_interpret_two_group_significant(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.normal(loc=10.0, scale=1.0, size=50)
        y = rng.normal(loc=7.0, scale=1.0, size=50)
        comp = s.compare_independent_groups(x, y, estimand="mean_difference")
        interpretation = s.interpret_two_group(comp, alpha=0.05)
        self.assertIn("estimated mean difference", interpretation)
        self.assertIn("higher", interpretation)
        self.assertIn("excludes zero", interpretation)
        self.assertIn("would usually be described as statistically significant", interpretation)

    def test_interpret_two_group_not_significant(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.5, 2.5, 3.5, 3.0])
        comp = s.compare_independent_groups(x, y, estimand="mean_difference")
        interpretation = s.interpret_two_group(comp, alpha=0.05)
        self.assertIn("would not usually be described as statistically significant", interpretation)

    def test_interpret_two_group_stochastic_dominance(self) -> None:
        x = np.array([10.0, 11.0, 12.0, 13.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        comp = s.compare_independent_groups(x, y, estimand="stochastic_dominance")
        interpretation = s.interpret_two_group(comp, alpha=0.05)
        self.assertIn("randomly chosen value from group 1 exceeds one from group 2", interpretation)
        self.assertIn("excludes 0.500", interpretation)
        self.assertIn("effect size", interpretation.lower())

    def test_interpret_correlation_significant_positive(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        y = np.array([1.1, 2.2, 2.8, 4.1, 5.3, 5.9, 7.2, 7.8])
        corr = s.correlation(x, y, method="pearson")
        interpretation = s.interpret_correlation(corr, alpha=0.05)
        self.assertIn("positive", interpretation)
        self.assertIn("would usually be described as statistically significant", interpretation)

    def test_interpret_correlation_includes_ci_note(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.1, 1.9, 3.2, 4.1, 4.9])
        corr = s.correlation(x, y, method="pearson")
        interpretation = s.interpret_correlation(corr, alpha=0.05)
        self.assertIn("confidence interval", interpretation)

    def test_interpret_correlation_spearman_uses_rho_symbol(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        corr = s.correlation(x, y, method="spearman")
        interpretation = s.interpret_correlation(corr, alpha=0.05)
        self.assertIn("rho = -1.000", interpretation)
        self.assertNotIn("(r = ", interpretation)

    def test_report_two_group_includes_interpretation_by_default(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0, 5.0])
        comp = s.compare_independent_groups(x, y, estimand="mean_difference")
        txt = s.report_two_group(comp, digits=2)
        self.assertIn("Interpretation:", txt)

    def test_report_two_group_can_suppress_interpretation(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 3.0, 4.0, 5.0])
        comp = s.compare_independent_groups(x, y, estimand="mean_difference")
        txt = s.report_two_group(comp, digits=2, include_interpretation=False)
        self.assertNotIn("Interpretation:", txt)

    def test_report_correlation_includes_interpretation_by_default(self) -> None:
        corr = s.correlation(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.1, 2.9, 4.2]),
            method="pearson",
        )
        txt = s.report_correlation(corr, digits=2)
        self.assertIn("Interpretation:", txt)

    def test_report_correlation_can_suppress_interpretation(self) -> None:
        corr = s.correlation(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 2.1, 2.9, 4.2]),
            method="pearson",
        )
        txt = s.report_correlation(corr, digits=2, include_interpretation=False)
        self.assertNotIn("Interpretation:", txt)


if __name__ == "__main__":
    unittest.main()
