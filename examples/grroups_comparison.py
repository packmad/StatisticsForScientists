"""
Example: comparing two independent groups.

Demonstrates both estimands available in stats4science:
  - mean_difference  : Welch t-test + Hedges' g
  - stochastic_dominance : Mann-Whitney U + Cliff's delta

Run with:
    uv run python examples/groups_comparison.py
"""

import stats4science as stats

# Reaction times (ms) recorded for two UI interfaces.
interface_a = [398, 410, 405, 392, 430, 415, 401, 389, 418, 407, 395, 423]
interface_b = [412, 439, 421, 445, 433, 427, 416, 438, 450, 429, 420]

# --- Estimand 1: mean difference (how many ms faster on average?) ---
mean_result = stats.compare_independent_groups(
    interface_a,
    interface_b,
    estimand="mean_difference",
)
print("=== Mean difference (Welch t-test) ===")
print(stats.report_two_group(mean_result, digits=2))

print()

# --- Estimand 2: stochastic dominance (how often is A faster than B?) ---
dominance_result = stats.compare_independent_groups(
    interface_a,
    interface_b,
    estimand="stochastic_dominance",
)
print("=== Stochastic dominance (Mann-Whitney U) ===")
print(stats.report_two_group(dominance_result, digits=2))
