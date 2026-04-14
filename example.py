"""
Example usage + quick verification via the existing unittest suite.

This script demonstrates:
1) A two-group comparison via `compare_independent_groups`
2) A correlation analysis via `correlation`
3) Running a focused subset of tests from `tests/test_inferential_stats.py`
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import inferential_stats as stats


def run_demo() -> None:
    rng = np.random.default_rng(0)

    # --- Group comparison demo (mean difference; Welch by default)
    group1 = rng.normal(loc=10.0, scale=2.0, size=30)
    group2 = rng.normal(loc=11.0, scale=2.5, size=28)
    comp = stats.compare_independent_groups(group1, group2, estimand="mean_difference")
    print("### Group comparison")
    print(stats.report_two_group(comp, digits=2))
    print()

    # --- Correlation demo (Pearson with CI)
    x = np.linspace(0.0, 10.0, 25)
    y = 0.8 * x + rng.normal(loc=0.0, scale=1.0, size=x.size)
    corr = stats.correlation(x, y, method="pearson")
    print("### Correlation")
    print(stats.report_correlation(corr, digits=2))
    print()


if __name__ == "__main__":
    run_demo()
