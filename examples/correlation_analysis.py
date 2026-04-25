"""
Example: measuring the association between two continuous variables.

Demonstrates both correlation methods available in stats4science:
  - pearson  : linear association + Fisher z confidence interval
  - spearman : monotonic association + bootstrap confidence interval

Run with:
    uv run python examples/correlation_analysis.py
"""

import stats4science as stats

# Study hours and exam scores for 15 students.
study_hours = [2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 12]
exam_scores = [58, 66, 59, 69, 70, 66, 78, 72, 82, 75, 86, 83, 90, 88, 92]

# --- Pearson: are the variables linearly associated? ---
pearson_result = stats.correlation(study_hours, exam_scores, method="pearson")
print("=== Pearson correlation ===")
print(stats.report_correlation(pearson_result, digits=2))

print()

# --- Spearman: is the association at least monotonic? ---
spearman_result = stats.correlation(study_hours, exam_scores, method="spearman")
print("=== Spearman correlation ===")
print(stats.report_correlation(spearman_result, digits=2))
