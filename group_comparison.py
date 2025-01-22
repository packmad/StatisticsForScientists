#!/usr/bin/env python3

import statistics
import numpy as np
from scipy.stats import shapiro, ttest_ind, pearsonr, levene
from typing import Optional


def calculate_statistics(data, dec_num: int = 2):
    assert len(data) > 1
    avg = round(statistics.mean(data), dec_num)
    stddev = round(statistics.stdev(data), dec_num)
    median = round(statistics.median(data), dec_num)
    max_value = round(max(data), dec_num)
    min_value = round(min(data), dec_num)
    print(f"Avg: {avg}, StDev: {stddev}, Med: {median}, Max: {max_value}, Min: {min_value}")


def is_data_normally_distributed(data, alpha: float = 0.05) -> bool:
    s, p = shapiro(data)  # Null hypothesis: data was drawn from a normal distribution
    if p > alpha:
        print("Fail to reject the null hypothesis: Data appears to be normally distributed.")
        return True
    print("Reject the null hypothesis: Data is not normally distributed.")
    return False


def cohen_effect_size(group1, group2):
    mean_diff = statistics.mean(group1) - statistics.mean(group2)
    n1 = len(group1)
    n2 = len(group2)
    s_pooled = np.sqrt(((n1 - 1) * statistics.stdev(group1)**2 + (n2 - 1) * statistics.stdev(group2)**2) / (n1 + n2 - 2))
    d = mean_diff / s_pooled
    if n1 < 21 or n2 < 21:
        print(f"Small sample size group1={n1} group2={n2} -> Adjusting Cohen's d with Hedges' g")
        d = d * (1 - (3 / (4 * (n1 + n2) - 9)))
        interpretation = 'g'
    else:
        interpretation = 'd'
    interpretation += f"={d} indicates a "
    if abs(d) < 0.2:
        interpretation += "negligible"
    elif abs(d) < 0.5:
        interpretation += "small"
    elif abs(d) < 0.8:
        interpretation += "medium"
    else:
        interpretation += "large"
    interpretation += " effect size."
    print(interpretation)


def equal_variances(group1, group2, alpha: float = 0.05) -> bool:
    if is_data_normally_distributed(group1) and is_data_normally_distributed(group2):
        levene_center = 'mean'  # Recommended for symmetric, moderate-tailed distributions
    else:
        levene_center = 'median'  # AKA Brown-Forsythe test - Recommended for skewed (non-normal) distributions
    stat, p_value = levene(group1, group2, center=levene_center)  # Null hypothesis: populations with equal variances
    if p_value < alpha:
        return False  # Variances are significantly different.
    else:
        return True  # Variances are NOT significantly different.


def independent_ttest(group1, group2, alpha: float = 0.05) -> Optional[bool]:
    if is_data_normally_distributed(group1) and is_data_normally_distributed(group2):
        if equal_variances(group1, group2):
            print("Variances are significantly different -> Using Welch's t-test.")
            equal_var = False
        else:
            print("Variances are NOT significantly different -> Using standard t-test.")
            equal_var = True
        stat, p_value = ttest_ind(group1, group2, equal_var=equal_var)  # Null hypothesis: means of the two groups are equal
        print(f"t-statistic: {stat}, p-value: {p_value}")
        if p_value >= alpha:
            print("Fail to reject the null hypothesis: NO significant difference between groups.")
            return False
        else:
            print("Reject the null hypothesis: Significant difference between groups.")
            cohen_effect_size(group1, group2)
            return True
    print('Data is not normally distributed')
    return None


def pearson_correlation(group1, group2, alpha: float = 0.05) -> Optional[bool]:
    assert len(group1) == len(group2)
    if is_data_normally_distributed(group1) and is_data_normally_distributed(group2):
        r, p_value = pearsonr(group1, group2)
        print(f"Pearson Correlation = {r}, p-value = {p_value}")
        if p_value >= alpha:
            print("NO significant correlation between variables.")
            return False
        else:
            print("Significant correlation between variables.")
            if abs(r) < 0.1:
                interpretation = "small"
            elif abs(r) < 0.3:
                interpretation = "medium"
            elif abs(r) < 0.5:
                interpretation = "large"
            elif abs(r) <= 1.0:
                interpretation = "very large"
            print(f"According to Jacob Cohen, in the context of the social sciences, {interpretation} effect size.")
            return True
    print('Data is not normally distributed')
    return None


dist1 = np.random.normal(21, 8, 32)
dist2 = np.random.normal(42, 10, 32)
independent_ttest(dist1, dist2)

print()
dist3 = np.arange(10, 20)
dist4 = np.array([5, 8, 12, 13, 18, 25, 40, 56, 66, 86])
pearson_correlation(dist3, dist4)
