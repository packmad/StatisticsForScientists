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

# sawilowsky2009new
def effect_size_interpretation_sawilowsky(effect_size: float, dec_num: int = 2) -> str:
    interpretation = f"={round(effect_size, dec_num)} -> Sawilowsky guidelines suggest a "
    if abs(effect_size) < 0.01:
        interpretation += "very small"
    elif abs(effect_size) < 0.2:
        interpretation += "small"
    elif abs(effect_size) < 0.5:
        interpretation += "medium"
    elif abs(effect_size) < 0.8:
        interpretation += "large"
    elif abs(effect_size) < 1.2:
        interpretation += "very large"
    else:
        interpretation += "huge"
    interpretation += " effect size."
    return interpretation


# funder2019evaluating
def effect_size_interpretation_funder_and_ozer(effect_size: float, dec_num: int = 2) -> str:
    interpretation = f"={round(effect_size, dec_num)} -> Funder&Ozer guidelines suggest a "
    if abs(effect_size) < 0.1:
        interpretation += "small"
    elif abs(effect_size) < 0.2:
        interpretation += "medium"
    elif abs(effect_size) < 0.8:
        interpretation += "large"
    else:
        interpretation += "very large"
    interpretation += " effect size."
    return interpretation


def cohen_d(group1, group2):
    mean_diff = statistics.mean(group1) - statistics.mean(group2)
    n1 = len(group1)
    n2 = len(group2)
    s_pooled = np.sqrt(((n1 - 1) * statistics.stdev(group1)**2 + (n2 - 1) * statistics.stdev(group2)**2) / (n1 + n2 - 2))
    d = mean_diff / s_pooled
    if n1 < 21 or n2 < 21:
        print(f"Small sample size group1={n1} group2={n2} -> Adjusting Cohen's d with Hedges' g")
        d = d * (1 - (3 / (4 * (n1 + n2) - 9)))
        interpretation = 'g'  # hedges1981distribution
    else:
        interpretation = 'd'  # cohen2013statistical
    interpretation += effect_size_interpretation_funder_and_ozer(d)
    print(interpretation)


# gastwirth2009impact
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
            cohen_d(group1, group2)
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
            print(f"r{effect_size_interpretation_funder_and_ozer(r)}")
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
