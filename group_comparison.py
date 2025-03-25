#!/usr/bin/env python3

import statistics
import numpy as np

from typing import Optional
from scipy.stats import shapiro, ttest_ind, pearsonr, levene, mannwhitneyu


def calculate_statistics(data, dec_num: int = 2):
    assert len(data) > 1
    avg = round(statistics.mean(data), dec_num)
    stddev = round(statistics.stdev(data), dec_num)
    median = round(statistics.median(data), dec_num)
    max_value = round(max(data), dec_num)
    min_value = round(min(data), dec_num)
    print(f"Avg: {avg}, StDev: {stddev}, Med: {median}, Max: {max_value}, Min: {min_value}")


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


def is_data_normally_distributed(data, alpha: float = 0.05) -> bool:
    assert 6 < len(data) < 50  # Not recommended otherwise
    s, p = shapiro(data)  # Null hypothesis: data was drawn from a normal distribution
    if p > alpha:
        #print("Fail to reject the null hypothesis: Data appears to be normally distributed.")
        return True
    #print("Reject the null hypothesis: Data is not normally distributed.")
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


# cliff1993dominance
def cliffs_delta(group1, group2):
    # Measure of how often values in one distribution
    # tend to be larger than values in another distribution.
    x = np.array(group1)
    y = np.array(group2)
    Nx = len(x)
    Ny = len(y)
    bigger = 0
    smaller = 0
    for i in range(Nx):
        for j in range(Ny):
            if x[i] > y[j]:
                bigger += 1
            elif x[i] < y[j]:
                smaller += 1
    cd = (bigger - smaller) / float(Nx * Ny)
    interpretation: str
    if abs(cd) < 0.147:
        interpretation = 'negligible'
    elif abs(cd) < 0.33:
        interpretation = 'small'
    elif abs(cd) < 0.474:
        interpretation = 'medium'
    else:
        interpretation = 'large'
    print(f"Cliff's Delta: {cd} -> {interpretation} effect size")


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
        print('Data is normally distributed -> Using t-test')
        if equal_variances(group1, group2):
            print("Variances are significantly different -> Using Welch's t-test.")
            equal_var = False
        else:
            print("Variances are NOT significantly different -> Using standard t-test.")
            equal_var = True
        # Null hypothesis: means of the two groups are equal
        stat, p_value = ttest_ind(group1, group2, equal_var=equal_var)
        print(f"t-statistic: {stat}, p-value: {p_value}")
        if p_value >= alpha:
            print("Fail to reject the null hypothesis: suggests NO significant difference between groups.")
            return False
        else:
            print("Reject the null hypothesis: statistically significant difference in the means.")
            cohen_d(group1, group2)
            return True
    return None


def mannwhitneyu_test(group1, group2, alpha: float = 0.05) -> bool:
    # Null hypothesis: The distributions of the two groups are the same with respect to stochastic dominance.
    # Namely, the probability that a random value in one group is larger than a random value in the other.
    # method='auto' -> 'exact' when len(group*) < 9 and there are no ties; chooses 'asymptotic' otherwise
    U_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided', method='auto')
    print(f"Mann–Whitney U statistic: {U_stat}, p-value: {p_value}")
    if p_value >= alpha:
        print("Fail to reject the null hypothesis: suggests NO significant difference between groups.")
        return False
    else:
        print("Reject the null hypothesis: statistically significant difference in the distribution/location.")
        cliffs_delta(group1, group2)
        return True


def compare_groups(group1, group2, alpha: float = 0.05):
    if len(group1) < 3 or len(group2) < 3:
        print(f'Sample size too small: {group1=} {group2=} -> Collect more data!')
        return
    elif len(group1) < 11 or len(group2) < 11:
        print(f'Small sample size" {group1=} {group2=} -> Using Mann–Whitney U Test.')
        result = mannwhitneyu_test(group1, group2, alpha)
    else:
        result = independent_ttest(group1, group2, alpha)
        if result is None:
            print('Data is NOT normally distributed -> Using Mann–Whitney U Test.')
            result = mannwhitneyu_test(group1, group2, alpha)
    if not result:
        print('Failing to reject the null hypothesis means you lack statistical evidence of a difference -> '
              ' it is not proof of "no difference," it SUGGESTS that there is no difference.')


if __name__ == '__main__':
    dist1 = np.random.normal(21, 8, 32)
    dist2 = np.random.normal(42, 10, 32)
    compare_groups(dist1, dist2)

    print()

    dist3 = np.random.uniform(low=1, high=10, size=32)
    compare_groups(dist1, dist3)

    print()

    dist4 = np.arange(10, 20)
    dist5 = np.array([5, 8, 12, 13, 18, 25, 40, 56, 66, 86])
    pearson_correlation(dist4, dist5)
