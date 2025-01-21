# Statistics For Scientists


---
### Independent vs. Dependent Variable

| Feature        | Independent Variable                             | Dependent Variable                         |
| -------------- | ------------------------------------------------ | ------------------------------------------ |
| **Definition** | The variable that is manipulated or categorized. | The variable that is measured or observed. |
| **Role**       | Acts as the cause or predictor.                  | Acts as the effect or outcome.             |
| **Examples**   | Dosage of medication, teaching method, age.      | Recovery time, test scores, weight.        |
### Examples:

- **In an Experiment**:
    - A researcher studies the effect of different doses of a drug (independent variable) on patient recovery time (dependent variable).
- **In an Observational Study**:
    - A study investigates how education level (independent variable) influences income (dependent variable).

--- 
# p-values

A **p-value** is a statistical measure that helps researchers determine the significance of their results in hypothesis testing. It provides the probability of observing the data, or something more extreme, under the assumption that the null hypothesis is true.

### Key Points About p-values:

1. **Null Hypothesis ($H_0$​)**:
    
    - The null hypothesis typically states that there is no effect or no difference between groups.
    - Example: In a drug study, $H_0$​ could state that the drug has no effect compared to a placebo.
2. **Interpreting p-values**:
    
    - A small p-value (e.g., $p<0.05p$) suggests that the observed data is unlikely under $H_0$, leading to a rejection of the null hypothesis.
    - A large p-value (e.g., $p \geq 0.05$) suggests that the observed data is consistent with $H_0$​, and there is insufficient evidence to reject it.
3. **Threshold (α)**:
    
    - The threshold for statistical significance ($\alpha$) is often set at 0.05. This means there is a 5% chance of rejecting $H_0$​ when it is actually true (Type I error).
    - Researchers can choose different thresholds, depending on the field or the study's rigor.
4. **Misinterpretations**:
    
    - A p-value does **not** measure the probability that $H_0$​ is true.
    - A p-value does **not** indicate the size or importance of an effect.

---

# Tests for Normality

Some tests assume a normal distribution, so there is a need of objective measures of whether a dataset deviates significantly from normality.
- **For Small Datasets**: Use the Shapiro-Wilk test.
- **For Larger Datasets**: Anderson-Darling and/or D’Agostino-Pearson test.

### Shapiro-Wilk Test
- Tests the null hypothesis that the data is normally distributed.
- Suitable for small to medium-sized datasets.

```
from scipy.stats import shapiro

stat, p = shapiro(data)
print(f"Shapiro-Wilk Test: W-statistic = {stat}, p-value = {p}")

if p > 0.05:
    print("Data appears to be normally distributed (fail to reject H0).")
else:
    print("Data is not normally distributed (reject H0).")
```

### Anderson-Darling Test
- Provides critical values for different significance levels to test normality.

```
from scipy.stats import anderson

result = anderson(data, dist='norm')
print("Anderson-Darling Test Statistic:", result.statistic)
for i, cv in enumerate(result.critical_values):
    sig_level = result.significance_level[i]
    print(f"At {sig_level}% significance level: Critical Value = {cv}")
if result.statistic < result.critical_values[2]:  # Typically use 5% level
    print("Data appears to be normally distributed.")
else:
    print("Data is not normally distributed.")

```

### D’Agostino and Pearson’s Test
- Tests for skewness and kurtosis to assess normality.

```
from scipy.stats import normaltest

stat, p = normaltest(data)
print(f"D’Agostino and Pearson Test: Statistic = {stat}, p-value = {p}")

if p > 0.05:
    print("Data appears to be normally distributed (fail to reject H0).")
else:
    print("Data is not normally distributed (reject H0).")

```

---

# **Student's t-test**


### Purpose:

Used to determine if there is a significant difference between the means of two groups.

### Types:

- **Independent t-test**: Compares means of two independent/unrelated groups.
	- Examples:
		- Comparing test scores of students from two different schools.
		- Comparing the effect of two different treatments on two separate groups of patients.
	-  **Assumptions**:
	    1. Observations in each group are independent.
	    2. Data in each group are normally distributed (especially for small sample sizes).
	    3. Both groups have equal variances (can be tested using Levene's test).
	- **Hypotheses**:
	    - Null hypothesis ($H_0$​): The means of the two groups are equal ($\mu_1 = \mu_2$​).
	    - Alternative hypothesis ($H_a$​): The means of the two groups are not equal ($\mu_1 \neq \mu_2$​).
    
- **Paired t-test**: Compares means of the same group at different times.
    - Examples:
		- Measuring blood pressure before and after treatment on the same patients.
		- Comparing test scores of the same students before and after a training session.
	- - **Assumptions**:
	    1. Observations are dependent (paired by subject or condition).
	    2. Differences between paired observations are normally distributed.
	- **Hypotheses**:
	    - Null hypothesis ($H_0$​): The mean difference between the paired samples is zero ($\mu_d = 0$).
	    - Alternative hypothesis ($H_a$​): The mean difference is not zero ($\mu_d \neq 0$).

| Aspect                  | Independent t-Test                               | Paired t-Test                                |
|-------------------------|--------------------------------------------------|---------------------------------------------|
| **Groups**              | Two unrelated groups.                           | Two related groups or repeated measures.    |
| **Examples**            | Control vs. experimental group.                 | Pre-treatment vs. post-treatment in the same group. |
| **Data Structure**      | Each group has separate measurements.           | Measurements are paired (e.g., before/after for the same individual). |
| **Hypotheses**          | Compares means of two groups.                   | Compares the mean of differences.           |
| **Error Reduction**     | More variation due to independent observations. | Less variation since differences are within subjects. |
| **Statistical Power**   | May require larger sample sizes.                | More powerful for detecting small effects.  |

### Python Example:

```
from scipy.stats import ttest_ind, ttest_rel
import numpy as np

# Independent t-test
group1 = np.random.normal(50, 10, 30)
group2 = np.random.normal(55, 10, 30)
stat, p = ttest_ind(group1, group2)
print("Independent t-test: p-value =", p)

# Paired t-test
before = np.random.normal(50, 10, 30)
after = before + np.random.normal(0, 5, 30)
stat, p = ttest_rel(before, after)
print("Paired t-test: p-value =", p)
```

---

# **Analysis of Variance (ANOVA)**

### Purpose:

Used to compare means of three or more groups.
### Types:

- **One-way ANOVA**: One independent variable.
- **Two-way ANOVA**: Two independent variables.
    
### Python Example (One-way ANOVA):

```
from scipy.stats import f_oneway
import numpy as np

group1 = np.random.normal(50, 10, 30)
group2 = np.random.normal(55, 10, 30)
group3 = np.random.normal(60, 10, 30)
stat, p = f_oneway(group1, group2, group3)
print("One-way ANOVA: p-value =", p)
```


### Python Example (Two-way ANOVA):

A researcher wants to study the effect of **teaching method** (Factor A) and **gender** (Factor B) on students' test scores (dependent variable).

```
import pandas as pd import statsmodels.api as sm from statsmodels.formula.api import ols  

data = {     'Score': [85, 78, 92, 88, 91, 95, 72, 68, 79, 83, 88, 85,               81, 75, 86, 89, 90, 94, 70, 65, 76, 82, 87, 84],     'Method': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'B', 'B',                'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'A', 'A', 'B', 'B'],     'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female',                'Male', 'Female', 'Male', 'Female', 'Male', 'Female',                'Male', 'Female', 'Male', 'Female', 'Male', 'Female',                'Male', 'Female', 'Male', 'Female', 'Male', 'Female'] }  
df = pd.DataFrame(data)
model = ols('Score ~ C(Method) + C(Gender) + C(Method):C(Gender)', data=df).fit() 
anova_table = sm.stats.anova_lm(model, typ=2)  print(anova_table)`
```

The output is an ANOVA table that includes:

- **Sum of Squares (SS)**: Variation explained by each factor.
- **Degrees of Freedom (df)**: Number of levels minus 1.
- **F-statistic**: Test statistic for the effect.
- **p-value**: Indicates the significance of each factor and their interaction.

Example Output:
```
                       sum_sq    df          F        PR(>F)
C(Method)         720.666667   1.0  30.300000  0.000125
C(Gender)          96.000000   1.0   4.033333  0.058746
C(Method):C(Gender) 16.666667   1.0   0.700000  0.414001
Residual          380.000000  20.0        NaN       NaN

```

Interpretation:
1. **Main Effects**:
    - `C(Method)`: Significant effect (p<0.05p < 0.05p<0.05), meaning teaching methods affect scores.
    - `C(Gender)`: Not significant (p>0.05p > 0.05p>0.05), meaning gender does not affect scores.
2. **Interaction Effect**:
    - `C(Method):C(Gender)`: Not significant (p>0.05p > 0.05p>0.05), meaning no interaction between teaching method and gender.

---

# **Effect Size** 

Wffect size is a measure of the magnitude of a phenomenon or the strength of a relationship in the population.

While inferential tests (like t-tests, ANOVA, or p-values) assess whether an effect exists, effect sizes provide additional context by indicating how meaningful that effect is. 

For differences between groups:

1) Cohen’s $d$
    - Measures the standardized mean difference between two groups.
    - Interpretation (Cohen’s guidelines):
        - Small: 0.2
        - Medium: 0.5
        - Large: 0.8
2) Hedges’ $g$
    - Similar to Cohen’s $d$ but adjusts for small sample sizes.


---

# **Correlation Tests**

### Purpose:

Measure the relationship between two variables.

### Types:

- **Pearson correlation**: Linear relationship between two continuous variables.
	- **Assumptions**:  
	    1. Both variables are continuous and normally distributed.
	    2. The relationship between variables is linear.
	    3. Outliers can distort results.
	- **Examples**:
	    - Measuring the relationship between height and weight.
	    - Checking the linear relationship between temperature and ice cream sales.
    - **Range**: from -1 to 1
        - $r=1$: Perfect positive linear correlation.
        - $r=−1$: Perfect negative linear correlation.
        - $r=0$: No linear relationship.
		-  Jacob Cohen gives the following guidelines for the social sciences:
			- Small $r < 0.10$
			- Medium	$r < 0.30$
			- Large $r < 0.50$
```
@book{cohen2013statistical,
  title={Statistical power analysis for the behavioral sciences},
  author={Cohen, Jacob},
  year={2013},
  publisher={routledge}
}
```

- **Spearman correlation**: Monotonic relationship (rank-based).
    
- **Kendall's tau**: Strength of association between two ranked variables.
    

```
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np

x = np.random.normal(50, 10, 30)
y = x + np.random.normal(0, 5, 30)

# Pearson correlation
stat, p = pearsonr(x, y)
print("Pearson correlation: p-value =", p)

# Spearman correlation
stat, p = spearmanr(x, y)
print("Spearman correlation: p-value =", p)

# Kendall's tau
stat, p = kendalltau(x, y)
print("Kendall's tau: p-value =", p)
```

---

# **Multiple Testing Corrections**

### Purpose:

When performing multiple statistical tests, the likelihood of incorrectly rejecting at least one null hypothesis increases. Controlling the False Discovery Rate (FDR) is essential to reduce the risk of making too many false discoveries.

### Methods:

- **Bonferroni correction**: Adjusts p-values by multiplying by the number of tests. Careful: it's very strict; useful when avoiding any false positives is critical.
    
- **Benjamini-Hochberg procedure**: Controls the false discovery rate.
    
### Python Example:

```
from statsmodels.stats.multitest import multipletests
import numpy as np

# Example p-values from multiple tests
p_values = np.random.uniform(0, 0.05, 10)

# Bonferroni correction
reject, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
print("Bonferroni corrected p-values:", p_corrected)

# Benjamini-Hochberg procedure
reject, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
print("Benjamini-Hochberg corrected p-values:", p_corrected)
```

---

