# MLE Procedural Notebook with Evaluation - Complete Explanation

## Overview

The `mle_procedural_notebook_with_evaluation.ipynb` implements a **Maximum Likelihood Estimation (MLE) pipeline** for handling missing data in multivariate continuous variables using the **Expectation-Maximization (EM) algorithm**. This notebook is designed to work with birth weight and maternal health data, specifically focusing on 5 key continuous variables.

## Data Structure and Variables

### Key Variables Analyzed
The notebook focuses on 5 critical continuous variables:
1. **`f1_bw`** - F1 birth weight (grams) - **Primary outcome variable**
2. **`f0_m_age`** - F0 maternal age (years)
3. **`f0_m_bmi_prepreg`** - F0 maternal BMI pre-pregnancy (kg/m²)
4. **`f0_m_ht`** - F0 maternal height (cm)
5. **`f0_m_wt_prepreg`** - F0 maternal weight pre-pregnancy (kg)

### Dataset Characteristics
- **791 rows** (individuals/observations)
- **856 total columns** (707 continuous + 148 categorical + 1 ID)
- **Missing data pattern**: Only 1 variable has >20% missing data, most have <5% missing

## Step-by-Step Pipeline Explanation

### Step 1: Data Loading and Setup
```python
data_df, variable_groups = load_data(DATA_PATH, VAR_GROUP_PATH)
```

**What it does:**
- Loads the cleaned dataset with engineered features
- Loads variable classification table (Continuous vs Categorical)
- Validates file existence and reports data dimensions

**Why it's important:**
- Ensures data integrity before analysis
- Provides variable type classification for proper statistical modeling
- Sets up the foundation for all subsequent analysis

### Step 2: Variable Classification and Missing Data Analysis
```python
continuous_vars, categorical_vars = classify_variables(data_df, variable_groups)
analyze_missing_patterns(data_df)
```

**What it does:**
- Separates variables into continuous (707) and categorical (148) types
- Analyzes missing data patterns and creates visualizations
- Generates missingness heatmap for key variables

**Why it's important:**
- Determines which variables are suitable for multivariate normal modeling
- Identifies missing data patterns that could affect imputation quality
- Helps understand data quality before modeling

### Step 3: Normality Testing and Model Specification
```python
normality = test_normality(data_df)
model_specs = define_model_specs(continuous_vars, categorical_vars, normality)
```

**What it does:**
- Tests normality of key continuous variables using Shapiro-Wilk test
- All 5 key variables are **non-normal** (p < 0.05)
- Defines model specifications for multivariate normal distribution

**Why it's important:**
- Despite non-normality, multivariate normal assumption is still used (common practice)
- Provides framework for joint likelihood modeling
- Sets up parameters for EM algorithm

### Step 4: Data Preparation for Analysis
```python
analysis_vars = [v for v in DEFAULT_KEY_CONTINUOUS if v in data_df.columns]
data_continuous, analysis_vars = prepare_continuous_matrix(data_df, analysis_vars)
```

**What it does:**
- Selects the 5 key continuous variables for analysis
- Creates a 791×5 matrix of continuous data
- Handles missing values (NaN) in the matrix

**Why it's important:**
- Focuses analysis on most clinically relevant variables
- Prepares data in format required for multivariate normal modeling
- Maintains data structure for EM algorithm

### Step 5: Expectation-Maximization (EM) Algorithm
```python
em_results = simple_em_for_missing(data_continuous, multivariate_normal_log_likelihood, max_iter=100, tol=1e-4)
```

**What it does:**
- **E-step**: Imputes missing values using current parameter estimates
- **M-step**: Updates mean and covariance parameters using imputed data
- Iterates until convergence (tolerance = 1e-4) or max iterations (100)
- **Converged in just 2 iterations** with final log-likelihood = -12,819.66

**Why EM is used:**
- Handles missing data without discarding observations
- Provides maximum likelihood estimates under missing data
- More efficient than complete case analysis
- Accounts for uncertainty in missing values

### Step 6: Parameter Estimation Results

**Mean Vector (Parameter Estimates):**
- `f1_bw`: 2,575.68 grams (birth weight)
- `f0_m_age`: 21.31 years (maternal age)
- `f0_m_bmi_prepreg`: 17.97 kg/m² (maternal BMI)
- `f0_m_ht`: 151.58 cm (maternal height)
- `f0_m_wt_prepreg`: 41.31 kg (maternal weight)

**Covariance Matrix:**
- Captures relationships between all variable pairs
- Diagonal elements represent variances
- Off-diagonal elements represent covariances
- Matrix is positive definite (all eigenvalues > 0)

## Evaluation Methods Explained

### 1. Regular MLE RMSE (Baseline RMSE)

**What it measures:**
- Root Mean Square Error between **observed values** and **model mean estimates**
- Per-variable RMSE comparing actual observed data to the estimated mean

**Formula:**
```
RMSE = √(Σ(observed - model_mean)² / n_observed)
```

**Results:**
- `f0_m_bmi_prepreg`: 2.92 (best fit)
- `f0_m_age`: 3.51
- `f0_m_wt_prepreg`: 6.21
- `f0_m_ht`: 9.49
- `f1_bw`: 405.43 (worst fit - largest scale)
- **Overall RMSE**: 181.39

**Why this matters:**
- Shows how well the model mean represents observed data
- Indicates model fit quality
- Helps identify which variables are most/least predictable

### 2. Masked Entry RMSE (Imputation Validation)

**What it measures:**
- **Validation of imputation quality** by artificially creating missing data
- Tests how well the EM algorithm can recover known values

**Process:**
1. **Mask 5% of observed entries** (randomly selected)
2. **Run EM algorithm** on the artificially masked data
3. **Compare imputed values** to the original (known) values
4. **Calculate RMSE** between imputed and true values

**Results:**
- `f0_m_bmi_prepreg`: 0.13 (excellent imputation)
- `f0_m_wt_prepreg`: 0.80
- `f0_m_ht`: 1.56
- `f0_m_age`: 2.81
- `f1_bw`: 358.07 (challenging to impute)
- **Overall RMSE**: 161.35

## Key Differences: Regular RMSE vs Masked Entry RMSE

### Regular MLE RMSE (181.39)
- **Purpose**: Model fit assessment
- **Comparison**: Observed values vs model mean
- **Interpretation**: How well does the estimated mean represent the data?
- **Scale**: Larger values due to comparing individual observations to mean

### Masked Entry RMSE (161.35)
- **Purpose**: Imputation quality validation
- **Comparison**: Imputed values vs true values
- **Interpretation**: How accurately can the model recover missing data?
- **Scale**: Smaller values because EM uses conditional distributions, not just means

### Why Masked Entry RMSE is Lower (Better)

1. **Conditional Imputation**: EM doesn't just use the mean - it uses conditional distributions based on observed values in the same row
2. **Information Utilization**: The algorithm leverages correlations between variables to make better predictions
3. **Context-Aware**: If you know a mother's height and age, you can better predict her BMI than just using the overall mean

## Clinical Interpretation

### Birth Weight (f1_bw) - 388 grams difference
- **Regular RMSE**: 405.43 grams
- **Masked RMSE**: 358.07 grams
- **Difference**: ~47 grams improvement with conditional imputation
- **Clinical significance**: This represents a meaningful improvement in birth weight prediction accuracy

### Why Birth Weight is Hardest to Predict
1. **Largest variance**: Birth weight has the highest variability among all variables
2. **Complex relationships**: Influenced by many unmeasured factors (genetics, nutrition, health conditions)
3. **Non-linear relationships**: May not follow simple linear patterns with maternal characteristics

## Technical Implementation Details

### EM Algorithm Convergence
- **Converged in 2 iterations** (very fast)
- **Tolerance**: 1e-4 (high precision)
- **Final log-likelihood**: -12,819.66
- **Convergence indicates**: Stable parameter estimates

### Matrix Operations
- **Covariance regularization**: Adds 1e-6 to diagonal for numerical stability
- **Conditional mean calculation**: Uses matrix inversion for optimal imputation
- **Positive definiteness**: Ensures valid covariance matrix

### Missing Data Handling
- **Pattern**: Most variables have <5% missing data
- **Strategy**: EM algorithm preserves all observations
- **Quality**: Masked evaluation shows good imputation performance

## Practical Applications

### For Research
- **Complete dataset**: All 791 observations can be used for analysis
- **Uncertainty quantification**: EM provides parameter estimates with proper uncertainty
- **Model validation**: Masked evaluation confirms imputation quality

### For Clinical Practice
- **Risk assessment**: Better birth weight predictions using maternal characteristics
- **Missing data handling**: Robust method for incomplete medical records
- **Population studies**: Enables analysis of datasets with missing values

## Summary

This MLE procedural notebook provides a comprehensive solution for handling missing data in multivariate continuous variables. The EM algorithm successfully estimates parameters while preserving all observations, and the masked evaluation confirms that the imputation quality is good, with particularly strong performance for maternal characteristics and reasonable performance for birth weight prediction. The 388-gram difference between regular and masked RMSE demonstrates the value of using conditional distributions rather than simple means for imputation.
