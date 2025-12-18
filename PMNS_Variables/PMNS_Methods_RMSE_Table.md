# PMNS Variables: Methods Performance Comparison

This document provides a comprehensive comparison of all machine learning methods applied to the PMNS (Pune Maternal Nutrition Study) dataset with 20 variables, showing both baseline and optimized RMSE (Root Mean Squared Error) values on the test set.

## Performance Metrics Table

| Method | Baseline RMSE (g) | Optimized RMSE (g) | Improvement (g) | Improvement (%) | Baseline R² | Optimized R² |
|--------|-------------------|---------------------|-----------------|-----------------|-------------|--------------|
| **XGBoost** | 243.61 | **218.90** | 24.71 | 10.14% | 0.6435 | **0.7121** |
| **Random Forest** | 251.45 | 230.22 | 21.23 | 8.44% | 0.6202 | 0.6816 |
| **BART** | 231.43 | 227.03 | 4.40 | 1.90% | 0.6782 | 0.6903 |
| **MLE** | 240.25 | 240.25 | 0.00 | 0.00% | 0.6532 | 0.6532 |

## Detailed Results

### 1. XGBoost
- **Baseline RMSE**: 243.61 g
- **Optimized RMSE**: 218.90 g ⭐ **Best Overall Performance**
- **Improvement**: 24.71 g (10.14%)
- **Optimized Hyperparameters**:
  - subsample: 0.6
  - n_estimators: 100
  - min_child_weight: 7
  - max_depth: 4
  - learning_rate: 0.05
  - gamma: 0.3
  - colsample_bytree: 0.8

### 2. Random Forest
- **Baseline RMSE**: 251.45 g
- **Optimized RMSE**: 230.22 g
- **Improvement**: 21.23 g (8.44%)
- **Optimized Hyperparameters**:
  - n_estimators: 400
  - min_samples_split: 2
  - min_samples_leaf: 1
  - max_features: 0.5
  - max_depth: null
  - bootstrap: true

### 3. BART (Bayesian Additive Regression Trees)
- **Baseline RMSE**: 231.43 g
- **Optimized RMSE**: 227.03 g
- **Improvement**: 4.40 g (1.90%)
- **Optimized Hyperparameters**:
  - n_trees: 100
  - n_samples: 1000
  - n_burn: 300
  - beta: 1.5
  - alpha: 0.9

### 4. MLE (Maximum Likelihood Estimation)
- **Baseline RMSE**: 240.25 g
- **Optimized RMSE**: 240.25 g
- **Improvement**: 0.00 g (0.00%)
- **Note**: MLE showed minimal improvement with hyperparameter optimization, indicating the baseline parameters were already near-optimal for this method.

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Best Overall RMSE** | 218.90 g (XGBoost Optimized) |
| **Best Baseline RMSE** | 231.43 g (BART Baseline) |
| **Largest Improvement** | 24.71 g (XGBoost: 10.14%) |
| **Average Baseline RMSE** | 241.69 g |
| **Average Optimized RMSE** | 229.10 g |
| **Average Improvement** | 12.59 g (5.21%) |

## Key Findings

1. **XGBoost Optimized** achieved the best performance with an RMSE of **218.90 g** and R² of **0.7121**, representing a 10.14% improvement over its baseline.

2. **Random Forest Optimized** showed the second-best performance with an RMSE of **230.22 g**, improving by 8.44% from baseline.

3. **BART** had the best baseline performance (231.43 g) but showed minimal improvement with optimization (1.90%), suggesting the baseline parameters were already well-tuned.

4. **MLE** showed no improvement with optimization, indicating that the default EM algorithm parameters were already optimal for this dataset.

5. **Tree-based ensemble methods** (XGBoost and Random Forest) showed the largest improvements with hyperparameter optimization, while **Bayesian methods** (BART) and **classical statistical methods** (MLE) showed more limited gains.

## Dataset Information

- **Dataset**: PMNS (Pune Maternal Nutrition Study) Variables
- **Number of Features**: 20 variables
- **Test Set Size**: 159 samples
- **Target Variable**: Birth weight (grams)
- **Evaluation Metric**: RMSE (Root Mean Squared Error) on test set

## Notes

- All RMSE values are reported in grams
- R² values represent the coefficient of determination
- Test set performance is reported to ensure fair comparison across methods
- Hyperparameter optimization was performed using cross-validation
- Results generated on: 2025-12-06

---

**Generated**: 2025-12-06  
**Dataset**: PMNS Variables (20 features)  
**Test Set**: 159 samples
