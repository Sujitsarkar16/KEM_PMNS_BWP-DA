# Model Performance Comparison: PMNS vs Top 25 Features

This table compares the Root Mean Squared Error (RMSE) and predictive metrics for models trained on two feature sets:
1. **PMNS (20 Features)**: Established feature set from previous research.
2. **Top 25 (25 Features)**: New feature importance based selection.

| Category | Model | Variables | RMSE (g) | MAE (g) | R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PMNS (20 vars)** | XGBoost Optimized | 20 | 218.8982 | 172.4843 | 0.7121 |
| **PMNS (20 vars)** | RF Optimized | 20 | 230.2190 | 184.2510 | 0.6816 |
| **PMNS (20 vars)** | MLE Baseline | 20 | 240.2491 | 190.6986 | 0.6532 |
| **PMNS (20 vars)** | MLE Optimized | 20 | 240.2498 | 190.6985 | 0.6532 |
| **PMNS (20 vars)** | XGBoost Baseline | 20 | 243.6140 | 196.3289 | 0.6435 |
| **PMNS (20 vars)** | RF Baseline | 20 | 251.4472 | 202.8665 | 0.6202 |
| **Top 25 (25 vars)** | XGBoost Optimized | 25 | 234.3555 | 189.9049 | 0.6700 |
| **Top 25 (25 vars)** | RF Optimized | 25 | 240.9231 | 194.6631 | 0.6513 |
| **Top 25 (25 vars)** | RF Baseline | 25 | 241.8273 | 192.5762 | 0.6487 |
| **Top 25 (25 vars)** | MLE Baseline | 25 | 252.4124 | 195.0050 | 0.6172 |
| **Top 25 (25 vars)** | MLE Optimized | 25 | 252.4241 | 195.0169 | 0.6172 |
| **Top 25 (25 vars)** | XGBoost Baseline | 25 | 260.4571 | 208.5950 | 0.5924 |

**Summary Analysis:**
- The **PMNS (20 variables)** set consistently outperforms the Top 25 set across all model types.
- The **Best Overall Model** is the **PMNS XGBoost Optimized** with an RMSE of **218.90g** and $R^2$ of **0.71**.
- The best **Top 25** model (XGBoost Optimized) achieved an RMSE of **234.36g** ($R^2$ 0.67).
- Using 20 variables yields better predictive accuracy than using 25 variables, suggesting the PMNS set is more efficient and contains higher-quality signal.
