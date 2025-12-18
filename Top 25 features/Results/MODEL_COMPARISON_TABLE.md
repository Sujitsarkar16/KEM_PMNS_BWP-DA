# Model Performance Comparison

Comparison of models across different feature sets: PMNS (20 variables), Top 25 (25 variables), and Optimized (15 variables).

| Category | Model | Variables | RMSE (g) | MAE (g) | R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PMNS (20 vars)** | MLE Baseline | 20 | 240.2491 | 190.6986 | 0.6532 |
| **PMNS (20 vars)** | MLE Optimized | 20 | 240.2498 | 190.6985 | 0.6532 |
| **PMNS (20 vars)** | RF Baseline | 20 | 251.4472 | 202.8665 | 0.6202 |
| **PMNS (20 vars)** | RF Optimized | 20 | 230.2190 | 184.2510 | 0.6816 |
| **PMNS (20 vars)** | XGBoost Baseline | 20 | 243.6140 | 196.3289 | 0.6435 |
| **PMNS (20 vars)** | XGBoost Optimized | 20 | 218.8982 | 172.4843 | 0.7121 |
| **Top 25 (25 vars)** | MLE Baseline | 25 | 252.4124 | 195.0050 | 0.6172 |
| **Top 25 (25 vars)** | MLE Optimized | 25 | 252.4241 | 195.0169 | 0.6172 |
| **Top 25 (25 vars)** | RF Baseline | 25 | 241.8273 | 192.5762 | 0.6487 |
| **Top 25 (25 vars)** | RF Optimized | 25 | 240.9231 | 194.6631 | 0.6513 |
| **Top 25 (25 vars)** | XGBoost Baseline | 25 | 260.4571 | 208.5950 | 0.5924 |
| **Top 25 (25 vars)** | XGBoost Optimized | 25 | 234.3555 | 189.9049 | 0.6700 |
| **Optimized (15 vars)** | MLE Baseline | 15 | 251.8498 | 199.1213 | 0.6189 |
| **Optimized (15 vars)** | MLE Optimized | 15 | 251.7600 | 198.7403 | 0.6192 |
| **Optimized (15 vars)** | RF Baseline | 15 | 249.2036 | 201.8249 | 0.6269 |
| **Optimized (15 vars)** | RF Optimized | 15 | 237.1542 | 190.7727 | 0.6621 |
| **Optimized (15 vars)** | XGBoost Baseline | 15 | 263.3269 | 218.4116 | 0.5834 |
| **Optimized (15 vars)** | XGBoost Optimized | 15 | 233.6122 | 187.3339 | 0.6721 |


**Best Performing Model:** PMNS (20 vars) - XGBoost Optimized (RMSE: 218.8982)
