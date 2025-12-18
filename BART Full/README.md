
# BART Full Analysis

## Overview
This directory contains the results of SHAP analysis on the full dataset and subsequent BART hyperparameter optimization on the top 30 features.

## Data Source
- Input File: `Data/raw/IMPUTED_DATA_WITH REDUCED_columns_21_09_2025.xlsx`
- Target Variable: `f1_bw`

## Methodology
1. **SHAP Analysis**:
   - Model: XGBoost Regressor
   - Features: All available columns (excluding target and identifiers like `row_index`)
   - Outcome: Feature importance ranking based on mean absolute SHAP values.

2. **Top 30 Features Selection**:
   - The top 30 features were selected based on the SHAP analysis.

3. **BART Hyperparameter Optimization**:
   - Model: BART (using `bartpy` via `sklearn` wrapper)
   - Features: Top 30 features from SHAP
   - Optimization: RandomizedSearchCV (CV=3)
   - Metric: RMSE

## Files
- `shap_feature_importance_full.csv`: Full ranking of features.
- `shap_summary_plot_full.png`: SHAP summary plot.
- `bart_optimization_results.json`: Best hyperparameters and metrics.
- `bart_actual_vs_predicted.png`: Plot of actual vs predicted values for the best model.
