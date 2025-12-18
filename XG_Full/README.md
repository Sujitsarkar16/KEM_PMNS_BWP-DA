# XGBoost Full Dataset Training

## Overview
This directory contains the complete XGBoost training pipeline for birth weight prediction using the full dataset with all available variables (851 features).

## Dataset Information
- **Source**: `Data/raw/IMPUTED_DATA_WITH REDUCED_columns_21_09_2025.xlsx`
- **Total Samples**: 791
- **Total Features Used**: 851
- **Target Variable**: `f1_bw` (birth weight in grams)

## Data Leakage Prevention
The following variables were excluded to prevent data leakage and overfitting:
- `f1_bw` - Target variable
- `f1_sex` - Post-birth variable (data leakage)
- `Unnamed: 0` - Index column
- `row_index` - Index column

## Methodology

### 1. Data Preprocessing
- Loaded the full imputed dataset
- Removed non-predictive and data leakage variables
- Handled missing values (imputed with median for numerical features)
- Removed infinite values
- Final feature set: 851 variables

### 2. Train-Test Split
- Training Set: 80% (632 samples)
- Test Set: 20% (159 samples)
- Random State: 42 (for reproducibility)

### 3. Model Training

#### Baseline Model
- Default XGBoost parameters
- Purpose: Establish performance baseline

#### Hyperparameter Optimization
- Method: RandomizedSearchCV
- Number of Iterations: 50
- Cross-Validation: 5-fold
- Optimization Metric: Negative Mean Squared Error
- Hyperparameters Tuned:
  - `n_estimators`: [100, 1000]
  - `max_depth`: [3, 15]
  - `learning_rate`: [0.01, 0.31]
  - `subsample`: [0.6, 1.0]
  - `colsample_bytree`: [0.6, 1.0]
  - `min_child_weight`: [1, 10]
  - `gamma`: [0, 0.5]
  - `reg_alpha`: [0, 1]
  - `reg_lambda`: [0, 1]

### 4. Model Evaluation
- **Metrics Used**:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (Coefficient of Determination)
- **Cross-Validation**: 5-fold CV on entire dataset

### 5. SHAP Analysis
SHAP (SHapley Additive exPlanations) analysis provides interpretable feature importance:
- **SHAP Summary Plot**: Shows feature importance and impact direction
- **SHAP Bar Plot**: Ranks features by mean absolute SHAP value
- **SHAP Dependence Plots**: Individual analysis for top 5 features
- **Sample Size for SHAP**: 500 test samples (for computational efficiency)

## Output Files

### Models
- `xgboost_full_model.pkl` - Trained model (pickle format)
- `xgboost_full_model.json` - Trained model (XGBoost native format)

### Results
- `results_summary.json` - Comprehensive results including:
  - Dataset information
  - Baseline metrics
  - Optimized hyperparameters
  - Training and test metrics
  - Cross-validation scores
  - Top 20 features by SHAP importance

### Feature Importance
- `shap_feature_importance.csv` - Features ranked by mean absolute SHAP value
- `xgboost_feature_importance.csv` - Features ranked by XGBoost importance

### Visualizations
- `shap_summary_plot.png` - SHAP summary (beeswarm plot)
- `shap_importance_bar.png` - SHAP feature importance bar chart
- `shap_dependence_[1-5]_*.png` - SHAP dependence plots for top 5 features
- `xgboost_feature_importance_plot.png` - XGBoost feature importance (top 30)
- `prediction_vs_actual.png` - Predicted vs actual birth weight scatter plot
- `residual_plot.png` - Residual analysis plot

## Key Features

### Script Features
- Comprehensive data cleaning and validation
- Automatic handling of missing and infinite values
- Baseline model for comparison
- Advanced hyperparameter optimization
- Robust evaluation with cross-validation
- SHAP-based model interpretability
- Multiple visualization outputs
- Model persistence in multiple formats

### Reproducibility
- Fixed random seed (42)
- Deterministic train-test split
- Saved hyperparameters
- Complete results logging

## Usage

### Running the Pipeline
```bash
cd E:\KEM\Project\XG_Full
python xgboost_full_dataset.py
```

### Loading the Trained Model
```python
import pickle
import xgboost as xgb

# Load pickle format
with open('xgboost_full_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Or load XGBoost native format
model = xgb.XGBRegressor()
model.load_model('xgboost_full_model.json')
```

### Making Predictions
```python
import pandas as pd

# Load new data
new_data = pd.read_excel('path_to_new_data.xlsx')

# Ensure same features as training
# (exclude f1_bw, f1_sex, Unnamed: 0, row_index)
X_new = new_data.drop(['f1_bw', 'f1_sex', 'Unnamed: 0', 'row_index'], axis=1)

# Predict
predictions = model.predict(X_new)
```

## Interpretation

### SHAP Values
- **Positive SHAP value**: Feature contributes to higher birth weight
- **Negative SHAP value**: Feature contributes to lower birth weight
- **Magnitude**: Importance of the feature's contribution

### Feature Importance
Two types of feature importance are provided:
1. **XGBoost Importance**: Based on number of splits/gain
2. **SHAP Importance**: Based on average impact on model output

SHAP importance is generally more reliable as it accounts for feature interactions.

## Technical Requirements
```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
openpyxl (for Excel file reading)
```

## Notes
- Training time varies based on system specs (typically 10-30 minutes)
- SHAP analysis uses a sample of 500 test instances for efficiency
- All plots are saved at 300 DPI for publication quality
- Results are reproducible with the same random seed

## Author
Generated by ML Pipeline
Date: 2025-12-07

## References
- XGBoost: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- SHAP: [https://shap.readthedocs.io/](https://shap.readthedocs.io/)
