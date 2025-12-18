
"""
XGBoost Model Training on Top 30 SHAP Features
=========================================================================================

This script trains an XGBoost model on the Top 30 features identified by SHAP analysis
on the full dataset.

Author: ML Pipeline
Date: 2025-12-07
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("XGBoost Top 30 SHAP Features Training Pipeline")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND FEATURE SELECTION
# ============================================================================
print("\n[1/7] Loading dataset...")
data_path = 'E:/KEM/Project/Data/raw/IMPUTED_DATA_WITH REDUCED_columns_21_09_2025.xlsx'
df = pd.read_excel(data_path)

print(f"Initial dataset shape: {df.shape}")

# Define Target
TARGET = 'f1_bw'

# Define Top 30 Features from SHAP Analysis
TOP_30_FEATURES = [
    'f0_m_GA_Del', 'f0_m_plac_wt', 'f0_m_fundal_ht_v2', 'f0_m_abd_cir_v2',
    'f0_m_g_sc_v2', 'f0_m_glu_f_v2', 'f0_m_pulse_r1_v2', 'f0_f_plt_ini',
    'f0_f_head_cir_ini', 'f0_m_parity_v1', 'f0_m_r4_v2', 'f0_m_rcf_v2',
    'f0_m_g_sc_v1', 'f0_m_wt_prepreg', 'f0_m_l_sc_v1', 'f0_m_waist_circ_v1',
    'f0_m_wt_v2', 'f0_m_h_sc_v2', 'f0_m_o1_sc_v1', 'f0_m_plt_v1',
    'f0_m_lunch_fat_v2', 'f0_m_rotis_fat_v1', 'f0_m_iron_fve_v1', 'f0_m_fer_v2',
    'f0_f_ferr_ini', 'f0_m_h7_sc_v1', 'f0_m_cal_ch_v2', 'f0_m_j7_sc_v2',
    'f0_m_d3_sc_v2', 'f0_f_ly_perc_ini'
]

print(f"\nEvaluating with the following {len(TOP_30_FEATURES)} features:")
for i, feature in enumerate(TOP_30_FEATURES, 1):
    print(f"  {i}. {feature}")

# Check if all features exist
missing_features = [f for f in TOP_30_FEATURES if f not in df.columns]
if missing_features:
    print(f"\n[WARNING] The following features are missing from the dataset: {missing_features}")
    # Update list to only include existing features
    TOP_30_FEATURES = [f for f in TOP_30_FEATURES if f in df.columns]
    print(f"Proceeding with {len(TOP_30_FEATURES)} features.")

# Filter Data
X = df[TOP_30_FEATURES].copy()
y = df[TARGET].copy()

# Drop rows with missing target
valid_indices = y.dropna().index
X = X.loc[valid_indices]
y = y.loc[valid_indices]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Handle Missing Values (Impute with Median)
X = X.fillna(X.median())

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================
print("\n[2/7] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================================
# 3. HYPERPARAMETER OPTIMIZATION
# ============================================================================
print("\n[3/7] Hyperparameter optimization with RandomizedSearchCV...")

# Define hyperparameter grid
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.2), # Reduced upper bound for stability
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 7),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 0.5),
    'reg_lambda': uniform(0.5, 1.0)
}

# Initialize base model
base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("Starting randomized search...")
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

print("\nBest hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

# ============================================================================
# 4. EVALUATION
# ============================================================================
print("\n[4/7] Evaluating optimized model...")

# Predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Calculate metrics
def calculate_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

train_metrics = calculate_metrics(y_train, y_pred_train, "Training")
test_metrics = calculate_metrics(y_test, y_pred_test, "Test")

# ============================================================================
# 5. SAVE RESULTS
# ============================================================================
output_dir = 'E:/KEM/Project/XG_Full/Top30_Optimization'
os.makedirs(output_dir, exist_ok=True)

# Save Results JSON
results = {
    'features': TOP_30_FEATURES,
    'best_params': random_search.best_params_,
    'train_metrics': train_metrics,
    'test_metrics': test_metrics
}

with open(f'{output_dir}/results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {output_dir}")

# ============================================================================
# 6. PLOTTING
# ============================================================================
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Birth Weight')
plt.ylabel('Predicted Birth Weight')
plt.title(f'XGBoost (Top 30 Features)\nRMSE: {test_metrics["RMSE"]:.2f}, R²: {test_metrics["R2"]:.3f}')
plt.savefig(f'{output_dir}/actual_vs_predicted.png')
print("Saved actual_vs_predicted.png")

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"RMSE: {test_metrics['RMSE']:.4f}")
print("="*80)
