"""
XGBoost Model Training on Full Dataset with Hyperparameter Optimization and SHAP Analysis
=========================================================================================

This script trains an XGBoost model on the full dataset with all available variables,
excluding potential data leakage variables.

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
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("XGBoost Full Dataset Training Pipeline")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================
print("\n[1/7] Loading dataset...")
data_path = 'E:/KEM/Project/Data/raw/IMPUTED_DATA_WITH REDUCED_columns_21_09_2025.xlsx'
df = pd.read_excel(data_path)

print(f"Initial dataset shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")

# ============================================================================
# 2. DATA CLEANING AND PREPROCESSING
# ============================================================================
print("\n[2/7] Data cleaning and preprocessing...")

# Define target variable
TARGET = 'f1_bw'

# Define columns to exclude (data leakage and non-predictive)
EXCLUDE_COLUMNS = [
    'Unnamed: 0',      # Index column
    'row_index',       # Index column
    'f1_bw',          # Target variable
    'f1_sex',         # Post-birth variable (data leakage)
]

# Additional exclusions based on naming patterns (any other f1_ variables that might exist)
f1_columns = [col for col in df.columns if col.startswith('f1_') and col != TARGET]
EXCLUDE_COLUMNS.extend([col for col in f1_columns if col not in EXCLUDE_COLUMNS])

# Remove duplicates from exclude list
EXCLUDE_COLUMNS = list(set(EXCLUDE_COLUMNS))

print(f"\nExcluded columns ({len(EXCLUDE_COLUMNS)}):")
for col in sorted(EXCLUDE_COLUMNS):
    if col in df.columns:
        print(f"  - {col}")

# Verify target exists
if TARGET not in df.columns:
    raise ValueError(f"Target variable '{TARGET}' not found in dataset!")

# Create feature set
available_exclude = [col for col in EXCLUDE_COLUMNS if col in df.columns]
feature_columns = [col for col in df.columns if col not in available_exclude]

print(f"\nTotal features for modeling: {len(feature_columns)}")

# Separate features and target
X = df[feature_columns].copy()
y = df[TARGET].copy()

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Check for missing values
missing_counts = X.isnull().sum()
if missing_counts.sum() > 0:
    print(f"\nWarning: Found {missing_counts.sum()} missing values")
    print("Columns with missing values:")
    print(missing_counts[missing_counts > 0])
    # Fill missing values with median for numerical columns
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype in ['float64', 'int64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0], inplace=True)
    print("Missing values imputed successfully")

# Check for infinite values
inf_counts = np.isinf(X.select_dtypes(include=[np.number])).sum()
if inf_counts.sum() > 0:
    print(f"\nWarning: Found {inf_counts.sum()} infinite values")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    print("Infinite values replaced successfully")

print("\nTarget variable statistics:")
print(y.describe())

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n[3/7] Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================================
# 4. BASELINE XGBoost MODEL
# ============================================================================
print("\n[4/7] Training baseline XGBoost model...")

baseline_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

baseline_model.fit(X_train, y_train)

# Baseline predictions
y_pred_baseline = baseline_model.predict(X_test)

# Baseline metrics
baseline_metrics = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_baseline)),
    'MAE': mean_absolute_error(y_test, y_pred_baseline),
    'R2': r2_score(y_test, y_pred_baseline)
}

print("\nBaseline Model Performance:")
print(f"  RMSE: {baseline_metrics['RMSE']:.4f}")
print(f"  MAE: {baseline_metrics['MAE']:.4f}")
print(f"  R²: {baseline_metrics['R2']:.4f}")

# ============================================================================
# 5. HYPERPARAMETER OPTIMIZATION
# ============================================================================
print("\n[5/7] Hyperparameter optimization with RandomizedSearchCV...")

# Define hyperparameter grid
param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
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
    n_iter=50,  # Number of parameter settings sampled
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("Starting randomized search (this may take a while)...")
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_

print("\nBest hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param}: {value}")

# ============================================================================
# 6. OPTIMIZED MODEL EVALUATION
# ============================================================================
print("\n[6/7] Evaluating optimized model...")

# Predictions
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Calculate metrics
train_metrics = {
    'RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
    'MAE': mean_absolute_error(y_train, y_pred_train),
    'R2': r2_score(y_train, y_pred_train)
}

test_metrics = {
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'MAE': mean_absolute_error(y_test, y_pred_test),
    'R2': r2_score(y_test, y_pred_test)
}

print("\nOptimized Model Performance:")
print("\nTraining Set:")
print(f"  RMSE: {train_metrics['RMSE']:.4f}")
print(f"  MAE: {train_metrics['MAE']:.4f}")
print(f"  R²: {train_metrics['R2']:.4f}")

print("\nTest Set:")
print(f"  RMSE: {test_metrics['RMSE']:.4f}")
print(f"  MAE: {test_metrics['MAE']:.4f}")
print(f"  R²: {test_metrics['R2']:.4f}")

# Cross-validation
print("\nPerforming 5-fold cross-validation...")
cv_scores = cross_val_score(
    best_model, X, y, 
    cv=5, 
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
cv_rmse = np.sqrt(-cv_scores)

print(f"CV RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")

# ============================================================================
# 7. SHAP ANALYSIS
# ============================================================================
print("\n[7/7] Performing SHAP analysis...")

# Create SHAP explainer
explainer = shap.TreeExplainer(best_model)

# Calculate SHAP values on a sample of test data (for computational efficiency)
sample_size = min(500, len(X_test))
X_test_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)

print(f"Calculating SHAP values for {sample_size} test samples...")
shap_values = explainer.shap_values(X_test_sample)

# SHAP Summary Plot
print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(14, 10))
shap.summary_plot(shap_values, X_test_sample, show=False, max_display=30)
plt.tight_layout()
plt.savefig('E:/KEM/Project/XG_Full/shap_summary_plot.png', dpi=300, bbox_inches='tight')
print("Saved: shap_summary_plot.png")
plt.close()

# SHAP Bar Plot
print("Generating SHAP importance bar plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False, max_display=30)
plt.tight_layout()
plt.savefig('E:/KEM/Project/XG_Full/shap_importance_bar.png', dpi=300, bbox_inches='tight')
print("Saved: shap_importance_bar.png")
plt.close()

# Calculate mean absolute SHAP values for feature importance
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'feature': X_test_sample.columns,
    'mean_abs_shap_value': mean_abs_shap
}).sort_values('mean_abs_shap_value', ascending=False)

print("\nTop 20 features by SHAP importance:")
print(feature_importance_df.head(20).to_string(index=False))

# Save SHAP feature importance
feature_importance_df.to_csv('E:/KEM/Project/XG_Full/shap_feature_importance.csv', index=False)
print("\nSaved: shap_feature_importance.csv")

# Individual SHAP plots for top features
print("\nGenerating individual SHAP dependence plots for top 5 features...")
top_features = feature_importance_df.head(5)['feature'].tolist()

for i, feature in enumerate(top_features, 1):
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature, 
        shap_values, 
        X_test_sample,
        show=False
    )
    plt.title(f'SHAP Dependence Plot: {feature}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'E:/KEM/Project/XG_Full/shap_dependence_{i}_{feature}.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: shap_dependence_{i}_{feature}.png")
    plt.close()

# ============================================================================
# 8. SAVE RESULTS AND MODEL
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save model
model_path = 'E:/KEM/Project/XG_Full/xgboost_full_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
print(f"\nSaved trained model: {model_path}")

# XGBoost native format
best_model.save_model('E:/KEM/Project/XG_Full/xgboost_full_model.json')
print("Saved XGBoost native format: xgboost_full_model.json")

# Save results summary
results = {
    'dataset_info': {
        'total_samples': len(df),
        'total_features': len(feature_columns),
        'excluded_columns': EXCLUDE_COLUMNS,
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    },
    'baseline_metrics': baseline_metrics,
    'best_hyperparameters': random_search.best_params_,
    'training_metrics': train_metrics,
    'test_metrics': test_metrics,
    'cv_metrics': {
        'cv_rmse_mean': float(cv_rmse.mean()),
        'cv_rmse_std': float(cv_rmse.std())
    },
    'top_20_features': feature_importance_df.head(20)[['feature', 'mean_abs_shap_value']].to_dict('records')
}

with open('E:/KEM/Project/XG_Full/results_summary.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Saved: results_summary.json")

# Generate XGBoost feature importance
xgb_importance = best_model.feature_importances_
xgb_importance_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': xgb_importance
}).sort_values('importance', ascending=False)

xgb_importance_df.to_csv('E:/KEM/Project/XG_Full/xgboost_feature_importance.csv', index=False)
print("Saved: xgboost_feature_importance.csv")

# XGBoost feature importance plot
plt.figure(figsize=(12, 10))
top_30_xgb = xgb_importance_df.head(30)
plt.barh(range(len(top_30_xgb)), top_30_xgb['importance'])
plt.yticks(range(len(top_30_xgb)), top_30_xgb['feature'])
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Top 30 Features - XGBoost Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('E:/KEM/Project/XG_Full/xgboost_feature_importance_plot.png', dpi=300, bbox_inches='tight')
print("Saved: xgboost_feature_importance_plot.png")
plt.close()

# Prediction vs Actual plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Birth Weight (g)', fontsize=12)
plt.ylabel('Predicted Birth Weight (g)', fontsize=12)
plt.title('Predicted vs Actual Birth Weight - Full Dataset XGBoost', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('E:/KEM/Project/XG_Full/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("Saved: prediction_vs_actual.png")
plt.close()

# Residual plot
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Birth Weight (g)', fontsize=12)
plt.ylabel('Residuals (g)', fontsize=12)
plt.title('Residual Plot - Full Dataset XGBoost', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('E:/KEM/Project/XG_Full/residual_plot.png', dpi=300, bbox_inches='tight')
print("Saved: residual_plot.png")
plt.close()

print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
print("\nAll results saved in: E:/KEM/Project/XG_Full/")
print("\nGenerated files:")
print("  - xgboost_full_model.pkl (trained model - pickle)")
print("  - xgboost_full_model.json (trained model - XGBoost format)")
print("  - results_summary.json (comprehensive results)")
print("  - shap_summary_plot.png (SHAP summary)")
print("  - shap_importance_bar.png (SHAP importance bar)")
print("  - shap_feature_importance.csv (SHAP feature rankings)")
print("  - shap_dependence_[1-5]_*.png (top 5 SHAP dependence plots)")
print("  - xgboost_feature_importance.csv (XGBoost feature rankings)")
print("  - xgboost_feature_importance_plot.png (XGBoost importance plot)")
print("  - prediction_vs_actual.png (model performance visualization)")
print("  - residual_plot.png (residual analysis)")

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\nModel Type: XGBoost Regressor")
print(f"Total Features: {len(feature_columns)}")
print(f"Training Samples: {len(X_train)}")
print(f"Test Samples: {len(X_test)}")
print(f"\nTest Set Performance:")
print(f"  RMSE: {test_metrics['RMSE']:.4f} g")
print(f"  MAE: {test_metrics['MAE']:.4f} g")
print(f"  R²: {test_metrics['R2']:.4f}")
print(f"\nCross-Validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std():.4f})")
print("\n" + "="*80)
