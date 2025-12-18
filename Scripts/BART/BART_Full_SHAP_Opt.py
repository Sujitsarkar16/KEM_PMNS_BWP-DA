
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
import os
import json
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from scipy.stats import pearsonr
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = r'Data/raw/IMPUTED_DATA_WITH REDUCED_columns_21_09_2025.xlsx'
OUTPUT_DIR = r'BART Full'
TARGET_COL = 'f1_bw'
EXCLUDE_COLS = ['f1_sex', 'Unnamed: 0', 'row_index', 'index']  # Common exclusions

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_data(data_path):
    print(f"Loading data from {data_path}...")
    if data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
    
    print(f"Original shape: {df.shape}")
    
    # Drop excluded columns if they exist
    cols_to_drop = [col for col in EXCLUDE_COLS if col in df.columns]
    if cols_to_drop:
        print(f"Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    # Drop rows with missing target
    if df[TARGET_COL].isnull().any():
        print("Dropping rows with missing target...")
        df = df.dropna(subset=[TARGET_COL])
    
    # Handle object columns (simple label encoding for SHAP/XGBoost)
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    print(f"Shape after preprocessing: {df.shape}")
    return df

def perform_shap_analysis(df):
    print("\n" + "="*50)
    print("STEP 1: SHAP Analysis for Feature Selection")
    print("="*50)
    
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    print("Training XGBoost model for SHAP analysis...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Calculate mean absolute SHAP values for ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values(by='mean_abs_shap', ascending=False)
    
    # Save SHAP summary plot
    print("Saving SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_plot_full.png'), bbox_inches='tight')
    plt.close()
    
    # Save feature importance
    csv_path = os.path.join(OUTPUT_DIR, 'shap_feature_importance_full.csv')
    feature_importance.to_csv(csv_path, index=False)
    print(f"SHAP feature importance saved to {csv_path}")
    
    # Get top 30 features
    top_30_features = feature_importance['feature'].head(30).tolist()
    print(f"Top 30 features selected: {top_30_features}")
    
    return top_30_features

# -------------------------------------------------------------------------
# BART Implementation (Adapted from BART_Optimized.py)
# -------------------------------------------------------------------------

try:
    from bartpy.sklearnmodel import SklearnModel
    BART_AVAILABLE = True
except ImportError:
    BART_AVAILABLE = False
    print("[WARNING] bartpy library not found. Please install: pip install bartpy")

class BARTWrapper:
    """Wrapper for bartpy to work with RandomizedSearchCV"""
    def __init__(self, n_trees=200, n_burn=200, n_samples=1000, alpha=0.95, beta=2.0):
        self.n_trees = n_trees
        self.n_burn = n_burn
        self.n_samples = n_samples
        self.alpha = alpha
        self.beta = beta
        self.model = None
    
    def get_params(self, deep=True):
        return {
            'n_trees': self.n_trees,
            'n_burn': self.n_burn,
            'n_samples': self.n_samples,
            'alpha': self.alpha,
            'beta': self.beta
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def fit(self, X, y):
        # Ensure X is passed as numpy array or similar, bartpy sometimes picky
        self.model = SklearnModel(
            n_trees=self.n_trees,
            n_burn=self.n_burn,
            n_samples=self.n_samples,
            alpha=self.alpha,
            beta=self.beta,
            n_jobs=1 
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

def perform_bart_optimization(df, features):
    print("\n" + "="*50)
    print("STEP 2: BART Hyperparameter Optimization on Top 30 Features")
    print("="*50)
    
    if not BART_AVAILABLE:
        print("BART not available. Skipping optimization.")
        return
    
    X = df[features]
    y = df[TARGET_COL]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Parameter Grid
    param_grid = {
        'n_trees': [50, 100, 200],
        'n_burn': [100, 200],
        'n_samples': [500, 1000],
        'alpha': [0.90, 0.95],
        'beta': [1.5, 2.0]
    }
    
    print("Starting RandomizedSearchCV...")
    # Smaller n_iter to save time if needed, user didn't specify strict constraint but BART is slow
    n_iter = 10 
    
    base_model = BARTWrapper()
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring=rmse_scorer,
        n_jobs=1, # bartpy not threadsafe usually
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train.values, y_train.values)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    print(f"Best Parameters: {best_params}")
    print(f"Best CV RMSE: {-random_search.best_score_:.4f}")
    
    # Evaluate on Test
    y_pred = best_model.predict(X_test.values)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\nTest Set Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Save Results
    results = {
        'best_params': best_params,
        'metrics': {
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        },
        'features': features
    }
    
    with open(os.path.join(OUTPUT_DIR, 'bart_optimization_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    # Plots
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'BART Optimized (Top 30 Features)\nRMSE={rmse:.4f}, R2={r2:.4f}')
    plt.savefig(os.path.join(OUTPUT_DIR, 'bart_actual_vs_predicted.png'))
    plt.close()

if __name__ == "__main__":
    # 1. Load Data
    df = load_and_preprocess_data(DATA_PATH)
    
    # 2. SHAP Analysis -> Top 30
    top_30 = perform_shap_analysis(df)
    
    # 3. BART Optimization
    perform_bart_optimization(df, top_30)
