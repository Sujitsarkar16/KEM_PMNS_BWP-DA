"""
CatBoost Baseline Model for Birth Weight Prediction
Uses power interaction features with CatBoost's ordered boosting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠ CatBoost not installed. Install with: pip install catboost")

# Paths
BASE_DIR = Path(r"e:\KEM\Project")
DATA_DIR = BASE_DIR / "Advanced_Stacking_Models" / "Data"
RESULTS_DIR = BASE_DIR / "Advanced_Stacking_Models" / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the power features dataset"""
    data_path = DATA_DIR / "modeling_dataset_with_power_features.csv"
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        print("   Please run feature_engineering.py first!")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples with {len(df.columns)} features")
    return df

def prepare_data(df):
    """Prepare features and target"""
    # Separate features and target
    target_col = 'f1_bw'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found!")
    
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    
    # Handle missing values using median imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    
    return X, y, X.columns.tolist()

def train_catboost_baseline(X, y, feature_names):
    """Train CatBoost with default/conservative parameters"""
    print("\n" + "="*80)
    print("TRAINING CATBOOST BASELINE MODEL")
    print("="*80)
    
    # Define baseline parameters (conservative for small dataset)
    params = {
        'iterations': 1000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'early_stopping_rounds': 50
    }
    
    print("\nModel Parameters:")
    for key, value in params.items():
        print(f"  • {key}: {value}")
    
    # Initialize model
    model = CatBoostRegressor(**params)
    
    # 5-Fold Cross-Validation
    print("\nPerforming 5-Fold Cross-Validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_rmse_scores = []
    cv_r2_scores = []
    cv_mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        
        cv_rmse_scores.append(rmse)
        cv_r2_scores.append(r2)
        cv_mae_scores.append(mae)
        
        print(f"  Fold {fold}: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")
    
    # Summary statistics
    print("\n" + "-"*80)
    print("Cross-Validation Results:")
    print(f"  • RMSE: {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}")
    print(f"  • R²:   {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
    print(f"  • MAE:  {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")
    
    # Train final model on full data
    print("\nTraining final model on full dataset...")
    final_model = CatBoostRegressor(**params)
    final_model.fit(X, y, verbose=False)
    
    # Full dataset metrics
    y_pred_full = final_model.predict(X)
    rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))
    r2_full = r2_score(y, y_pred_full)
    mae_full = mean_absolute_error(y, y_pred_full)
    
    print(f"\nFull Dataset Performance:")
    print(f"  • RMSE: {rmse_full:.4f}")
    print(f"  • R²:   {r2_full:.4f}")
    print(f"  • MAE:  {mae_full:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    return {
        'model': final_model,
        'cv_metrics': {
            'rmse_mean': float(np.mean(cv_rmse_scores)),
            'rmse_std': float(np.std(cv_rmse_scores)),
            'r2_mean': float(np.mean(cv_r2_scores)),
            'r2_std': float(np.std(cv_r2_scores)),
            'mae_mean': float(np.mean(cv_mae_scores)),
            'mae_std': float(np.std(cv_mae_scores))
        },
        'full_metrics': {
            'rmse': float(rmse_full),
            'r2': float(r2_full),
            'mae': float(mae_full)
        },
        'feature_importance': feature_importance,
        'parameters': params
    }

def save_results(results, timestamp):
    """Save all results to files"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save metrics
    metrics_path = RESULTS_DIR / f"catboost_baseline_metrics_{timestamp}.json"
    metrics_output = {
        'timestamp': datetime.now().isoformat(),
        'model': 'CatBoost_Baseline',
        'cv_metrics': results['cv_metrics'],
        'full_metrics': results['full_metrics'],
        'parameters': results['parameters']
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")
    
    # Save feature importance
    importance_path = RESULTS_DIR / f"catboost_baseline_importance_{timestamp}.csv"
    results['feature_importance'].to_csv(importance_path, index=False)
    print(f"✓ Saved feature importance to: {importance_path}")
    
    # Save model
    model_path = RESULTS_DIR / f"catboost_baseline_model_{timestamp}.cbm"
    results['model'].save_model(str(model_path))
    print(f"✓ Saved model to: {model_path}")
    
    print()

def main():
    if not CATBOOST_AVAILABLE:
        print("\n❌ Cannot proceed without CatBoost. Please install it first.")
        print("   Run: pip install catboost")
        return
    
    print("=" * 80)
    print("CATBOOST BASELINE MODEL - BIRTH WEIGHT PREDICTION")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading Dataset...")
    print("-" * 80)
    df = load_data()
    if df is None:
        return
    print()
    
    # Prepare data
    print("Preparing Data...")
    print("-" * 80)
    X, y, feature_names = prepare_data(df)
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print(f"✓ Feature names: {feature_names}")
    print()
    
    # Train model
    results = train_catboost_baseline(X, y, feature_names)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, timestamp)
    
    # Final summary
    print("=" * 80)
    print("CATBOOST BASELINE COMPLETE")
    print("=" * 80)
    print(f"\n🎯 Performance Summary:")
    print(f"  • Cross-Val RMSE: {results['cv_metrics']['rmse_mean']:.4f} ± {results['cv_metrics']['rmse_std']:.4f}")
    print(f"  • Cross-Val R²:   {results['cv_metrics']['r2_mean']:.4f} ± {results['cv_metrics']['r2_std']:.4f}")
    print(f"  • Full Data RMSE: {results['full_metrics']['rmse']:.4f}")
    print(f"  • Full Data R²:   {results['full_metrics']['r2']:.4f}")
    print()
    print("🎯 Next Steps:")
    print("  1. Run: python catboost_optimized.py (for hyperparameter tuning)")
    print("  2. Run: python stacking_ensemble.py")
    print()

if __name__ == "__main__":
    main()
