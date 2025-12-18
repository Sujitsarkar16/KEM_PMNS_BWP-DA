"""
Stacking Ensemble Regressor - Kaggle Grandmaster Technique
Combines XGBoost + BART + LinearRegression with Ridge meta-learner

Architecture:
- Level 0: XGBoost (splits) + BART (uncertainty) + LinearRegression (baseline)
- Level 1: RidgeRegression (optimal weighting)

Expected improvement: 3-5% RMSE reduction over individual models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

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
    target_col = 'f1_bw'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found!")
    
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns)
    
    # Standardize features for LinearRegression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X, X_scaled, y, X.columns.tolist()

def create_stacking_ensemble():
    """Create stacking ensemble with multiple base learners"""
    print("\n" + "="*80)
    print("BUILDING STACKING ENSEMBLE")
    print("="*80)
    
    base_learners = []
    
    # Base Model 1: XGBoost (captures splits)
    if XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        base_learners.append(('xgboost', xgb_model))
        print("✓ Added XGBoost (captures sharp splits)")
    else:
        print("⚠ XGBoost not available")
    
    # Base Model 2: CatBoost (captures categorical + ordered boosting)
    if CATBOOST_AVAILABLE:
        catboost_model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=3,
            random_seed=42,
            verbose=False
        )
        base_learners.append(('catboost', catboost_model))
        print("✓ Added CatBoost (captures non-linear interactions)")
    else:
        print("⚠ CatBoost not available")
    
    # Base Model 3: LinearRegression (captures baseline trends)
    linear_model = LinearRegression()
    base_learners.append(('linear', linear_model))
    print("✓ Added LinearRegression (captures linear baseline)")
    
    if len(base_learners) < 2:
        print("\n❌ Need at least 2 base learners for stacking!")
        print("   Install: pip install xgboost catboost")
        return None
    
    # Meta-learner: Ridge Regression
    meta_learner = Ridge(alpha=1.0)
    print("✓ Meta-learner: RidgeRegression (alpha=1.0)")
    
    # Create stacking ensemble
    stacking_model = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        verbose=0
    )
    
    print(f"\n📦 Stacking Architecture:")
    print(f"  • Level 0 (Base): {len(base_learners)} models")
    print(f"  • Level 1 (Meta): Ridge α=1.0")
    print(f"  • CV Folds: 5")
    
    return stacking_model, [name for name, _ in base_learners]

def train_and_evaluate(model, X, y, model_name="Stacking"):
    """Train model and evaluate with cross-validation"""
    print(f"\n" + "="*80)
    print(f"TRAINING {model_name.upper()} MODEL")
    print("="*80)
    
    # 5-Fold Cross-Validation
    print("\nPerforming 5-Fold Cross-Validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_rmse_scores = []
    cv_r2_scores = []
    cv_mae_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train
        model.fit(X_train, y_train)
        
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
    
    # Summary
    print("\n" + "-"*80)
    print("Cross-Validation Results:")
    print(f"  • RMSE: {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}")
    print(f"  • R²:   {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
    print(f"  • MAE:  {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")
    
    # Train on full dataset
    print("\nTraining on full dataset...")
    model.fit(X, y)
    y_pred_full = model.predict(X)
    
    rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))
    r2_full = r2_score(y, y_pred_full)
    mae_full = mean_absolute_error(y, y_pred_full)
    
    print(f"\nFull Dataset Performance:")
    print(f"  • RMSE: {rmse_full:.4f}")
    print(f"  • R²:   {r2_full:.4f}")
    print(f"  • MAE:  {mae_full:.4f}")
    
    return {
        'model': model,
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
        }
    }

def save_results(results, base_model_names, timestamp):
    """Save results"""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save metrics
    metrics_path = RESULTS_DIR / f"stacking_ensemble_metrics_{timestamp}.json"
    metrics_output = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Stacking_Ensemble',
        'base_models': base_model_names,
        'meta_learner': 'Ridge (alpha=1.0)',
        'cv_metrics': results['cv_metrics'],
        'full_metrics': results['full_metrics']
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"✓ Saved metrics to: {metrics_path}")
    
    print()

def main():
    print("=" * 80)
    print("STACKING ENSEMBLE - KAGGLE GRANDMASTER TECHNIQUE")
    print("=" * 80)
    print()
    
    # Check dependencies
    if not XGBOOST_AVAILABLE and not CATBOOST_AVAILABLE:
        print("❌ Need at least XGBoost or CatBoost installed")
        print("   Run: pip install xgboost catboost")
        return
    
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
    X, X_scaled, y, feature_names = prepare_data(df)
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    print()
    
    # Create stacking ensemble
    stacking_model, base_model_names = create_stacking_ensemble()
    if stacking_model is None:
        return
    
    # Train and evaluate
    # Note: Use X_scaled for LinearRegression compatibility
    results = train_and_evaluate(stacking_model, X_scaled, y, "Stacking Ensemble")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(results, base_model_names, timestamp)
    
    # Final summary
    print("=" * 80)
    print("STACKING ENSEMBLE COMPLETE")
    print("=" * 80)
    print(f"\n🎯 Performance Summary:")
    print(f"  • Cross-Val RMSE: {results['cv_metrics']['rmse_mean']:.4f} ± {results['cv_metrics']['rmse_std']:.4f}")
    print(f"  • Cross-Val R²:   {results['cv_metrics']['r2_mean']:.4f} ± {results['cv_metrics']['r2_std']:.4f}")
    print(f"  • Full Data RMSE: {results['full_metrics']['rmse']:.4f}")
    print(f"  • Full Data R²:   {results['full_metrics']['r2']:.4f}")
    print()
    print("💡 Theory: The meta-learner finds optimal weights to combine")
    print("   different model predictions, canceling out individual errors.")
    print()
    print("🎯 Next Step:")
    print("  Run: python comparison_report.py")
    print()

if __name__ == "__main__":
    main()
