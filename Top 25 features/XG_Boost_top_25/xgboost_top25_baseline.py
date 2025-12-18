"""
XGBoost Baseline - Top 25 Variables
====================================

This script implements a baseline XGBoost model using the top 25 features.

Author: Sujit Sarkar
Date: 2025-12-06
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb
import json
import joblib
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')


TOP_25_FEATURES = [
    "f0_m_GA_Del", "f0_m_plac_wt", "f0_m_abd_cir_v2", "f0_m_wt_v2",
    "f0_m_fundal_ht_v2", "f0_m_hip_circ_v2", "f0_m_rcf_v2", "f0_m_ht",
    "f0_m_bmi_v2", "f0_m_wt_prepreg", "f0_m_snacks_sc_v1", "f0_m_waist_circ_v2",
    "f0_m_j8_sc_v1", "f0_m_f10_sc_v1", "f0_m_bmi_6yr", "f0_m_age",
    "f0_m_m2_sc_v1", "f0_m_n_sc_v1", "f0_m_g1_sc_v2", "f0_m_o_sc_v1",
    "f0_m_g_sc_v2", "f0_f_wt_ini", "f0_m_glv_sc_v", "f0_m_p10_sc_v1",
    "f0_m_d1_sc_v1"
]


def load_and_prepare_data(data_path):
    """Load and prepare data"""
    print("=" * 80)
    print("DATA LOADING AND PREPARATION")
    print("=" * 80)
    
    df = pd.read_csv(data_path)
    print(f"[OK] Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    target = 'f1_bw'
    available_features = [f for f in TOP_25_FEATURES if f in df.columns]
    
    X = df[available_features].copy()
    y = df[target].copy()
    
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    print(f"[OK] Using {len(available_features)} features, {len(X_imputed)} samples")
    
    return X_imputed, y, available_features


def train_xgboost_baseline(X, y, features):
    """Train XGBoost baseline model"""
    print("\n" + "=" * 80)
    print("XGBOOST BASELINE TRAINING")
    print("=" * 80)
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"[OK] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Default XGBoost parameters
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("\n[INFO] Training XGBoost with default parameters...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("[OK] Model trained successfully")
    
    results = {}
    for name, X_split, y_split in [('train', X_train, y_train), 
                                     ('validation', X_val, y_val), 
                                     ('test', X_test, y_test)]:
        y_pred = model.predict(X_split)
        
        rmse = np.sqrt(mean_squared_error(y_split, y_pred))
        mae = mean_absolute_error(y_split, y_pred)
        r2 = r2_score(y_split, y_pred)
        mape = np.mean(np.abs((y_split - y_pred) / y_split)) * 100
        correlation, p_value = pearsonr(y_split, y_pred)
        
        results[name] = {
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R²': float(r2),
            'MAPE': float(mape),
            'Correlation': float(correlation),
            'P-value': float(p_value),
            'Sample_Size': int(len(y_split))
        }
        
        print(f"\n[{name.upper()} SET] Performance:")
        print(f"  - RMSE:         {rmse:.4f} grams")
        print(f"  - MAE:          {mae:.4f} grams")
        print(f"  - R²:           {r2:.4f}")
        print(f"  - Correlation:  {correlation:.4f}")
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, results, feature_importance


def save_results(model, results, feature_importance, features):
    """Save model and results"""
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_dir = 'e:/KEM/Project/Top 25 features/Results'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_file = f'{output_dir}/xgboost_top25_baseline_model_{timestamp}.pkl'
    joblib.dump(model, model_file)
    
    comprehensive_results = {
        'model_type': 'XGBoost_Baseline_Top25',
        'num_features': len(features),
        'features': features,
        'parameters': 'Default (n_estimators=100, max_depth=6, lr=0.1)',
        'performance_metrics': results,
        'timestamp': timestamp
    }
    
    results_file = f'{output_dir}/xgboost_top25_baseline_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    metrics_data = []
    for split_name in ['train', 'validation', 'test']:
        row = {'split': split_name}
        row.update(results[split_name])
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_file = f'{output_dir}/xgboost_top25_baseline_metrics_{timestamp}.csv'
    metrics_df.to_csv(metrics_file, index=False)
    
    importance_file = f'{output_dir}/xgboost_top25_baseline_importance_{timestamp}.csv'
    feature_importance.to_csv(importance_file, index=False)
    
    print(f"[OK] Model saved: {model_file}")
    print(f"[OK] Results saved: {results_file}")
    
    return results_file, metrics_file


def main():
    """Main function"""
    print("=" * 80)
    print("XGBOOST BASELINE - TOP 25 VARIABLES")
    print("=" * 80)
    
    data_path = 'e:/KEM/Project/Data/Top_25_Data.csv'
    
    X, y, features = load_and_prepare_data(data_path)
    model, results, feature_importance = train_xgboost_baseline(X, y, features)
    save_results(model, results, feature_importance, features)
    
    print("\n" + "=" * 80)
    print("XGBOOST BASELINE COMPLETED!")
    print("=" * 80)
    print(f"\n[FINAL RESULTS]")
    print(f"  Test RMSE:          {results['test']['RMSE']:.4f} grams")
    print(f"  Test R²:            {results['test']['R²']:.4f}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
