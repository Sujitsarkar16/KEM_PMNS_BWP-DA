"""
XGBoost Optimized - Clean Top 15 Features (No Data Leakage)
============================================================

This script implements an optimized XGBoost model with comprehensive
hyperparameter optimization using RandomizedSearchCV.

!! DATA LEAKAGE FIXED !!
Removed features:
- LBW_flag (created from target variable)
- plac_bw_ratio (uses target variable in calculation)

Hyperparameters optimized:
- n_estimators: Number of boosting rounds
- max_depth: Maximum tree depth (REDUCED to prevent overfitting)
- learning_rate: Step size shrinkage
- subsample: Subsample ratio of training instances
- colsample_bytree: Subsample ratio of columns
- min_child_weight: Minimum sum of instance weight in a child (INCREASED)
- gamma: Minimum loss reduction for split

Author: Sujit Sarkar
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from scipy.stats import pearsonr
import xgboost as xgb
import json
import warnings
import os
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')


class XGBoostTop30Optimized:
    """
    XGBoost optimized model with hyperparameter tuning
    """
    
    def __init__(self, data_path='paper/data/clean_top15_features_20251206.csv'):
        """Initialize XGBoost optimized implementation with CLEAN features (no data leakage)"""
        self.data_path = data_path
        self.data = None
        self.model = None
        self.best_model = None
        self.results = {}
        self.best_params = {}
        
        # CLEAN Top 15 features (NO DATA LEAKAGE)
        # Removed: LBW_flag, plac_bw_ratio (data leakage)
        # Removed: 10 features with zero SHAP importance
        # Ordered by SHAP importance (highest to lowest)
        self.selected_features = [
            # Top 10 features (79:1 sample ratio)
            'f0_m_plac_wt',          # Rank 1: 23.06% importance
            'f0_m_GA_Del',           # Rank 2: 4.98%
            'gestational_health_index',  # Rank 3: 4.66%
            'f0_m_ht',               # Rank 4: 1.20%
            'bmi_age_interaction',   # Rank 5: 1.14%
            'f0_m_abd_cir_v2',       # Rank 6: 0.88%
            'f0_m_rcf_v2',           # Rank 7: 0.86%
            'f0_m_wt_prepreg_squared',  # Rank 8: 0.79%
            'f0_m_fundal_ht_v2',     # Rank 9: 0.74%
            'bmi_age_ratio',         # Rank 10: 0.66%
            
            # Additional 5 features for Top-15 (53:1 sample ratio)
            'f0_m_bi_v1',            # Rank 11: 0.65%
            'nutritional_status',    # Rank 12: 0.64%
            'wt_ht_interaction',     # Rank 13: 0.42%
            'f0_m_int_sin_ma',       # Rank 14: 0.34%
            'f0_m_age'               # Rank 15: 0.31%
        ]
        
        self.target = 'f1_bw'
        
    def load_and_prepare_data(self):
        """Step 1: Load and prepare data"""
        print("=" * 80)
        print("STEP 1: DATA LOADING")
        print("=" * 80)
        
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded CLEAN dataset (no data leakage): {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        print(f"[OK] Using {len(self.selected_features)} features (removed LBW_flag & plac_bw_ratio)")
        print(f"[OK] Sample-to-feature ratio: {self.data.shape[0]/len(self.selected_features):.1f}:1")
        
        return self.data
    
    def prepare_data_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Step 2: Prepare splits"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA SPLITTING")
        print("=" * 80)
        
        X = self.data[self.selected_features].values
        y = self.data[self.target].values
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=True
        )
        
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Combine for CV
        self.X_cv = np.vstack([X_train, X_val])
        self.y_cv = np.concatenate([y_train, y_val])
        
        print(f"[OK] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(f"[OK] CV set: {self.X_cv.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def define_hyperparameter_search_space(self):
        """Step 3: Define search space"""
        print("\n" + "=" * 80)
        print("STEP 3: HYPERPARAMETER SEARCH SPACE")
        print("=" * 80)
        
        # UPDATED: More aggressive regularization to prevent overfitting
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7],  # REDUCED from [3,5,7,9,11] to prevent overfitting
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],  # Lower values for better generalization
            'subsample': [0.5, 0.6, 0.7, 0.8],  # REDUCED to increase regularization
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8],  # REDUCED to prevent feature overfitting
            'min_child_weight': [5, 7, 10, 15, 20],  # INCREASED from [1,3,5,7] for regularization
            'gamma': [0, 0.1, 0.2, 0.3, 0.5]  # Increased range for split regularization
        }
        
        print("\n[Hyperparameter Grid]:")
        for key, values in param_distributions.items():
            print(f"  {key}: {values}")
        
        # Calculate combinations
        total = 1
        for values in param_distributions.values():
            total *= len(values)
        print(f"\n[INFO] Total combinations: {total}")
        
        return param_distributions
    
    def perform_random_search(self, param_distributions, n_iter=50, cv_folds=5, random_state=42):
        """Step 4: Randomized hyperparameter search"""
        print("\n" + "=" * 80)
        print("STEP 4: RANDOMIZED HYPERPARAMETER SEARCH")
        print("=" * 80)
        
        print(f"\n[INFO] Configuration:")
        print(f"  - Iterations: {n_iter}")
        print(f"  - CV folds: {cv_folds}")
        print(f"  - This may take several minutes...")
        
        # Base model
        xgb_base = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=random_state,
            n_jobs=-1,
            tree_method='hist'
        )
        
        # RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
            return_train_score=True
        )
        
        # Fit
        print("\n[Training] Starting randomized search...")
        random_search.fit(self.X_cv, self.y_cv)
        
        # Get best parameters
        self.best_params = random_search.best_params_
        self.best_cv_score = np.sqrt(-random_search.best_score_)
        self.search_results = random_search.cv_results_
        
        print("\n" + "=" * 80)
        print("SEARCH RESULTS")
        print("=" * 80)
        print(f"\n[Best Hyperparameters]:")
        for key, value in sorted(self.best_params.items()):
            print(f"  {key}: {value}")
        print(f"\n[Best CV RMSE]: {self.best_cv_score:.4f} grams")
        
        return self.best_params, self.best_cv_score
    
    def train_final_model(self, random_state=42):
        """Step 5: Train final model with best params"""
        print("\n" + "=" * 80)
        print("STEP 5: TRAINING FINAL MODEL")
        print("=" * 80)
        
        print("\n[INFO] Training with best hyperparameters...")
        
        # Train on full training + validation set
        self.best_model = xgb.XGBRegressor(
            **self.best_params,
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=random_state,
            n_jobs=-1,
            tree_method='hist'
        )
        
        self.best_model.fit(self.X_cv, self.y_cv)
        
        print(f"[OK] Model trained!")
        print(f"  - Estimators: {self.best_model.n_estimators}")
        print(f"  - Max depth: {self.best_model.max_depth}")
        print(f"  - Learning rate: {self.best_model.learning_rate}")
        
        return self.best_model
    
    def evaluate_model(self):
        """Step 6: Evaluate on all splits"""
        print("\n" + "=" * 80)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 80)
        
        results = {}
        
        for split_name, X, y in [
            ('train', self.X_train, self.y_train),
            ('validation', self.X_val, self.y_val),
            ('test', self.X_test, self.y_test)
        ]:
            y_pred = self.best_model.predict(X)
            
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            correlation, p_value = pearsonr(y, y_pred)
            
            results[split_name] = {
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R²': float(r2),
                'MAPE': float(mape),
                'Correlation': float(correlation),
                'P-value': float(p_value),
                'Sample_Size': int(len(y))
            }
            
            print(f"\n[{split_name.upper()}]:")
            print(f"  RMSE: {rmse:.4f}g, MAE: {mae:.4f}g, R²: {r2:.4f}")
        
        self.results = results
        return results
    
    def calculate_feature_importance(self):
        """Step 7: Feature importance"""
        print("\n" + "=" * 80)
        print("STEP 7: FEATURE IMPORTANCE")
        print("=" * 80)
        
        importances = self.best_model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Calculate percentage importance
        importance_df['importance_pct'] = (importance_df['importance'] / 
                                          importance_df['importance'].sum()) * 100
        
        self.feature_importance = importance_df
        
        print("\n[Top 10 Features]:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:35s}: {row['importance_pct']:6.2f}%")
        
        return importance_df
    
    def save_results(self):
        """Step 8: Save results"""
        print("\n" + "=" * 80)
        print("STEP 8: SAVING RESULTS")
        print("=" * 80)
        
        output_dir = 'paper/results/xgboost_clean_top15'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Comprehensive results
        comprehensive_results = {
            'model_type': 'XGBoost_Clean_Top15_NoLeakage',
            'num_features': len(self.selected_features),
            'features': self.selected_features,
            'best_hyperparameters': self.best_params,
            'best_cv_rmse': float(self.best_cv_score),
            'performance_metrics': self.results,
            'timestamp': timestamp
        }
        
        # Save files
        results_file = f'{output_dir}/xgboost_top30_optimized_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Metrics CSV
        metrics_data = []
        for split in ['train', 'validation', 'test']:
            row = {'split': split}
            row.update(self.results[split])
            metrics_data.append(row)
        
        metrics_file = f'{output_dir}/xgboost_top30_optimized_metrics_{timestamp}.csv'
        pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)
        
        # Feature importance
        importance_file = f'{output_dir}/xgboost_top30_optimized_importance_{timestamp}.csv'
        self.feature_importance.to_csv(importance_file, index=False)
        
        # Search results
        search_df = pd.DataFrame(self.search_results)
        search_file = f'{output_dir}/xgboost_top30_search_results_{timestamp}.csv'
        search_df.to_csv(search_file, index=False)
        
        # Model
        model_file = f'{output_dir}/xgboost_top30_optimized_model_{timestamp}.pkl'
        joblib.dump(self.best_model, model_file)
        
        print(f"[OK] Saved all results to {output_dir}")
        
        return results_file
    
    def run_optimization(self, n_iter=50, cv_folds=5, random_state=42):
        """Run complete optimization pipeline"""
        print("=" * 80)
        print("XGBOOST HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        
        self.load_and_prepare_data()
        self.prepare_data_splits()
        param_dist = self.define_hyperparameter_search_space()
        self.perform_random_search(param_dist, n_iter=n_iter, cv_folds=cv_folds, random_state=random_state)
        self.train_final_model()
        self.evaluate_model()
        self.calculate_feature_importance()
        self.save_results()
        
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print(f"\n[FINAL RESULTS]")
        print(f"  Features:     {len(self.selected_features)}")
        print(f"  Best CV RMSE: {self.best_cv_score:.4f}g")
        print(f"  Test RMSE:    {self.results['test']['RMSE']:.4f}g")
        print(f"  Test R²:      {self.results['test']['R²']:.4f}")
        print(f"  Test Corr:    {self.results['test']['Correlation']:.4f}")
        print("=" * 80)
        
        return {
            'best_params': self.best_params,
            'best_cv_rmse': self.best_cv_score,
            'test_metrics': self.results['test']
        }


def main():
    """Main function"""
    print("=" * 80)
    print("XGBOOST OPTIMIZED - TOP 30 ENGINEERED FEATURES")
    print("=" * 80)
    
    print(f"\n[Configuration]:")
    print(f"  - Data: top30_engineered_features.csv")
    print(f"  - Optimization: RandomizedSearchCV")
    print(f"  - Iterations: 50")
    print(f"  - CV folds: 5")
    
    optimizer = XGBoostTop30Optimized()
    results = optimizer.run_optimization(n_iter=50, cv_folds=5, random_state=42)
    
    return results


if __name__ == "__main__":
    results = main()
