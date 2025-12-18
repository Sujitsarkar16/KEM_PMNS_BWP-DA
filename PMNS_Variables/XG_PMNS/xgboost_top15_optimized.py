"""
XGBoost Optimized - PMNS Variables (20 Features from Previous Work)
====================================================================

This script implements an optimized XGBoost model using the 20 features
identified from previous PMNS research work with hyperparameter optimization.

Author: Sujit Sarkar
Date: 2025-12-06
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import xgboost as xgb
import json
import warnings
import os
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')


class XGBoostPMNSOptimized:
    """
    XGBoost optimized model with hyperparameter tuning for PMNS features
    """
    
    def __init__(self, data_path='e:/KEM/Project/Data/PMNS_Data.csv'):
        """Initialize XGBoost optimized implementation"""
        self.data_path = data_path
        self.data = None
        self.model = None
        self.best_model = None
        self.results = {}
        self.best_params = {}
        
        # PMNS Features from previous work (20 features)
        self.selected_features = [
            'f0_m_parity_v1', 'f0_m_wt_prepreg', 'f0_m_fundal_ht_v2', 
            'f0_m_abd_cir_v2', 'f0_m_wt_v2', 'f0_m_r4_v2', 'f0_m_lunch_cal_v1',
            'f0_m_p_sc_v1', 'f0_m_o_sc_v1', 'f0_m_pulse_r1_v2', 'f0_m_pulse_r2_v2',
            'f0_m_glu_f_v2', 'f0_m_rcf_v2', 'f0_m_g_sc_v1', 'f0_m_plac_wt',
            'f0_m_GA_Del', 'f0_f_head_cir_ini', 'f0_f_plt_ini', 'f1_sex', 
            'f0_m_age_eld_child'
        ]
        
        self.target = 'f1_bw'
        
    def load_and_prepare_data(self):
        """Step 1: Load and prepare data"""
        print("=" * 80)
        print("STEP 1: DATA LOADING")
        print("=" * 80)
        
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        available_features = [f for f in self.selected_features if f in self.data.columns]
        self.selected_features = available_features
        print(f"[OK] Using {len(self.selected_features)} features")
        
        return self.data
    
    def prepare_data_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Step 2: Prepare splits"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA SPLITTING")
        print("=" * 80)
        
        X = self.data[self.selected_features].values
        y = self.data[self.target].values
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
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
        
        self.X_cv = np.vstack([X_train, X_val])
        self.y_cv = np.concatenate([y_train, y_val])
        
        print(f"[OK] Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def define_hyperparameter_search_space(self):
        """Step 3: Define search space"""
        print("\n" + "=" * 80)
        print("STEP 3: HYPERPARAMETER SEARCH SPACE")
        print("=" * 80)
        
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'min_child_weight': [3, 5, 7, 10],
            'gamma': [0, 0.1, 0.2, 0.3]
        }
        
        print("\n[Hyperparameter Grid]:")
        for key, values in param_distributions.items():
            print(f"  {key}: {values}")
        
        return param_distributions
    
    def perform_random_search(self, param_distributions, n_iter=50, cv_folds=5, random_state=42):
        """Step 4: Randomized hyperparameter search"""
        print("\n" + "=" * 80)
        print("STEP 4: RANDOMIZED HYPERPARAMETER SEARCH")
        print("=" * 80)
        
        print(f"\n[INFO] Running {n_iter} iterations with {cv_folds}-fold CV...")
        
        xgb_base = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=random_state,
            n_jobs=-1,
            tree_method='hist'
        )
        
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
        
        random_search.fit(self.X_cv, self.y_cv)
        
        self.best_params = random_search.best_params_
        self.best_cv_score = np.sqrt(-random_search.best_score_)
        self.search_results = random_search.cv_results_
        
        print(f"\n[Best Hyperparameters]:")
        for key, value in sorted(self.best_params.items()):
            print(f"  {key}: {value}")
        print(f"[Best CV RMSE]: {self.best_cv_score:.4f} grams")
        
        return self.best_params, self.best_cv_score
    
    def train_final_model(self, random_state=42):
        """Step 5: Train final model with best params"""
        print("\n" + "=" * 80)
        print("STEP 5: TRAINING FINAL MODEL")
        print("=" * 80)
        
        self.best_model = xgb.XGBRegressor(
            **self.best_params,
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=random_state,
            n_jobs=-1,
            tree_method='hist'
        )
        
        self.best_model.fit(self.X_cv, self.y_cv)
        
        print(f"[OK] Model trained with {self.best_model.n_estimators} estimators")
        
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
            
            print(f"\n[{split_name.upper()}]: RMSE={rmse:.4f}g, R²={r2:.4f}")
        
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
        
        importance_df['importance_pct'] = (importance_df['importance'] / 
                                          importance_df['importance'].sum()) * 100
        
        self.feature_importance = importance_df
        
        print("\n[Top 10 Features]:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:30s}: {row['importance_pct']:6.2f}%")
        
        return importance_df
    
    def save_results(self):
        """Step 8: Save results"""
        print("\n" + "=" * 80)
        print("STEP 8: SAVING RESULTS")
        print("=" * 80)
        
        output_dir = 'e:/KEM/Project/PMNS_Variables/Results_PMNS'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        comprehensive_results = {
            'model_type': 'XGBoost_Optimized_PMNS_Variables',
            'num_features': len(self.selected_features),
            'features': self.selected_features,
            'best_hyperparameters': self.best_params,
            'best_cv_rmse': float(self.best_cv_score),
            'performance_metrics': self.results,
            'timestamp': timestamp
        }
        
        results_file = f'{output_dir}/xgboost_pmns_optimized_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        metrics_data = []
        for split in ['train', 'validation', 'test']:
            row = {'split': split}
            row.update(self.results[split])
            metrics_data.append(row)
        
        metrics_file = f'{output_dir}/xgboost_pmns_optimized_metrics_{timestamp}.csv'
        pd.DataFrame(metrics_data).to_csv(metrics_file, index=False)
        
        importance_file = f'{output_dir}/xgboost_pmns_optimized_importance_{timestamp}.csv'
        self.feature_importance.to_csv(importance_file, index=False)
        
        model_file = f'{output_dir}/xgboost_pmns_optimized_model_{timestamp}.pkl'
        joblib.dump(self.best_model, model_file)
        
        print(f"[OK] Saved all results to {output_dir}")
        
        return results_file
    
    def run_optimization(self, n_iter=50, cv_folds=5, random_state=42):
        """Run complete optimization pipeline"""
        print("=" * 80)
        print("XGBOOST OPTIMIZED - PMNS VARIABLES (20 FEATURES)")
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
        print(f"  Features:      {len(self.selected_features)}")
        print(f"  Best CV RMSE:  {self.best_cv_score:.4f}g")
        print(f"  Test RMSE:     {self.results['test']['RMSE']:.4f}g")
        print(f"  Test R²:       {self.results['test']['R²']:.4f}")
        print("=" * 80)
        
        return {
            'best_params': self.best_params,
            'best_cv_rmse': self.best_cv_score,
            'test_metrics': self.results['test']
        }


def main():
    """Main function"""
    data_path = 'e:/KEM/Project/Data/PMNS_Data.csv'
    optimizer = XGBoostPMNSOptimized(data_path=data_path)
    results = optimizer.run_optimization(n_iter=50, cv_folds=5, random_state=42)
    return results


if __name__ == "__main__":
    results = main()
