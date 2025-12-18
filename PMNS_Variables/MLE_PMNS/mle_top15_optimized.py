"""
MLE Optimized - PMNS Variables (20 Features from Previous Work)
================================================================

This script implements an optimized MLE model using the 20 features
identified from previous PMNS research work with hyperparameter optimization.

Author: Sujit Sarkar
Date: 2025-12-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
import json
import warnings
import os
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')


class MLEPMNSOptimized:
    """
    MLE optimized model using 20 PMNS features with hyperparameter optimization
    """
    
    def __init__(self, data_path='e:/KEM/Project/Data/PMNS_Data.csv'):
        """Initialize MLE optimized implementation"""
        self.data_path = data_path
        self.data = None
        self.mle_results = {}
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
        print("STEP 1: DATA LOADING AND PREPARATION")
        print("=" * 80)
        
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        available_features = [f for f in self.selected_features if f in self.data.columns]
        self.selected_features = available_features
        print(f"[OK] Using {len(self.selected_features)} features")
        
        return self.data
    
    def prepare_data_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Step 2: Prepare train/validation/test splits"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA SPLITTING (60/20/20)")
        print("=" * 80)
        
        all_vars = [self.target] + self.selected_features
        data_subset = self.data[all_vars].copy()
        data_subset = data_subset[data_subset[self.target].notna()]
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        features_imputed = imputer.fit_transform(data_subset[self.selected_features])
        data_subset[self.selected_features] = features_imputed
        
        print(f"[OK] Data after cleaning: {data_subset.shape[0]} samples")
        
        data_temp, data_test = train_test_split(data_subset, test_size=test_size, random_state=random_state)
        val_size_adjusted = val_size / (1 - test_size)
        data_train, data_val = train_test_split(data_temp, test_size=val_size_adjusted, random_state=random_state)
        
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        self.data_cv = pd.concat([data_train, data_val], axis=0)
        
        print(f"[OK] Splits: Train={data_train.shape[0]}, Val={data_val.shape[0]}, Test={data_test.shape[0]}")
        
        return data_train, data_val, data_test
    
    def define_hyperparameter_search_space(self):
        """Step 3: Define hyperparameter search space"""
        print("\n" + "=" * 80)
        print("STEP 3: HYPERPARAMETER SEARCH SPACE")
        print("=" * 80)
        
        param_grid = {
            'max_iter': [50, 100, 150, 200],
            'tol': [1e-4, 1e-5, 1e-6],
            'cov_reg': [1e-6, 1e-5, 1e-4],
            'shrinkage_alpha': [0.0, 0.1, 0.2],
        }
        
        print("\n[Hyperparameter Search Space]:")
        for key, values in param_grid.items():
            print(f"  {key}: {values}")
        
        return param_grid
    
    def perform_grid_search(self, param_grid, cv_folds=5, random_state=42):
        """Step 4: Perform grid search with cross-validation"""
        print("\n" + "=" * 80)
        print("STEP 4: GRID SEARCH WITH CROSS-VALIDATION")
        print("=" * 80)
        
        all_vars = [self.target] + self.selected_features
        cv_data = self.data_cv[all_vars].values
        
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        print(f"[INFO] Testing {len(param_combinations)} parameter combinations...")
        
        best_score = np.inf
        best_params = None
        results_list = []
        
        for i, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            cv_scores = []
            
            for train_idx, val_idx in kfold.split(cv_data):
                train_fold = cv_data[train_idx]
                val_fold = cv_data[val_idx]
                
                try:
                    mle_result = self._train_em_with_params(train_fold, **param_dict)
                    score = self._evaluate_on_fold(mle_result, val_fold)
                    cv_scores.append(score)
                except:
                    cv_scores.append(np.inf)
            
            mean_cv_score = np.mean(cv_scores)
            
            results_list.append({
                'params': param_dict,
                'mean_cv_rmse': mean_cv_score,
                'std_cv_rmse': np.std(cv_scores)
            })
            
            if mean_cv_score < best_score:
                best_score = mean_cv_score
                best_params = param_dict
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(param_combinations)}")
        
        self.best_params = best_params
        self.best_cv_rmse = best_score
        self.search_results = results_list
        
        print(f"\n[Best Hyperparameters]:")
        for key, value in sorted(best_params.items()):
            print(f"  {key}: {value}")
        print(f"[Best CV RMSE]: {best_score:.4f} grams")
        
        return best_params, best_score
    
    def _train_em_with_params(self, data, max_iter=100, tol=1e-6, cov_reg=1e-6, shrinkage_alpha=0.0):
        """Train EM algorithm with specific hyperparameters"""
        mean_init = np.nanmean(data, axis=0)
        
        valid_data = data[~np.isnan(data).any(axis=1)]
        if len(valid_data) > 0:
            cov_init = np.cov(valid_data.T, rowvar=True)
            if shrinkage_alpha > 0:
                target = np.eye(cov_init.shape[0]) * np.mean(np.diag(cov_init))
                cov_init = (1 - shrinkage_alpha) * cov_init + shrinkage_alpha * target
            cov_init = cov_init + np.eye(data.shape[1]) * cov_reg
        else:
            cov_init = np.eye(data.shape[1])
        
        current_mean = mean_init.copy()
        current_cov = cov_init.copy()
        prev_likelihood = -np.inf
        
        for iteration in range(max_iter):
            data_imputed = data.copy()
            for i in range(len(data)):
                if np.isnan(data[i]).any():
                    observed_mask = ~np.isnan(data[i])
                    missing_mask = np.isnan(data[i])
                    
                    if observed_mask.any():
                        mu_obs = current_mean[observed_mask]
                        mu_miss = current_mean[missing_mask]
                        cov_obs = current_cov[np.ix_(observed_mask, observed_mask)]
                        cov_miss_obs = current_cov[np.ix_(missing_mask, observed_mask)]
                        
                        try:
                            cov_obs_inv = np.linalg.inv(cov_obs + np.eye(cov_obs.shape[0]) * cov_reg)
                            conditional_mean = mu_miss + cov_miss_obs @ cov_obs_inv @ (data[i][observed_mask] - mu_obs)
                            data_imputed[i][missing_mask] = conditional_mean
                        except:
                            data_imputed[i][missing_mask] = mu_miss
                    else:
                        data_imputed[i][missing_mask] = current_mean[missing_mask]
            
            current_mean = np.mean(data_imputed, axis=0)
            current_cov = np.cov(data_imputed.T, rowvar=True)
            
            if shrinkage_alpha > 0:
                target = np.eye(current_cov.shape[0]) * np.mean(np.diag(current_cov))
                current_cov = (1 - shrinkage_alpha) * current_cov + shrinkage_alpha * target
            
            current_cov = current_cov + np.eye(data_imputed.shape[1]) * cov_reg
            
            try:
                current_likelihood = self._calculate_log_likelihood(data_imputed, current_mean, current_cov)
            except:
                current_likelihood = -np.inf
            
            if abs(current_likelihood - prev_likelihood) < tol:
                break
            
            prev_likelihood = current_likelihood
        
        return {
            'mean': current_mean,
            'covariance': current_cov,
            'likelihood': current_likelihood,
            'iterations': iteration + 1,
            'converged': abs(current_likelihood - prev_likelihood) < tol
        }
    
    def _evaluate_on_fold(self, mle_result, val_data):
        """Evaluate MLE result on validation fold"""
        val_clean = val_data[~np.isnan(val_data[:, 0])]
        
        if len(val_clean) == 0:
            return np.inf
        
        y_true = val_clean[:, 0]
        X = val_clean[:, 1:]
        
        mean_vector = mle_result['mean']
        cov_matrix = mle_result['covariance']
        
        mu_y = mean_vector[0]
        mu_x = mean_vector[1:]
        sigma_yx = cov_matrix[0, 1:]
        sigma_xx = cov_matrix[1:, 1:]
        
        try:
            sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * 1e-6)
            beta = sigma_xx_inv @ sigma_yx
            X_centered = X - mu_x
            y_pred = mu_y + X_centered @ beta
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            return rmse
        except:
            return np.inf
    
    def _calculate_log_likelihood(self, data, mean, cov):
        """Calculate log-likelihood"""
        n, p = data.shape
        cov_reg = cov + np.eye(p) * 1e-6
        
        try:
            log_lik = -0.5 * n * p * np.log(2 * np.pi)
            log_lik -= 0.5 * n * np.log(np.linalg.det(cov_reg))
            diff = data - mean
            inv_cov = np.linalg.inv(cov_reg)
            quadratic = np.sum(diff @ inv_cov * diff)
            log_lik -= 0.5 * quadratic
            return log_lik
        except:
            return -np.inf
    
    def train_final_model(self):
        """Step 5: Train final model with best hyperparameters"""
        print("\n" + "=" * 80)
        print("STEP 5: TRAINING FINAL MODEL")
        print("=" * 80)
        
        all_vars = [self.target] + self.selected_features
        train_data = self.data_train[all_vars].values
        
        self.mle_results = self._train_em_with_params(train_data, **self.best_params)
        
        print(f"[OK] Model trained:")
        print(f"  - Iterations: {self.mle_results['iterations']}")
        print(f"  - Converged: {self.mle_results['converged']}")
        
        return self.mle_results
    
    def evaluate_model(self):
        """Step 6: Evaluate model on all splits"""
        print("\n" + "=" * 80)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 80)
        
        mean_vector = self.mle_results['mean']
        cov_matrix = self.mle_results['covariance']
        
        results = {}
        for split_name, data_split in [('train', self.data_train), ('validation', self.data_val), ('test', self.data_test)]:
            all_vars = [self.target] + self.selected_features
            eval_data = data_split[all_vars].dropna()
            
            y_true = eval_data[self.target].values
            X = eval_data[self.selected_features].values
            
            mu_y = mean_vector[0]
            mu_x = mean_vector[1:]
            sigma_yx = cov_matrix[0, 1:]
            sigma_xx = cov_matrix[1:, 1:]
            
            try:
                sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * self.best_params['cov_reg'])
                beta = sigma_xx_inv @ sigma_yx
                X_centered = X - mu_x
                y_pred = mu_y + X_centered @ beta
            except:
                y_pred = np.full_like(y_true, mu_y)
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            correlation, p_value = pearsonr(y_true, y_pred)
            
            results[split_name] = {
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R²': float(r2),
                'MAPE': float(mape),
                'Correlation': float(correlation),
                'P-value': float(p_value),
                'Sample_Size': int(len(y_true))
            }
            
            print(f"\n[{split_name.upper()}]: RMSE={rmse:.4f}g, R²={r2:.4f}")
        
        self.results = results
        return results
    
    def save_results(self):
        """Step 7: Save results"""
        print("\n" + "=" * 80)
        print("STEP 7: SAVING RESULTS")
        print("=" * 80)
        
        output_dir = 'e:/KEM/Project/PMNS_Variables/Results_PMNS'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        comprehensive_results = {
            'model_type': 'MLE_Optimized_PMNS_Variables',
            'num_features': len(self.selected_features),
            'features': self.selected_features,
            'best_hyperparameters': self.best_params,
            'best_cv_rmse': float(self.best_cv_rmse),
            'performance_metrics': self.results,
            'timestamp': timestamp
        }
        
        results_file = f'{output_dir}/mle_pmns_optimized_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        metrics_data = []
        for split_name in ['train', 'validation', 'test']:
            row = {'split': split_name}
            row.update(self.results[split_name])
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f'{output_dir}/mle_pmns_optimized_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        
        print(f"[OK] Saved: {results_file}")
        
        return results_file, metrics_file
    
    def run_optimization(self, cv_folds=5, random_state=42):
        """Run complete optimization pipeline"""
        print("=" * 80)
        print("MLE OPTIMIZED - PMNS VARIABLES (20 FEATURES)")
        print("=" * 80)
        
        self.load_and_prepare_data()
        self.prepare_data_splits()
        param_grid = self.define_hyperparameter_search_space()
        self.perform_grid_search(param_grid, cv_folds=cv_folds, random_state=random_state)
        self.train_final_model()
        self.evaluate_model()
        self.save_results()
        
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE!")
        print("=" * 80)
        print(f"\n[FINAL RESULTS]")
        print(f"  Features:      {len(self.selected_features)}")
        print(f"  Best CV RMSE:  {self.best_cv_rmse:.4f}g")
        print(f"  Test RMSE:     {self.results['test']['RMSE']:.4f}g")
        print(f"  Test R²:       {self.results['test']['R²']:.4f}")
        print("=" * 80)
        
        return {
            'best_params': self.best_params,
            'best_cv_rmse': self.best_cv_rmse,
            'test_metrics': self.results['test']
        }


def main():
    """Main function"""
    data_path = 'e:/KEM/Project/Data/PMNS_Data.csv'
    optimizer = MLEPMNSOptimized(data_path=data_path)
    results = optimizer.run_optimization(cv_folds=5, random_state=42)
    return results


if __name__ == "__main__":
    results = main()
