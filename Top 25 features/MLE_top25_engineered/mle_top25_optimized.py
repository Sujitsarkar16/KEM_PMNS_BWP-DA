"""
MLE Optimized - Top 25 Variables
=================================

This script implements an optimized MLE model using the top 25 features
identified from multi-method feature importance analysis with hyperparameter tuning.

Author: Sujit Sarkar
Date: 2025-12-06
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
import json
import warnings
import os
from datetime import datetime
from itertools import product
warnings.filterwarnings('ignore')


class MLETop25Optimized:
    """
    MLE optimized model using top 25 features with hyperparameter tuning
    """
    
    def __init__(self, data_path='e:/KEM/Project/Data/Top_25_Data.csv'):
        """Initialize MLE optimized implementation"""
        self.data_path = data_path
        self.data = None
        self.mle_results = {}
        
        # Top 25 features from feature importance ranking
        self.selected_features = [
            "f0_m_GA_Del",
            "f0_m_plac_wt",
            "f0_m_abd_cir_v2",
            "f0_m_wt_v2",
            "f0_m_fundal_ht_v2",
            "f0_m_hip_circ_v2",
            "f0_m_rcf_v2",
            "f0_m_ht",
            "f0_m_bmi_v2",
            "f0_m_wt_prepreg",
            "f0_m_snacks_sc_v1",
            "f0_m_waist_circ_v2",
            "f0_m_j8_sc_v1",
            "f0_m_f10_sc_v1",
            "f0_m_bmi_6yr",
            "f0_m_age",
            "f0_m_m2_sc_v1",
            "f0_m_n_sc_v1",
            "f0_m_g1_sc_v2",
            "f0_m_o_sc_v1",
            "f0_m_g_sc_v2",
            "f0_f_wt_ini",
            "f0_m_glv_sc_v",
            "f0_m_p10_sc_v1",
            "f0_m_d1_sc_v1"
        ]
        
        self.target = 'f1_bw'
        self.best_params = None
        self.search_results = []
        
    def load_and_prepare_data(self):
        """Step 1: Load and prepare data"""
        print("=" * 80)
        print("STEP 1: DATA LOADING AND PREPARATION")
        print("=" * 80)
        
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        available_features = [f for f in self.selected_features if f in self.data.columns]
        missing_features = [f for f in self.selected_features if f not in self.data.columns]
        
        if missing_features:
            print(f"[WARNING] {len(missing_features)} features not found in dataset")
        
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
        
        imputer = SimpleImputer(strategy='median')
        features_imputed = imputer.fit_transform(data_subset[self.selected_features])
        data_subset[self.selected_features] = features_imputed
        
        data_temp, data_test = train_test_split(
            data_subset, test_size=test_size, random_state=random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        data_train, data_val = train_test_split(
            data_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        
        print(f"[OK] Train: {data_train.shape[0]}, Val: {data_val.shape[0]}, Test: {data_test.shape[0]}")
        
        return data_train, data_val, data_test
    
    def hyperparameter_search(self):
        """Step 3: Hyperparameter search for EM algorithm"""
        print("\n" + "=" * 80)
        print("STEP 3: HYPERPARAMETER SEARCH")
        print("=" * 80)
        
        # Define hyperparameter grid
        param_grid = {
            'max_iter': [50, 100, 150],
            'tol': [1e-4, 1e-5, 1e-6],
            'regularization': [1e-6, 1e-5, 1e-4]
        }
        
        print(f"\n[INFO] Hyperparameter search space:")
        for param, values in param_grid.items():
            print(f"  - {param}: {values}")
        
        best_rmse = float('inf')
        best_params = None
        
        # Grid search with cross-validation
        all_combinations = list(product(
            param_grid['max_iter'],
            param_grid['tol'],
            param_grid['regularization']
        ))
        
        print(f"\n[INFO] Total combinations to evaluate: {len(all_combinations)}")
        
        for idx, (max_iter, tol, reg) in enumerate(all_combinations):
            # Run EM with current parameters
            rmse, _ = self._run_em_with_params(max_iter, tol, reg)
            
            self.search_results.append({
                'max_iter': max_iter,
                'tol': tol,
                'regularization': reg,
                'validation_rmse': rmse
            })
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {'max_iter': max_iter, 'tol': tol, 'regularization': reg}
            
            if (idx + 1) % 9 == 0:
                print(f"  Progress: {idx + 1}/{len(all_combinations)} - Best RMSE: {best_rmse:.4f}")
        
        self.best_params = best_params
        print(f"\n[OK] Best parameters found:")
        print(f"  - Max iterations: {best_params['max_iter']}")
        print(f"  - Tolerance: {best_params['tol']}")
        print(f"  - Regularization: {best_params['regularization']}")
        print(f"  - Validation RMSE: {best_rmse:.4f}")
        
        return best_params
    
    def _run_em_with_params(self, max_iter, tol, regularization):
        """Run EM algorithm with specific parameters"""
        all_vars = [self.target] + self.selected_features
        train_data = self.data_train[all_vars].values
        
        mean_init = np.nanmean(train_data, axis=0)
        valid_data = train_data[~np.isnan(train_data).any(axis=1)]
        
        if len(valid_data) > 0:
            cov_init = np.cov(valid_data.T, rowvar=True) + np.eye(train_data.shape[1]) * regularization
        else:
            cov_init = np.eye(train_data.shape[1])
        
        current_mean = mean_init.copy()
        current_cov = cov_init.copy()
        prev_likelihood = -np.inf
        
        for iteration in range(max_iter):
            data_imputed = train_data.copy()
            for i in range(len(train_data)):
                if np.isnan(train_data[i]).any():
                    observed_mask = ~np.isnan(train_data[i])
                    missing_mask = np.isnan(train_data[i])
                    
                    if observed_mask.any():
                        mu_obs = current_mean[observed_mask]
                        mu_miss = current_mean[missing_mask]
                        cov_obs = current_cov[np.ix_(observed_mask, observed_mask)]
                        cov_miss_obs = current_cov[np.ix_(missing_mask, observed_mask)]
                        
                        try:
                            cov_obs_inv = np.linalg.inv(cov_obs + np.eye(cov_obs.shape[0]) * regularization)
                            conditional_mean = mu_miss + cov_miss_obs @ cov_obs_inv @ (train_data[i][observed_mask] - mu_obs)
                            data_imputed[i][missing_mask] = conditional_mean
                        except:
                            data_imputed[i][missing_mask] = mu_miss
                    else:
                        data_imputed[i][missing_mask] = current_mean[missing_mask]
            
            current_mean = np.mean(data_imputed, axis=0)
            current_cov = np.cov(data_imputed.T, rowvar=True) + np.eye(data_imputed.shape[1]) * regularization
            
            try:
                n, p = data_imputed.shape
                log_lik = -0.5 * n * p * np.log(2 * np.pi)
                log_lik -= 0.5 * n * np.log(np.linalg.det(current_cov))
                diff = data_imputed - current_mean
                inv_cov = np.linalg.inv(current_cov)
                log_lik -= 0.5 * np.sum(diff @ inv_cov * diff)
                current_likelihood = log_lik
            except:
                current_likelihood = -np.inf
            
            if abs(current_likelihood - prev_likelihood) < tol:
                break
            prev_likelihood = current_likelihood
        
        # Evaluate on validation set
        val_data = self.data_val[all_vars].dropna()
        y_true = val_data[self.target].values
        X = val_data[self.selected_features].values
        
        mu_y = current_mean[0]
        mu_x = current_mean[1:]
        sigma_yx = current_cov[0, 1:]
        sigma_xx = current_cov[1:, 1:]
        
        try:
            sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * regularization)
            beta = sigma_xx_inv @ sigma_yx
            X_centered = X - mu_x
            y_pred = mu_y + X_centered @ beta
        except:
            y_pred = np.full_like(y_true, mu_y)
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        return rmse, {'mean': current_mean, 'covariance': current_cov}
    
    def train_final_model(self):
        """Step 4: Train final model with best parameters"""
        print("\n" + "=" * 80)
        print("STEP 4: TRAINING FINAL MODEL")
        print("=" * 80)
        
        max_iter = self.best_params['max_iter']
        tol = self.best_params['tol']
        reg = self.best_params['regularization']
        
        print(f"\n[INFO] Training with best parameters:")
        print(f"  - Max iterations: {max_iter}")
        print(f"  - Tolerance: {tol}")
        print(f"  - Regularization: {reg}")
        
        all_vars = [self.target] + self.selected_features
        train_data = self.data_train[all_vars].values
        
        mean_init = np.nanmean(train_data, axis=0)
        valid_data = train_data[~np.isnan(train_data).any(axis=1)]
        
        if len(valid_data) > 0:
            cov_init = np.cov(valid_data.T, rowvar=True) + np.eye(train_data.shape[1]) * reg
        else:
            cov_init = np.eye(train_data.shape[1])
        
        current_mean = mean_init.copy()
        current_cov = cov_init.copy()
        prev_likelihood = -np.inf
        likelihood_history = []
        
        for iteration in range(max_iter):
            data_imputed = train_data.copy()
            for i in range(len(train_data)):
                if np.isnan(train_data[i]).any():
                    observed_mask = ~np.isnan(train_data[i])
                    missing_mask = np.isnan(train_data[i])
                    
                    if observed_mask.any():
                        mu_obs = current_mean[observed_mask]
                        mu_miss = current_mean[missing_mask]
                        cov_obs = current_cov[np.ix_(observed_mask, observed_mask)]
                        cov_miss_obs = current_cov[np.ix_(missing_mask, observed_mask)]
                        
                        try:
                            cov_obs_inv = np.linalg.inv(cov_obs + np.eye(cov_obs.shape[0]) * reg)
                            conditional_mean = mu_miss + cov_miss_obs @ cov_obs_inv @ (train_data[i][observed_mask] - mu_obs)
                            data_imputed[i][missing_mask] = conditional_mean
                        except:
                            data_imputed[i][missing_mask] = mu_miss
                    else:
                        data_imputed[i][missing_mask] = current_mean[missing_mask]
            
            current_mean = np.mean(data_imputed, axis=0)
            current_cov = np.cov(data_imputed.T, rowvar=True) + np.eye(data_imputed.shape[1]) * reg
            
            try:
                n, p = data_imputed.shape
                log_lik = -0.5 * n * p * np.log(2 * np.pi)
                log_lik -= 0.5 * n * np.log(np.linalg.det(current_cov))
                diff = data_imputed - current_mean
                inv_cov = np.linalg.inv(current_cov)
                log_lik -= 0.5 * np.sum(diff @ inv_cov * diff)
                current_likelihood = log_lik
            except:
                current_likelihood = -np.inf
            
            likelihood_history.append(current_likelihood)
            
            if abs(current_likelihood - prev_likelihood) < tol:
                print(f"\n[OK] Converged at iteration {iteration + 1}")
                break
            prev_likelihood = current_likelihood
            
            if (iteration + 1) % 50 == 0:
                print(f"  Iteration {iteration + 1}: log-likelihood = {current_likelihood:.4f}")
        
        self.mle_results = {
            'mean': current_mean,
            'covariance': current_cov,
            'likelihood': current_likelihood,
            'iterations': iteration + 1,
            'likelihood_history': likelihood_history,
            'converged': abs(current_likelihood - prev_likelihood) < tol
        }
        
        print(f"\n[FINAL] EM Results:")
        print(f"  - Iterations: {self.mle_results['iterations']}")
        print(f"  - Converged: {self.mle_results['converged']}")
        
        return self.mle_results
    
    def evaluate_model(self):
        """Step 5: Evaluate model on all splits"""
        print("\n" + "=" * 80)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 80)
        
        mean_vector = self.mle_results['mean']
        cov_matrix = self.mle_results['covariance']
        reg = self.best_params['regularization']
        
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
                sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * reg)
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
            
            print(f"\n[{split_name.upper()} SET] Performance:")
            print(f"  - RMSE:         {rmse:.4f} grams")
            print(f"  - MAE:          {mae:.4f} grams")
            print(f"  - R²:           {r2:.4f}")
            print(f"  - Correlation:  {correlation:.4f}")
        
        self.results = results
        return results
    
    def save_results(self):
        """Step 6: Save results"""
        print("\n" + "=" * 80)
        print("STEP 6: SAVING RESULTS")
        print("=" * 80)
        
        output_dir = 'e:/KEM/Project/Top 25 features/Results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        comprehensive_results = {
            'model_type': 'MLE_Optimized_Top25_Variables',
            'num_features': len(self.selected_features),
            'features': self.selected_features,
            'best_parameters': self.best_params,
            'convergence': {
                'converged': bool(self.mle_results['converged']),
                'iterations': int(self.mle_results['iterations']),
                'final_likelihood': float(self.mle_results['likelihood'])
            },
            'performance_metrics': {
                'train': self.results['train'],
                'validation': self.results['validation'],
                'test': self.results['test']
            },
            'timestamp': timestamp
        }
        
        results_file = f'{output_dir}/mle_top25_optimized_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        metrics_data = []
        for split_name in ['train', 'validation', 'test']:
            row = {'split': split_name}
            row.update(self.results[split_name])
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f'{output_dir}/mle_top25_optimized_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save search results
        search_df = pd.DataFrame(self.search_results)
        search_file = f'{output_dir}/mle_top25_search_results_{timestamp}.csv'
        search_df.to_csv(search_file, index=False)
        
        print(f"[OK] Results saved:")
        print(f"  - Results JSON: {results_file}")
        print(f"  - Metrics CSV: {metrics_file}")
        print(f"  - Search results: {search_file}")
        
        return results_file, metrics_file
    
    def run_optimized(self):
        """Run complete optimized pipeline"""
        print("=" * 80)
        print("MLE OPTIMIZED MODEL - TOP 25 VARIABLES")
        print("=" * 80)
        
        self.load_and_prepare_data()
        self.prepare_data_splits()
        self.hyperparameter_search()
        self.train_final_model()
        self.evaluate_model()
        self.save_results()
        
        print("\n" + "=" * 80)
        print("OPTIMIZED MLE MODEL TRAINING COMPLETED!")
        print("=" * 80)
        print(f"\n[FINAL RESULTS]")
        print(f"  Number of features: {len(self.selected_features)}")
        print(f"  Test RMSE:          {self.results['test']['RMSE']:.4f} grams")
        print(f"  Test R²:            {self.results['test']['R²']:.4f}")
        print(f"  Test MAE:           {self.results['test']['MAE']:.4f} grams")
        print("=" * 80)
        
        return {
            'mle_results': self.mle_results,
            'test_metrics': self.results['test'],
            'results': self.results,
            'best_params': self.best_params
        }


def main():
    """Main function"""
    data_path = 'e:/KEM/Project/Data/Top_25_Data.csv'
    
    print(f"\n[Configuration]:")
    print(f"  - Using Top 25 features")
    print(f"  - Data source: {data_path}")
    
    optimized = MLETop25Optimized(data_path=data_path)
    results = optimized.run_optimized()
    
    return results


if __name__ == "__main__":
    results = main()
