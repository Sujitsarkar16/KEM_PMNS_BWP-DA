"""
MLE Baseline - Top 25 Variables
================================

This script implements a baseline MLE model using the top 25 features
identified from multi-method feature importance analysis.

Top 25 Features (Based on Combined Score):
- Gestational variables: GA_Del
- Anthropometric: plac_wt, abd_cir_v2, wt_v2, fundal_ht_v2, hip_circ_v2, ht, bmi_v2, wt_prepreg, waist_circ_v2
- Biochemistry: rcf_v2
- Dietary: snacks_sc_v1, j8_sc_v1, f10_sc_v1, m2_sc_v1, n_sc_v1, o_sc_v1, g_sc_v2, g1_sc_v2, glv_sc_v, p10_sc_v1, d1_sc_v1
- Demographics: age, bmi_6yr
- Paternal: f_wt_ini

Author: Sujit Sarkar
Date: 2025-12-06
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import json
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')


class MLETop25Baseline:
    """
    MLE baseline model using top 25 features
    """
    
    def __init__(self, data_path='e:/KEM/Project/Data/Top_25_Data.csv'):
        """Initialize MLE baseline implementation"""
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
        
        self.target = 'f1_bw'  # Birthweight (target variable)
        
    def load_and_prepare_data(self):
        """Step 1: Load and prepare data"""
        print("=" * 80)
        print("STEP 1: DATA LOADING AND PREPARATION")
        print("=" * 80)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Check which features exist in the dataset
        available_features = [f for f in self.selected_features if f in self.data.columns]
        missing_features = [f for f in self.selected_features if f not in self.data.columns]
        
        if missing_features:
            print(f"[WARNING] {len(missing_features)} features not found in dataset:")
            for feat in missing_features:
                print(f"  - {feat}")
        
        self.selected_features = available_features
        print(f"[OK] Using {len(self.selected_features)} features")
        
        # Display feature list
        print("\n[TOP 25 FEATURES]:")
        for i, feat in enumerate(self.selected_features, 1):
            print(f"  {i:2d}. {feat}")
        
        return self.data
    
    def prepare_data_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Step 2: Prepare train/validation/test splits (60/20/20)"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA SPLITTING (60/20/20)")
        print("=" * 80)
        
        # Prepare features and target
        all_vars = [self.target] + self.selected_features
        data_subset = self.data[all_vars].copy()
        
        # Remove rows with missing target
        data_subset = data_subset[data_subset[self.target].notna()]
        
        print(f"[OK] Data after removing missing target: {data_subset.shape[0]} samples")
        
        # Impute missing values in features with median
        imputer = SimpleImputer(strategy='median')
        features_imputed = imputer.fit_transform(data_subset[self.selected_features])
        data_subset[self.selected_features] = features_imputed
        
        print(f"[OK] Missing values imputed using median strategy")
        
        # Split data (60/20/20)
        data_temp, data_test = train_test_split(
            data_subset, test_size=test_size, random_state=random_state
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        data_train, data_val = train_test_split(
            data_temp, test_size=val_size_adjusted, random_state=random_state
        )
        
        # Store splits
        self.data_train = data_train
        self.data_val = data_val
        self.data_test = data_test
        
        print(f"\n[OK] Data splits created:")
        print(f"  - Training set:   {data_train.shape[0]} samples ({data_train.shape[0]/(data_train.shape[0]+data_val.shape[0]+data_test.shape[0])*100:.1f}%)")
        print(f"  - Validation set: {data_val.shape[0]} samples ({data_val.shape[0]/(data_train.shape[0]+data_val.shape[0]+data_test.shape[0])*100:.1f}%)")
        print(f"  - Test set:       {data_test.shape[0]} samples ({data_test.shape[0]/(data_train.shape[0]+data_val.shape[0]+data_test.shape[0])*100:.1f}%)")
        
        return data_train, data_val, data_test
    
    def implement_em_algorithm(self, max_iter=50, tol=1e-6):
        """Step 3: Implement EM algorithm for parameter estimation"""
        print("\n" + "=" * 80)
        print("STEP 3: EM ALGORITHM IMPLEMENTATION")
        print("=" * 80)
        
        print(f"\n[INFO] Running EM algorithm with:")
        print(f"  - Max iterations: {max_iter}")
        print(f"  - Tolerance: {tol}")
        
        # Prepare training data (target + features)
        all_vars = [self.target] + self.selected_features
        train_data = self.data_train[all_vars].values
        
        # Initialize parameters
        print("\n[INFO] Initializing parameters...")
        mean_init = np.nanmean(train_data, axis=0)
        
        # Calculate covariance with available data
        valid_data = train_data[~np.isnan(train_data).any(axis=1)]
        if len(valid_data) > 0:
            cov_init = np.cov(valid_data.T, rowvar=True) + np.eye(train_data.shape[1]) * 1e-6
        else:
            cov_init = np.eye(train_data.shape[1])
        
        current_mean = mean_init.copy()
        current_cov = cov_init.copy()
        prev_likelihood = -np.inf
        likelihood_history = []
        
        print(f"  - Initial mean shape: {current_mean.shape}")
        print(f"  - Initial covariance shape: {current_cov.shape}")
        
        # EM iterations
        for iteration in range(max_iter):
            # E-step: Impute missing values
            data_imputed = train_data.copy()
            for i in range(len(train_data)):
                if np.isnan(train_data[i]).any():
                    observed_mask = ~np.isnan(train_data[i])
                    missing_mask = np.isnan(train_data[i])
                    
                    if observed_mask.any():
                        # Conditional mean for missing values
                        mu_obs = current_mean[observed_mask]
                        mu_miss = current_mean[missing_mask]
                        
                        cov_obs = current_cov[np.ix_(observed_mask, observed_mask)]
                        cov_miss_obs = current_cov[np.ix_(missing_mask, observed_mask)]
                        
                        try:
                            cov_obs_inv = np.linalg.inv(cov_obs + np.eye(cov_obs.shape[0]) * 1e-6)
                            conditional_mean = mu_miss + cov_miss_obs @ cov_obs_inv @ (train_data[i][observed_mask] - mu_obs)
                            data_imputed[i][missing_mask] = conditional_mean
                        except:
                            data_imputed[i][missing_mask] = mu_miss
                    else:
                        data_imputed[i][missing_mask] = current_mean[missing_mask]
            
            # M-step: Update parameters
            current_mean = np.mean(data_imputed, axis=0)
            current_cov = np.cov(data_imputed.T, rowvar=True) + np.eye(data_imputed.shape[1]) * 1e-6
            
            # Calculate likelihood
            try:
                current_likelihood = self._calculate_log_likelihood(data_imputed, current_mean, current_cov)
            except:
                current_likelihood = -np.inf
            
            likelihood_history.append(current_likelihood)
            
            # Check convergence
            if abs(current_likelihood - prev_likelihood) < tol:
                print(f"\n[OK] EM algorithm converged at iteration {iteration + 1}")
                break
            
            prev_likelihood = current_likelihood
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: log-likelihood = {current_likelihood:.4f}")
        
        # Store results
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
        print(f"  - Final log-likelihood: {current_likelihood:.4f}")
        
        return self.mle_results
    
    def _calculate_log_likelihood(self, data, mean, cov):
        """Calculate log-likelihood for multivariate normal"""
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
    
    def evaluate_model(self):
        """Step 4: Evaluate model on all splits"""
        print("\n" + "=" * 80)
        print("STEP 4: MODEL EVALUATION")
        print("=" * 80)
        
        # Get MLE parameters
        mean_vector = self.mle_results['mean']
        cov_matrix = self.mle_results['covariance']
        
        # Evaluate on each split
        results = {}
        for split_name, data_split in [('train', self.data_train), ('validation', self.data_val), ('test', self.data_test)]:
            # Prepare data
            all_vars = [self.target] + self.selected_features
            eval_data = data_split[all_vars].dropna()
            
            # Extract actual birthweight
            y_true = eval_data[self.target].values
            
            # Extract predictors
            X = eval_data[self.selected_features].values
            
            # Make predictions using conditional distribution
            # E[Y|X] = μ_y + Σ_yx * Σ_xx^(-1) * (X - μ_x)
            mu_y = mean_vector[0]  # Target mean (first variable)
            mu_x = mean_vector[1:]  # Predictor means
            
            sigma_yy = cov_matrix[0, 0]  # Target variance
            sigma_yx = cov_matrix[0, 1:]  # Target-predictor covariances
            sigma_xx = cov_matrix[1:, 1:]  # Predictor covariance matrix
            
            try:
                sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * 1e-6)
                beta = sigma_xx_inv @ sigma_yx
                X_centered = X - mu_x
                y_pred = mu_y + X_centered @ beta
            except:
                print(f"[WARNING] Using fallback prediction for {split_name}")
                y_pred = np.full_like(y_true, mu_y)
            
            # Calculate metrics
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
            print(f"  - Sample size:  {len(y_true)}")
            print(f"  - RMSE:         {rmse:.4f} grams")
            print(f"  - MAE:          {mae:.4f} grams")
            print(f"  - R²:           {r2:.4f}")
            print(f"  - Correlation:  {correlation:.4f}")
            
            # Store predictions for visualization
            results[f'{split_name}_predictions'] = y_pred.tolist()
            results[f'{split_name}_actual'] = y_true.tolist()
        
        self.results = results
        return results
    
    def save_results(self):
        """Step 5: Save results"""
        print("\n" + "=" * 80)
        print("STEP 5: SAVING RESULTS")
        print("=" * 80)
        
        # Create output directory
        output_dir = 'e:/KEM/Project/Top 25 features/Results'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare comprehensive results
        comprehensive_results = {
            'model_type': 'MLE_Baseline_Top25_Variables',
            'num_features': len(self.selected_features),
            'features': self.selected_features,
            'parameters': 'Default EM parameters (max_iter=50, tol=1e-6)',
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
        
        # Save JSON results
        results_file = f'{output_dir}/mle_top25_baseline_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Save metrics to CSV
        metrics_data = []
        for split_name in ['train', 'validation', 'test']:
            row = {'split': split_name}
            row.update(self.results[split_name])
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f'{output_dir}/mle_top25_baseline_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        
        print(f"[OK] Results saved:")
        print(f"  - Results JSON: {results_file}")
        print(f"  - Metrics CSV: {metrics_file}")
        
        return results_file, metrics_file
    
    def run_baseline(self):
        """Run complete baseline pipeline"""
        print("=" * 80)
        print("MLE BASELINE MODEL - TOP 25 VARIABLES")
        print("=" * 80)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Prepare data splits
        self.prepare_data_splits()
        
        # Step 3: Run EM algorithm
        self.implement_em_algorithm(max_iter=50, tol=1e-6)
        
        # Step 4: Evaluate model
        self.evaluate_model()
        
        # Step 5: Save results
        self.save_results()
        
        print("\n" + "=" * 80)
        print("BASELINE MLE MODEL TRAINING COMPLETED!")
        print("=" * 80)
        print(f"\n[FINAL RESULTS]")
        print(f"  Number of features: {len(self.selected_features)}")
        print(f"  Convergence:        {self.mle_results['converged']}")
        print(f"  Iterations:         {self.mle_results['iterations']}")
        print(f"  Test RMSE:          {self.results['test']['RMSE']:.4f} grams")
        print(f"  Test R²:            {self.results['test']['R²']:.4f}")
        print(f"  Test MAE:           {self.results['test']['MAE']:.4f} grams")
        print(f"  Test Correlation:   {self.results['test']['Correlation']:.4f}")
        print("=" * 80)
        
        return {
            'mle_results': self.mle_results,
            'test_metrics': self.results['test'],
            'results': self.results
        }


def main():
    """Main function"""
    print("=" * 80)
    print("MLE BASELINE MODEL - TOP 25 VARIABLES")
    print("=" * 80)
    
    # Configuration
    data_path = 'e:/KEM/Project/Data/Top_25_Data.csv'
    
    print(f"\n[Configuration]:")
    print(f"  - Using Top 25 features from feature importance ranking")
    print(f"  - Data source: {data_path}")
    print(f"  - EM algorithm: max_iter=50, tol=1e-6")
    print(f"  - Data split: 60% train, 20% validation, 20% test")
    
    # Initialize and run baseline
    baseline = MLETop25Baseline(data_path=data_path)
    results = baseline.run_baseline()
    
    return results


if __name__ == "__main__":
    results = main()
