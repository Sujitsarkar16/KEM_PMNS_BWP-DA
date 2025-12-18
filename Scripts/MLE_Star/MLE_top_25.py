"""
Optimized MLE Implementation using Top 25 Variables from Feature Importance Ranking
==================================================================================

This script implements an enhanced MLE system using the TOP 25 variables
from feature importance analysis to improve birthweight prediction accuracy.

Key Enhancement: Uses top 25 variables from feature importance ranking
- Variables selected based on combined_score from feature importance analysis
- Includes the target variable f1_bw plus top 25 predictors

Author: Sujit sarkar
Date: 2025-11-02
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import json
import warnings
import os
warnings.filterwarnings('ignore')

class OptimizedMLE:
    """
    Optimized MLE implementation with top 25 variables from feature importance ranking
    """
    
    def __init__(self, data_path, ranking_path='Data/processed/MLE_New/feature_importance_ranking.csv'):
        """Initialize Optimized MLE implementation"""
        self.data_path = data_path
        self.ranking_path = ranking_path
        self.data = None
        self.variable_groups = None
        self.continuous_vars = []
        self.categorical_vars = []
        self.mle_results = {}
        
    def load_and_analyze_data(self):
        """Step 1: Data Structure Understanding"""
        print("=" * 80)
        print("STEP 1: DATA STRUCTURE UNDERSTANDING (TOP 25 VARIABLES)")
        print("=" * 80)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Load variable grouping
        self.variable_groups = pd.read_csv('Data/processed/variable_grouping_table.csv')
        print(f"[OK] Loaded variable grouping: {len(self.variable_groups)} variables categorized")
        
        # Load top 25 variables from feature importance ranking
        self._load_top_25_variables()
        
        # Classify variables
        self._classify_variables()
        
        # Analyze missing data patterns
        self._analyze_missing_patterns()
        
        print("[OK] Data structure analysis completed")
        return self.data, self.variable_groups
    
    def _load_top_25_variables(self):
        """Load top 25 variables from feature importance ranking"""
        print("\n--- Loading Top 25 Variables from Feature Importance Ranking ---")
        
        # Read feature importance ranking
        ranking_df = pd.read_csv(self.ranking_path)
        
        # Get top 25 variables (excluding header row)
        top_25_vars = ranking_df['variable'].head(25).tolist()
        
        print(f"[OK] Loaded top 25 variables from feature importance ranking")
        
        # Store for later use
        self.top_25_vars = top_25_vars
        
        # Print top 25 variables with their scores
        print("\n[TOP 25 VARIABLES]:")
        for i, (idx, row) in enumerate(ranking_df.head(25).iterrows(), 1):
            print(f"  {i:2d}. {row['variable']:30s} (score: {row['combined_score']:.2f})")
    
    def _classify_variables(self):
        """Classify variables into continuous and categorical"""
        print("\n--- Variable Classification (TOP 25) ---")
        
        # Get variable types from grouping table
        continuous_mask = self.variable_groups['Type'] == 'Continuous'
        categorical_mask = self.variable_groups['Type'] == 'Categorical'
        
        self.continuous_vars = self.variable_groups[continuous_mask]['Variable'].tolist()
        self.categorical_vars = self.variable_groups[categorical_mask]['Variable'].tolist()
        
        # Filter to only include variables present in data
        self.continuous_vars = [var for var in self.continuous_vars if var in self.data.columns]
        self.categorical_vars = [var for var in self.categorical_vars if var in self.data.columns]
        
        print(f"[OK] Continuous variables: {len(self.continuous_vars)}")
        print(f"[OK] Categorical variables: {len(self.categorical_vars)}")
        
        # Use top 25 variables + target variable (f1_bw)
        self.key_continuous = ['f1_bw'] + self.top_25_vars  # Target first, then top 25
        
        # Filter to only include variables present in data
        self.key_continuous = [var for var in self.key_continuous if var in self.data.columns]
        
        # For now, we'll treat all as continuous (or we can separate if needed)
        # Most of these should be continuous based on the naming
        self.key_categorical = []  # We can add categorical variables if needed
        
        print(f"\n[TOP 25 MODEL] Using {len(self.key_continuous)} variables:")
        print("  Target variable: f1_bw (Birthweight)")
        print("  Top 25 predictor variables:")
        for i, var in enumerate(self.key_continuous[1:], 1):  # Skip target
            if i <= 25:  # Show all 25
                print(f"    {i:2d}. {var}")
    
    def _analyze_missing_patterns(self):
        """Analyze missing data patterns"""
        print("\n--- Missing Data Analysis ---")
        
        # Calculate missing percentages
        missing_pct = (self.data.isnull().sum() / len(self.data)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        print(f"[OK] High missing (>20%): {len(missing_pct[missing_pct > 20])} variables")
        print(f"[OK] Moderate missing (5-20%): {len(missing_pct[(missing_pct > 5) & (missing_pct <= 20)])} variables")
        print(f"[OK] Low missing (0-5%): {len(missing_pct[(missing_pct > 0) & (missing_pct <= 5)])} variables")
        
        # Analyze missing for key variables
        key_vars = self.key_continuous + self.key_categorical
        key_vars = [var for var in key_vars if var in self.data.columns]
        
        print(f"\n[TOP 25 MODEL] Missing data in key variables:")
        for var in key_vars:
            miss_pct = (self.data[var].isnull().sum() / len(self.data)) * 100
            if miss_pct > 0:
                print(f"  - {var}: {miss_pct:.2f}%")
    
    def select_probability_models(self):
        """Step 2: Choose Probability Models"""
        print("\n" + "=" * 80)
        print("STEP 2: PROBABILITY MODEL SELECTION")
        print("=" * 80)
        
        # Test normality for continuous variables
        self._test_normality()
        
        # Define model specifications
        self._define_model_specifications()
        
        print("[OK] Probability model selection completed")
        return self.model_specs
    
    def _test_normality(self):
        """Test normality of continuous variables"""
        print("\n--- Normality Testing ---")
        
        self.normality_results = {}
        
        for var in self.key_continuous:
            if var in self.data.columns:
                data_clean = self.data[var].dropna()
                if len(data_clean) > 3:
                    stat, p_value = stats.shapiro(data_clean)
                    self.normality_results[var] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'is_normal': p_value > 0.05
                    }
        
        normal_vars = [var for var, result in self.normality_results.items() if result['is_normal']]
        non_normal_vars = [var for var, result in self.normality_results.items() if not result['is_normal']]
        
        print(f"[OK] Normal variables: {len(normal_vars)}")
        print(f"[OK] Non-normal variables: {len(non_normal_vars)}")
    
    def _define_model_specifications(self):
        """Define model specifications for different variable types"""
        print("\n--- Model Specifications ---")
        
        self.model_specs = {
            'continuous': {
                'distribution': 'multivariate_normal',
                'parameters': ['mean_vector', 'covariance_matrix'],
                'variables': self.continuous_vars,
                'normal_variables': [var for var, result in self.normality_results.items() if result['is_normal']]
            },
            'categorical': {
                'distribution': 'multinomial',
                'parameters': ['probability_vectors'],
                'variables': self.categorical_vars
            },
            'mixed_model': {
                'framework': 'joint_likelihood',
                'continuous_part': 'multivariate_normal',
                'categorical_part': 'multinomial',
                'integration': 'expectation_maximization'
            }
        }
        
        print("[OK] Model specifications defined")
    
    def implement_likelihood_functions(self):
        """Step 3: Implement Likelihood Functions"""
        print("\n" + "=" * 80)
        print("STEP 3: LIKELIHOOD FUNCTION IMPLEMENTATION")
        print("=" * 80)
        
        # Implement continuous likelihood
        self._implement_continuous_likelihood()
        
        print("[OK] Likelihood functions implemented")
        return True
    
    def _implement_continuous_likelihood(self):
        """Implement multivariate normal likelihood function"""
        print("\n--- Continuous Likelihood Function ---")
        
        def multivariate_normal_log_likelihood(data, mean, cov):
            """Calculate log-likelihood for multivariate normal distribution"""
            try:
                n, p = data.shape
                # Add regularization to covariance matrix
                cov_reg = cov + np.eye(p) * 1e-6
                
                # Calculate log-likelihood
                log_lik = -0.5 * n * p * np.log(2 * np.pi)
                log_lik -= 0.5 * n * np.log(np.linalg.det(cov_reg))
                
                # Calculate quadratic form
                diff = data - mean
                inv_cov = np.linalg.inv(cov_reg)
                quadratic = np.sum(diff @ inv_cov * diff)
                log_lik -= 0.5 * quadratic
                
                return log_lik
            except np.linalg.LinAlgError:
                return -np.inf
        
        self.continuous_likelihood = multivariate_normal_log_likelihood
        print("[OK] Multivariate normal likelihood function implemented")
    
    def implement_optimization_methods(self):
        """Step 4: Implement Optimization Methods"""
        print("\n" + "=" * 80)
        print("STEP 4: OPTIMIZATION METHODS")
        print("=" * 80)
        
        # Implement simple optimization
        self._implement_simple_optimization()
        
        print("[OK] Optimization methods implemented")
        return True
    
    def _implement_simple_optimization(self):
        """Implement simple optimization using scipy"""
        print("\n--- Simple Optimization Implementation ---")
        
        def optimize_mle(likelihood_func, initial_params, data):
            """Optimize MLE using scipy minimize"""
            def neg_log_likelihood(params):
                # Split parameters into mean and covariance
                n_vars = data.shape[1]
                mean = params[:n_vars]
                
                # Reconstruct covariance matrix from parameters
                cov_params = params[n_vars:]
                cov = np.eye(n_vars)
                idx = 0
                for i in range(n_vars):
                    for j in range(i, n_vars):
                        if i == j:
                            cov[i, j] = np.exp(cov_params[idx])  # Ensure positive
                        else:
                            cov[i, j] = cov_params[idx]
                            cov[j, i] = cov_params[idx]
                        idx += 1
                
                return -likelihood_func(data, mean, cov)
            
            # Initial parameters
            n_vars = data.shape[1]
            initial_mean = np.mean(data, axis=0)
            initial_cov_params = np.random.normal(0, 0.1, n_vars * (n_vars + 1) // 2)
            initial_params = np.concatenate([initial_mean, initial_cov_params])
            
            # Optimize
            result = minimize(neg_log_likelihood, initial_params, method='BFGS')
            
            # Extract results
            mean_est = result.x[:n_vars]
            cov_params = result.x[n_vars:]
            
            # Reconstruct covariance matrix
            cov_est = np.eye(n_vars)
            idx = 0
            for i in range(n_vars):
                for j in range(i, n_vars):
                    if i == j:
                        cov_est[i, j] = np.exp(cov_params[idx])
                    else:
                        cov_est[i, j] = cov_params[idx]
                        cov_est[j, i] = cov_params[idx]
                    idx += 1
            
            return mean_est, cov_est, -result.fun
        
        self.optimize_mle = optimize_mle
        print("[OK] Simple optimization implemented")
    
    def implement_em_algorithm(self):
        """Step 5: Implement EM Algorithm for Missing Data"""
        print("\n" + "=" * 80)
        print("STEP 5: EM ALGORITHM IMPLEMENTATION")
        print("=" * 80)
        
        # Implement simple EM
        self._implement_simple_em()
        
        print("[OK] EM algorithm implemented")
        return True
    
    def _implement_simple_em(self):
        """Implement simple EM algorithm"""
        print("\n--- Simple EM Algorithm ---")
        
        def simple_em(data, max_iter=50, tol=1e-6):
            """Simple EM algorithm for missing data"""
            # Initialize with observed data
            mean_init = np.nanmean(data, axis=0)
            cov_init = np.cov(data.T, rowvar=True, bias=True)
            
            # Handle NaN in covariance
            if np.any(np.isnan(cov_init)):
                cov_init = np.eye(data.shape[1])
            
            cov_init = cov_init + np.eye(data.shape[1]) * 1e-6
            
            current_mean = mean_init.copy()
            current_cov = cov_init.copy()
            
            prev_likelihood = -np.inf
            likelihood_history = []
            
            for iteration in range(max_iter):
                # E-step: Impute missing values
                data_imputed = data.copy()
                for i in range(len(data)):
                    if np.isnan(data[i]).any():
                        observed_mask = ~np.isnan(data[i])
                        missing_mask = np.isnan(data[i])
                        
                        if observed_mask.any():
                            # Conditional mean for missing values
                            mu_obs = current_mean[observed_mask]
                            mu_miss = current_mean[missing_mask]
                            
                            # Conditional covariance
                            cov_obs = current_cov[np.ix_(observed_mask, observed_mask)]
                            cov_miss_obs = current_cov[np.ix_(missing_mask, observed_mask)]
                            
                            try:
                                cov_obs_inv = np.linalg.inv(cov_obs + np.eye(cov_obs.shape[0]) * 1e-6)
                                conditional_mean = mu_miss + cov_miss_obs @ cov_obs_inv @ (data[i][observed_mask] - mu_obs)
                                data_imputed[i][missing_mask] = conditional_mean
                            except:
                                data_imputed[i][missing_mask] = mu_miss
                        else:
                            data_imputed[i][missing_mask] = current_mean[missing_mask]
                
                # M-step: Update parameters
                current_mean = np.mean(data_imputed, axis=0)
                current_cov = np.cov(data_imputed.T, rowvar=True, bias=True) + np.eye(data_imputed.shape[1]) * 1e-6
                
                # Calculate likelihood
                current_likelihood = self.continuous_likelihood(data_imputed, current_mean, current_cov)
                likelihood_history.append(current_likelihood)
                
                # Check convergence
                if abs(current_likelihood - prev_likelihood) < tol:
                    break
                
                prev_likelihood = current_likelihood
            
            return {
                'mean': current_mean,
                'covariance': current_cov,
                'likelihood': current_likelihood,
                'iterations': iteration + 1,
                'likelihood_history': likelihood_history,
                'converged': abs(current_likelihood - prev_likelihood) < tol
            }
        
        self.simple_em = simple_em
        print("[OK] Simple EM algorithm implemented")
    
    def run_complete_mle_analysis(self):
        """Step 6: Run Complete MLE Analysis"""
        print("\n" + "=" * 80)
        print("STEP 6: COMPLETE MLE ANALYSIS (TOP 25 VARIABLES)")
        print("=" * 80)
        
        # Prepare data for MLE
        self._prepare_mle_data()
        
        # Run EM algorithm
        self._run_em_algorithm()
        
        # Validate results
        self._validate_mle_results()
        
        # Calculate RMSE
        self._calculate_rmse()
        
        # Generate final report
        self._generate_final_report()
        
        print("[OK] Complete MLE analysis finished")
        return self.mle_results
    
    def _prepare_mle_data(self):
        """Prepare data for MLE analysis"""
        print("\n--- Preparing Data for MLE ---")
        
        # Select key variables for analysis
        analysis_vars = self.key_continuous
        analysis_vars = [var for var in analysis_vars if var in self.data.columns]
        
        # Prepare continuous data
        self.data_continuous = self.data[analysis_vars].values
        self.analysis_vars = analysis_vars
        
        print(f"[OK] Continuous data shape: {self.data_continuous.shape}")
        print(f"[OK] Variables: {analysis_vars}")
        print(f"[TOP 25 MODEL] Using {len(analysis_vars)} variables (target + top 25 predictors)")
    
    def _run_em_algorithm(self):
        """Run EM algorithm on prepared data"""
        print("\n--- Running EM Algorithm ---")
        
        # Run EM algorithm
        self.mle_results = self.simple_em(
            self.data_continuous,
            max_iter=100,
            tol=1e-4
        )
        
        print(f"[OK] EM algorithm completed")
        print(f"  - Iterations: {self.mle_results['iterations']}")
        print(f"  - Converged: {self.mle_results['converged']}")
        print(f"  - Final likelihood: {self.mle_results['likelihood']:.4f}")
    
    def _validate_mle_results(self):
        """Validate MLE results"""
        print("\n--- Validating MLE Results ---")
        
        # Check parameter estimates
        mean_est = self.mle_results['mean']
        cov_est = self.mle_results['covariance']
        
        print(f"[OK] Mean estimates shape: {mean_est.shape}")
        print(f"[OK] Covariance matrix shape: {cov_est.shape}")
        print(f"[OK] Covariance matrix positive definite: {np.all(np.linalg.eigvals(cov_est) > 0)}")
    
    def _calculate_rmse(self):
        """Calculate RMSE and other performance metrics"""
        print("\n--- Calculating Performance Metrics ---")
        
        # Get MLE parameters
        mean_vector = self.mle_results['mean']
        cov_matrix = self.mle_results['covariance']
        
        # Prepare data for evaluation (only complete cases)
        eval_data = self.data[self.analysis_vars].dropna()
        
        # Extract actual birthweight values
        actual_values = eval_data['f1_bw'].values
        
        # Prepare predictor variables (excluding birthweight)
        predictor_vars = [var for var in self.analysis_vars if var != 'f1_bw']
        X = eval_data[predictor_vars].values
        
        # Calculate predictions using conditional distribution
        # For multivariate normal: E[Y|X] = μ_y + Σ_yx * Σ_xx^(-1) * (X - μ_x)
        
        # Split mean vector (birthweight is first variable)
        mu_y = mean_vector[0]  # Birthweight mean
        mu_x = mean_vector[1:]  # Predictor means
        
        # Split covariance matrix
        sigma_yy = cov_matrix[0, 0]  # Birthweight variance
        sigma_yx = cov_matrix[0, 1:]  # Birthweight-predictor covariances
        sigma_xx = cov_matrix[1:, 1:]  # Predictor covariance matrix
        
        # Calculate conditional mean: E[Y|X] = μ_y + Σ_yx * Σ_xx^(-1) * (X - μ_x)
        # Use more numerically stable formula: beta = Σ_xx^(-1) * Σ_yx, then predictions = μ_y + (X - μ_x) @ beta
        try:
            sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * 1e-6)
            # Calculate beta coefficients: beta = sigma_xx_inv @ sigma_yx
            beta = sigma_xx_inv @ sigma_yx
            # Center predictors
            X_centered = X - mu_x
            # Calculate predictions: predictions = mu_y + X_centered @ beta
            predictions = mu_y + X_centered @ beta
        except np.linalg.LinAlgError:
            print("[WARNING] Using fallback prediction method")
            # Fallback: use simple linear relationship
            beta = np.linalg.lstsq(X, actual_values, rcond=None)[0]
            predictions = X @ beta
        
        # Calculate metrics
        mse = mean_squared_error(actual_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_values, predictions)
        r2 = r2_score(actual_values, predictions)
        
        # Additional metrics
        mape = np.mean(np.abs((actual_values - predictions) / actual_values)) * 100
        correlation, p_value = pearsonr(actual_values, predictions)
        
        # Store metrics
        self.metrics = {
            'RMSE': float(rmse),
            'MSE': float(mse),
            'MAE': float(mae),
            'R²': float(r2),
            'MAPE': float(mape),
            'Correlation': float(correlation),
            'P-value': float(p_value),
            'Sample_Size': int(len(actual_values)),
            'Num_Predictors': len(predictor_vars)
        }
        
        # Store predictions for visualization
        self.predictions = predictions
        self.actual_values = actual_values
        
        # Print results
        print(f"\n[TOP 25 MODEL] PERFORMANCE METRICS:")
        print(f"  - Number of predictors: {len(predictor_vars)}")
        print(f"  - Sample size: {len(actual_values)}")
        print(f"  - RMSE: {rmse:.4f} grams")
        print(f"  - MAE: {mae:.4f} grams")
        print(f"  - R²: {r2:.4f}")
        print(f"  - MAPE: {mape:.2f}%")
        print(f"  - Correlation: {correlation:.4f} (p={p_value:.4e})")
    
    def _generate_final_report(self):
        """Generate final MLE analysis report"""
        print("\n--- Generating Final Report ---")
        
        # Create output directories if they don't exist
        os.makedirs('Data/processed/MLE_Top25', exist_ok=True)
        os.makedirs('PLOTS/MLE_Top25', exist_ok=True)
        
        # Create results summary
        results_summary = {
            'model_type': 'MLE with Top 25 Variables from Feature Importance',
            'num_variables': len(self.analysis_vars),
            'variable_list': self.analysis_vars,
            'top_25_variables': self.top_25_vars,
            'convergence': {
                'converged': bool(self.mle_results['converged']),
                'iterations': int(self.mle_results['iterations']),
                'final_likelihood': float(self.mle_results['likelihood'])
            },
            'continuous_parameters': {
                'mean_vector': [float(x) for x in self.mle_results['mean'].tolist()],
                'covariance_matrix': [[float(x) for x in row] for row in self.mle_results['covariance'].tolist()]
            },
            'performance_metrics': self.metrics
        }
        
        # Save results
        with open('Data/processed/MLE_Top25/mle_top25_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv('Data/processed/MLE_Top25/mle_top25_metrics.csv', index=False)
        
        # Create visualization
        self._create_results_visualization()
        
        print("[OK] Final report generated and saved")
        print(f"     - Results: Data/processed/MLE_Top25/mle_top25_results.json")
        print(f"     - Metrics: Data/processed/MLE_Top25/mle_top25_metrics.csv")
        print(f"     - Plots: PLOTS/MLE_Top25/mle_top25_visualization.png")
    
    def _create_results_visualization(self):
        """Create visualization of MLE results"""
        print("\n--- Creating Results Visualization ---")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Subplot 1: Actual vs Predicted
        ax1 = plt.subplot(2, 3, 1)
        ax1.scatter(self.actual_values, self.predictions, alpha=0.6, s=20)
        ax1.plot([self.actual_values.min(), self.actual_values.max()], 
                [self.actual_values.min(), self.actual_values.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Birthweight (grams)')
        ax1.set_ylabel('Predicted Birthweight (grams)')
        ax1.set_title(f'Actual vs Predicted (Top 25 Variables)\nR² = {self.metrics["R²"]:.4f}')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Residuals
        ax2 = plt.subplot(2, 3, 2)
        residuals = self.actual_values - self.predictions
        ax2.scatter(self.predictions, residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Birthweight (grams)')
        ax2.set_ylabel('Residuals (grams)')
        ax2.set_title(f'Residuals Plot\nRMSE = {self.metrics["RMSE"]:.2f}')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Residuals distribution
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--')
        ax3.set_xlabel('Residuals (grams)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Residuals Distribution\nMAE = {self.metrics["MAE"]:.2f}')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Likelihood convergence
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(self.mle_results['likelihood_history'])
        ax4.set_title('EM Algorithm Convergence')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Log-Likelihood')
        ax4.grid(True)
        
        # Subplot 5: Mean parameter estimates
        ax5 = plt.subplot(2, 3, 5)
        mean_est = self.mle_results['mean']
        var_labels = [var.replace('f0_m_', '').replace('f1_', '').replace('f0_f_', '')[:20] for var in self.analysis_vars]
        ax5.barh(range(len(mean_est)), mean_est)
        ax5.set_yticks(range(len(mean_est)))
        ax5.set_yticklabels(var_labels, fontsize=7)
        ax5.set_title('Mean Parameter Estimates (Top 25)')
        ax5.set_xlabel('Mean Value')
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: Performance metrics
        ax6 = plt.subplot(2, 3, 6)
        metric_names = ['RMSE', 'MAE', 'R²', 'MAPE']
        metric_values = [self.metrics[name] for name in metric_names]
        colors = ['red', 'orange', 'green', 'blue']
        bars = ax6.bar(metric_names, metric_values, color=colors, alpha=0.7)
        ax6.set_ylabel('Metric Value')
        ax6.set_title(f'Performance Metrics\n({self.metrics["Num_Predictors"]} predictors)')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('PLOTS/MLE_Top25/mle_top25_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Results visualization saved")

def main():
    """Main function to run MLE implementation with top 25 variables"""
    print("=" * 80)
    print("MAXIMUM LIKELIHOOD ESTIMATION")
    print("FOR MATERNAL-CHILD HEALTH DATA ANALYSIS")
    print("USING TOP 25 VARIABLES FROM FEATURE IMPORTANCE RANKING")
    print("=" * 80)
    print("\nMODEL: Using Top 25 variables based on feature importance scores")
    print("=" * 80)
    
    # Initialize MLE implementation
    mle = OptimizedMLE('Data/processed/cleaned_dataset_with_engineered_features.csv')
    
    # Step 1: Data Structure Understanding
    mle.load_and_analyze_data()
    
    # Step 2: Probability Model Selection
    mle.select_probability_models()
    
    # Step 3: Likelihood Function Implementation
    mle.implement_likelihood_functions()
    
    # Step 4: Optimization Methods
    mle.implement_optimization_methods()
    
    # Step 5: EM Algorithm Implementation
    mle.implement_em_algorithm()
    
    # Step 6: Complete MLE Analysis (including RMSE calculation)
    results = mle.run_complete_mle_analysis()
    
    print("\n" + "=" * 80)
    print("MLE IMPLEMENTATION WITH TOP 25 VARIABLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n[FINAL RESULTS]")
    print(f"  Number of predictors: {mle.metrics['Num_Predictors']}")
    print(f"  RMSE: {mle.metrics['RMSE']:.4f} grams")
    print(f"  MAE: {mle.metrics['MAE']:.4f} grams")
    print(f"  R²: {mle.metrics['R²']:.4f}")
    print(f"  Correlation: {mle.metrics['Correlation']:.4f}")
    print(f"\nResults saved to:")
    print(f"  - Data/processed/MLE_Top25/mle_top25_results.json")
    print(f"  - Data/processed/MLE_Top25/mle_top25_metrics.csv")
    print(f"  - PLOTS/MLE_Top25/mle_top25_visualization.png")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    results = main()

