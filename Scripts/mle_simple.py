"""
Simple MLE Implementation for Maternal-Child Health Data
======================================================

This script implements a simplified MLE system for analyzing maternal-child health data,
focusing on birthweight prediction while handling missing data patterns.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

class SimpleMLE:
    """
    Simplified MLE implementation for maternal-child health data analysis
    """
    
    def __init__(self, data_path):
        """Initialize MLE implementation"""
        self.data_path = data_path
        self.data = None
        self.variable_groups = None
        self.continuous_vars = []
        self.categorical_vars = []
        self.mle_results = {}
        
    def load_and_analyze_data(self):
        """Step 1: Data Structure Understanding"""
        print("=" * 80)
        print("STEP 1: DATA STRUCTURE UNDERSTANDING")
        print("=" * 80)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Load variable grouping
        self.variable_groups = pd.read_csv('Data/processed/variable_grouping_table.csv')
        print(f"[OK] Loaded variable grouping: {len(self.variable_groups)} variables categorized")
        
        # Classify variables
        self._classify_variables()
        
        # Analyze missing data patterns
        self._analyze_missing_patterns()
        
        print("[OK] Data structure analysis completed")
        return self.data, self.variable_groups
    
    def _classify_variables(self):
        """Classify variables into continuous and categorical"""
        print("\n--- Variable Classification ---")
        
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
        
        # Key variables for analysis
        self.key_continuous = ['f1_bw', 'f0_m_age', 'f0_m_bmi_prepreg', 'f0_m_ht', 'f0_m_wt_prepreg']
        self.key_categorical = ['f0_m_edu', 'f0_f_edu', 'f0_occ_hou_head', 'f1_sex']
        
        print(f"[OK] Key continuous variables: {self.key_continuous}")
        print(f"[OK] Key categorical variables: {self.key_categorical}")
    
    def _analyze_missing_patterns(self):
        """Analyze missing data patterns"""
        print("\n--- Missing Data Analysis ---")
        
        # Calculate missing percentages
        missing_pct = (self.data.isnull().sum() / len(self.data)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        
        print(f"[OK] High missing (>20%): {len(missing_pct[missing_pct > 20])} variables")
        print(f"[OK] Moderate missing (5-20%): {len(missing_pct[(missing_pct > 5) & (missing_pct <= 20)])} variables")
        print(f"[OK] Low missing (0-5%): {len(missing_pct[(missing_pct > 0) & (missing_pct <= 5)])} variables")
        
        # Create missingness heatmap for key variables
        key_vars = self.key_continuous + self.key_categorical
        key_vars = [var for var in key_vars if var in self.data.columns]
        
        if key_vars:
            plt.figure(figsize=(12, 8))
            missing_data = self.data[key_vars].isnull()
            sns.heatmap(missing_data.T, cbar=True, yticklabels=True, cmap='viridis')
            plt.title('Missing Data Pattern for Key Variables')
            plt.tight_layout()
            plt.savefig('PLOTS/missingness_heatmap_key_vars.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("[OK] Missing data heatmap saved")
    
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
        
        if non_normal_vars:
            print(f"  Non-normal variables: {non_normal_vars}")
    
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
        print(f"  - Continuous: {self.model_specs['continuous']['distribution']}")
        print(f"  - Categorical: {self.model_specs['categorical']['distribution']}")
        print(f"  - Mixed model: {self.model_specs['mixed_model']['framework']}")
    
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
            cov_init = np.cov(data.T) + np.eye(data.shape[1]) * 1e-6
            
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
                current_cov = np.cov(data_imputed.T) + np.eye(data_imputed.shape[1]) * 1e-6
                
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
        print("STEP 6: COMPLETE MLE ANALYSIS")
        print("=" * 80)
        
        # Prepare data for MLE
        self._prepare_mle_data()
        
        # Run EM algorithm
        self._run_em_algorithm()
        
        # Validate results
        self._validate_mle_results()
        
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
        
        print(f"[OK] Continuous data shape: {self.data_continuous.shape}")
        print(f"[OK] Variables: {analysis_vars}")
    
    def _run_em_algorithm(self):
        """Run EM algorithm on prepared data"""
        print("\n--- Running EM Algorithm ---")
        
        # Run EM algorithm
        self.mle_results = self.simple_em(
            self.data_continuous,
            max_iter=50,
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
    
    def _generate_final_report(self):
        """Generate final MLE analysis report"""
        print("\n--- Generating Final Report ---")
        
        # Create results summary
        results_summary = {
            'convergence': {
                'converged': bool(self.mle_results['converged']),
                'iterations': int(self.mle_results['iterations']),
                'final_likelihood': float(self.mle_results['likelihood'])
            },
            'continuous_parameters': {
                'mean_vector': [float(x) for x in self.mle_results['mean'].tolist()],
                'covariance_matrix': [[float(x) for x in row] for row in self.mle_results['covariance'].tolist()]
            }
        }
        
        # Save results
        import json
        with open('Data/processed/mle_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Create visualization
        self._create_results_visualization()
        
        print("[OK] Final report generated and saved")
    
    def _create_results_visualization(self):
        """Create visualization of MLE results"""
        print("\n--- Creating Results Visualization ---")
        
        # Plot likelihood convergence
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Likelihood convergence
        plt.subplot(2, 2, 1)
        plt.plot(self.mle_results['likelihood_history'])
        plt.title('EM Algorithm Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.grid(True)
        
        # Subplot 2: Parameter estimates
        plt.subplot(2, 2, 2)
        mean_est = self.mle_results['mean']
        plt.bar(range(len(mean_est)), mean_est)
        plt.title('Mean Parameter Estimates')
        plt.xlabel('Variable Index')
        plt.ylabel('Mean Value')
        plt.grid(True)
        
        # Subplot 3: Covariance matrix heatmap
        plt.subplot(2, 2, 3)
        cov_est = self.mle_results['covariance']
        sns.heatmap(cov_est, annot=True, fmt='.3f', cmap='coolwarm')
        plt.title('Covariance Matrix')
        
        # Subplot 4: Data distribution
        plt.subplot(2, 2, 4)
        if len(self.key_continuous) > 0:
            var_name = self.key_continuous[0]
            if var_name in self.data.columns:
                plt.hist(self.data[var_name].dropna(), bins=30, alpha=0.7)
                plt.title(f'Distribution: {var_name}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('PLOTS/mle_results_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Results visualization saved")

def main():
    """Main function to run complete MLE implementation"""
    print("=" * 80)
    print("MAXIMUM LIKELIHOOD ESTIMATION IMPLEMENTATION")
    print("FOR MATERNAL-CHILD HEALTH DATA ANALYSIS")
    print("=" * 80)
    
    # Initialize MLE implementation
    mle = SimpleMLE('Data/processed/cleaned_dataset_with_engineered_features.csv')
    
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
    
    # Step 6: Complete MLE Analysis
    results = mle.run_complete_mle_analysis()
    
    print("\n" + "=" * 80)
    print("MLE IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Results saved to: Data/processed/mle_results.json")
    print(f"Visualizations saved to: PLOTS/mle_results_visualization.png")
    
    return results

if __name__ == "__main__":
    results = main()
