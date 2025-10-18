"""
Calculate RMSE and Performance Metrics for MLE Model
===================================================

This script calculates RMSE, MAE, R² and other performance metrics
for the MLE model on the maternal-child health dataset.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import json

class MLEPerformanceEvaluator:
    """
    Calculate performance metrics for MLE model
    """
    
    def __init__(self, data_path, results_path):
        """Initialize evaluator"""
        self.data_path = data_path
        self.results_path = results_path
        self.data = None
        self.mle_results = None
        self.predictions = None
        self.actual_values = None
        
    def load_data_and_results(self):
        """Load data and MLE results"""
        print("Loading data and MLE results...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape}")
        
        # Load MLE results
        with open(self.results_path, 'r') as f:
            self.mle_results = json.load(f)
        print(f"[OK] Loaded MLE results")
        
        return True
    
    def prepare_data_for_evaluation(self):
        """Prepare data for performance evaluation"""
        print("\n--- Preparing Data for Evaluation ---")
        
        # Key variables for analysis
        key_vars = ['f1_bw', 'f0_m_age', 'f0_m_bmi_prepreg', 'f0_m_ht', 'f0_m_wt_prepreg']
        key_vars = [var for var in key_vars if var in self.data.columns]
        
        # Prepare data
        self.eval_data = self.data[key_vars].copy()
        
        # Remove rows with any missing values for clean evaluation
        self.eval_data_clean = self.eval_data.dropna()
        
        print(f"[OK] Clean data shape: {self.eval_data_clean.shape}")
        print(f"[OK] Variables: {key_vars}")
        
        # Extract actual values (birthweight)
        self.actual_values = self.eval_data_clean['f1_bw'].values
        print(f"[OK] Actual birthweight values: {len(self.actual_values)} samples")
        
        return True
    
    def generate_predictions(self):
        """Generate predictions using MLE model"""
        print("\n--- Generating Predictions ---")
        
        # Get MLE parameters
        mean_vector = np.array(self.mle_results['continuous_parameters']['mean_vector'])
        cov_matrix = np.array(self.mle_results['continuous_parameters']['covariance_matrix'])
        
        # Prepare predictor variables (excluding birthweight)
        predictor_vars = ['f0_m_age', 'f0_m_bmi_prepreg', 'f0_m_ht', 'f0_m_wt_prepreg']
        predictor_vars = [var for var in predictor_vars if var in self.eval_data_clean.columns]
        
        X = self.eval_data_clean[predictor_vars].values
        
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
        try:
            sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * 1e-6)
            predictions = mu_y + sigma_yx @ sigma_xx_inv @ (X - mu_x).T
            self.predictions = predictions
        except np.linalg.LinAlgError:
            # Fallback: use simple linear relationship
            print("[WARNING] Using fallback prediction method")
            # Simple linear regression coefficients
            beta = np.linalg.lstsq(X, self.actual_values, rcond=None)[0]
            self.predictions = X @ beta
        
        print(f"[OK] Generated predictions: {len(self.predictions)} samples")
        print(f"[OK] Prediction range: {self.predictions.min():.2f} - {self.predictions.max():.2f}")
        
        return True
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        print("\n--- Calculating Performance Metrics ---")
        
        # Basic metrics
        mse = mean_squared_error(self.actual_values, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.actual_values, self.predictions)
        r2 = r2_score(self.actual_values, self.predictions)
        
        # Additional metrics
        mape = np.mean(np.abs((self.actual_values - self.predictions) / self.actual_values)) * 100
        correlation, p_value = pearsonr(self.actual_values, self.predictions)
        
        # Store metrics
        self.metrics = {
            'RMSE': rmse,
            'MSE': mse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape,
            'Correlation': correlation,
            'P-value': p_value,
            'Sample_Size': len(self.actual_values)
        }
        
        # Print results
        print(f"[OK] RMSE: {rmse:.4f} grams")
        print(f"[OK] MAE: {mae:.4f} grams")
        print(f"[OK] R²: {r2:.4f}")
        print(f"[OK] MAPE: {mape:.2f}%")
        print(f"[OK] Correlation: {correlation:.4f} (p={p_value:.4f})")
        
        return self.metrics
    
    def create_performance_visualization(self):
        """Create visualization of model performance"""
        print("\n--- Creating Performance Visualization ---")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(self.actual_values, self.predictions, alpha=0.6, s=20)
        axes[0, 0].plot([self.actual_values.min(), self.actual_values.max()], 
                       [self.actual_values.min(), self.actual_values.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Birthweight (grams)')
        axes[0, 0].set_ylabel('Predicted Birthweight (grams)')
        axes[0, 0].set_title(f'Actual vs Predicted\nR² = {self.metrics["R²"]:.4f}')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = self.actual_values - self.predictions
        axes[0, 1].scatter(self.predictions, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Birthweight (grams)')
        axes[0, 1].set_ylabel('Residuals (grams)')
        axes[0, 1].set_title(f'Residuals Plot\nRMSE = {self.metrics["RMSE"]:.2f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals (grams)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Residuals Distribution\nMAE = {self.metrics["MAE"]:.2f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance metrics bar chart
        metric_names = ['RMSE', 'MAE', 'R²', 'MAPE']
        metric_values = [self.metrics[name] for name in metric_names]
        colors = ['red', 'orange', 'green', 'blue']
        
        bars = axes[1, 1].bar(metric_names, metric_values, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].set_title('Performance Metrics Summary')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('PLOTS/mle_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Performance visualization saved")
    
    def save_results(self):
        """Save performance results"""
        print("\n--- Saving Results ---")
        
        # Save metrics to JSON
        results = {
            'performance_metrics': self.metrics,
            'model_info': {
                'method': 'MLE with EM Algorithm',
                'convergence': self.mle_results['convergence'],
                'sample_size': len(self.actual_values)
            },
            'predictions': {
                'actual_values': self.actual_values.tolist(),
                'predicted_values': self.predictions.tolist(),
                'residuals': (self.actual_values - self.predictions).tolist()
            }
        }
        
        with open('Data/processed/mle_performance_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv('Data/processed/mle_performance_metrics.csv', index=False)
        
        print("[OK] Results saved to JSON and CSV")
    
    def run_evaluation(self):
        """Run complete performance evaluation"""
        print("=" * 80)
        print("MLE MODEL PERFORMANCE EVALUATION")
        print("=" * 80)
        
        # Load data and results
        self.load_data_and_results()
        
        # Prepare data
        self.prepare_data_for_evaluation()
        
        # Generate predictions
        self.generate_predictions()
        
        # Calculate metrics
        self.calculate_performance_metrics()
        
        # Create visualization
        self.create_performance_visualization()
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 80)
        print("PERFORMANCE EVALUATION COMPLETED!")
        print("=" * 80)
        print(f"RMSE: {self.metrics['RMSE']:.4f} grams")
        print(f"MAE: {self.metrics['MAE']:.4f} grams")
        print(f"R²: {self.metrics['R²']:.4f}")
        print(f"Correlation: {self.metrics['Correlation']:.4f}")
        print("=" * 80)
        
        return self.metrics

def main():
    """Main function"""
    evaluator = MLEPerformanceEvaluator(
        'Data/processed/cleaned_dataset_with_engineered_features.csv',
        'Data/processed/mle_results.json'
    )
    
    metrics = evaluator.run_evaluation()
    return metrics

if __name__ == "__main__":
    metrics = main()
