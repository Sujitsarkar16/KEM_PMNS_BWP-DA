"""
BART (Bayesian Additive Regression Trees) Model - PMNS Dataset
=============================================================

This script implements BART model for birthweight prediction
using 20 variables from the PMNS Dataset.

BART is a Bayesian ensemble method that combines multiple regression trees
with a Bayesian prior to provide uncertainty quantification.

Author: Sujit sarkar
Date: 2025-12-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import pearsonr
import json
import warnings
import os
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')

# Try to import BART library
try:
    import bartpy.sklearnmodel
    from bartpy.sklearnmodel import SklearnModel
    BART_AVAILABLE = True
    BART_LIBRARY = 'bartpy'
except ImportError:
    BART_AVAILABLE = False
    BART_LIBRARY = None
    print("[WARNING] bartpy library not found. Please install:")
    print("  pip install bartpy")


class BARTModel:
    """
    BART (Bayesian Additive Regression Trees) model implementation
    for birthweight prediction using 20 PMNS variables
    """
    
    def __init__(self, data_path):
        """
        Initialize BART model
        
        Parameters:
        -----------
        data_path : str
            Path to PMNS data CSV file
        """
        self.data_path = data_path
        self.data = None
        self.features = []
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.results = {}
        self.bart_library = BART_LIBRARY
        
        if not BART_AVAILABLE:
            raise ImportError("BART library not available. Please install a BART implementation.")
    
    def load_and_prepare_features(self):
        """Step 1: Load data and prepare PMNS 20 variables"""
        print("=" * 80)
        print("STEP 1: DATA LOADING AND FEATURE PREPARATION (PMNS 20 VARIABLES)")
        print("=" * 80)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Define the 20 PMNS variables
        print("\n[INFO] Using 20 PMNS Dataset variables")
        pmns_20_vars = [
            'f0_m_parity_v1',
            'f0_m_wt_prepreg',
            'f0_m_fundal_ht_v2',
            'f0_m_abd_cir_v2',
            'f0_m_wt_v2',
            'f0_m_r4_v2',
            'f0_m_lunch_cal_v1',
            'f0_m_p_sc_v1',
            'f0_m_o_sc_v1',
            'f0_m_pulse_r1_v2',
            'f0_m_pulse_r2_v2',
            'f0_m_glu_f_v2',
            'f0_m_rcf_v2',
            'f0_m_g_sc_v1',
            'f0_m_plac_wt',
            'f0_m_GA_Del',
            'f0_f_head_cir_ini',
            'f0_f_plt_ini',
            'f1_sex',
            'f0_m_age_eld_child'
        ]
        
        # Check which variables exist in data
        self.features = [var for var in pmns_20_vars if var in self.data.columns]
        missing_vars = [var for var in pmns_20_vars if var not in self.data.columns]
        
        if missing_vars:
            print(f"[WARNING] {len(missing_vars)} variables not found in data")
            for var in missing_vars:
                print(f"  - {var}")
        
        print(f"[OK] Selected {len(self.features)} features from PMNS 20 variables")
        print(f"[INFO] Using BART library: {self.bart_library}")
        
        return self.data, self.features
    
    def prepare_data_splits(self, test_size=0.2, val_size=0.1, random_state=42):
        """Step 2: Prepare train/validation/test splits"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA SPLITTING")
        print("=" * 80)
        
        # Prepare features and target
        X = self.data[self.features].copy()
        y = self.data['f1_bw'].copy()
        
        # Remove rows with missing target
        valid_mask = ~y.isnull()
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        
        # Handle missing values in features using iterative imputation
        print("\n[INFO] Handling missing values using IterativeImputer...")
        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            # Replace infinite values with NaN first
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Use iterative imputation
            imputer = IterativeImputer(random_state=random_state, max_iter=10, 
                                      n_nearest_features=min(10, len(X.columns)))
            X_imputed = imputer.fit_transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
            
            missing_after = X.isnull().sum().sum()
            print(f"  - Missing values before: {missing_before}")
            print(f"  - Missing values after: {missing_after}")
        else:
            print("  - No missing values found")
        
        # Create train/test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        # Create train/val split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=True
        )
        
        # Store splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        print(f"\n[OK] Data splits created:")
        print(f"  - Training set: {X_train.shape[0]} samples")
        print(f"  - Validation set: {X_val.shape[0]} samples")
        print(f"  - Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_bart_model(self, n_trees=200, n_burn=200, n_samples=1000, alpha=0.95, beta=2.0):
        """
        Step 3: Train BART model
        
        Parameters:
        -----------
        n_trees : int
            Number of trees in the ensemble
        n_burn : int
            Number of burn-in MCMC iterations
        n_samples : int
            Number of posterior samples
        alpha : float
            Tree depth parameter
        beta : float
            Tree depth parameter
        """
        print("\n" + "=" * 80)
        print("STEP 3: BART MODEL TRAINING")
        print("=" * 80)
        
        print(f"\n[BART Hyperparameters]:")
        print(f"  - Number of trees: {n_trees}")
        print(f"  - Burn-in iterations: {n_burn}")
        print(f"  - Posterior samples: {n_samples}")
        print(f"  - Alpha (tree depth): {alpha}")
        print(f"  - Beta (tree depth): {beta}")
        print(f"  - Library: {self.bart_library}")
        
        if self.bart_library == 'bartpy':
            # Use bartpy library
            from bartpy.sklearnmodel import SklearnModel
            
            self.model = SklearnModel(
                n_trees=n_trees,
                n_burn=n_burn,
                n_samples=n_samples,
                alpha=alpha,
                beta=beta
            )
            
            print("\n[INFO] Training BART model (this may take a while)...")
            self.model.fit(self.X_train.values, self.y_train.values)
        else:
            raise NotImplementedError(f"BART library {self.bart_library} not available. Please install bartpy.")
        
        print("[OK] BART model training completed")
        
        return self.model
    
    def evaluate_model(self):
        """Step 4: Evaluate model on all splits"""
        print("\n" + "=" * 80)
        print("STEP 4: MODEL EVALUATION")
        print("=" * 80)
        
        # Make predictions on all splits
        splits = {
            'train': (self.X_train, self.y_train),
            'validation': (self.X_val, self.y_val),
            'test': (self.X_test, self.y_test)
        }
        
        self.results = {}
        self.predictions = {}
        
        for split_name, (X_split, y_split) in splits.items():
            if self.bart_library == 'bartpy':
                # bartpy returns mean prediction
                y_pred = self.model.predict(X_split.values)
            else:
                raise NotImplementedError(f"Prediction for {self.bart_library} not implemented")
            
            # Calculate metrics
            mse = mean_squared_error(y_split, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_split, y_pred)
            r2 = r2_score(y_split, y_pred)
            mape = np.mean(np.abs((y_split - y_pred) / y_split)) * 100
            correlation, p_value = pearsonr(y_split, y_pred)
            
            self.results[split_name] = {
                'RMSE': float(rmse),
                'MSE': float(mse),
                'MAE': float(mae),
                'R²': float(r2),
                'MAPE': float(mape),
                'Correlation': float(correlation),
                'P-value': float(p_value),
                'Sample_Size': int(len(y_split))
            }
            
            self.predictions[split_name] = (y_split, y_pred)
            
            print(f"\n[{split_name.upper()} SET] Performance Metrics:")
            print(f"  - Sample size: {len(y_split)}")
            print(f"  - RMSE: {rmse:.4f} grams")
            print(f"  - MAE: {mae:.4f} grams")
            print(f"  - R²: {r2:.4f}")
            print(f"  - MAPE: {mape:.2f}%")
            print(f"  - Correlation: {correlation:.4f} (p={p_value:.4e})")
        
        # Monitor train-validation gap for overfitting
        train_val_gap = self.results['train']['RMSE'] - self.results['validation']['RMSE']
        test_val_gap = self.results['test']['RMSE'] - self.results['validation']['RMSE']
        
        print("\n" + "=" * 80)
        print("OVERFITTING ANALYSIS")
        print("=" * 80)
        print(f"  - Train-Val RMSE Gap: {train_val_gap:.4f} grams")
        print(f"  - Test-Val RMSE Gap: {test_val_gap:.4f} grams")
        
        if train_val_gap > 20:
            print(f"  [WARNING] Large train-val gap ({train_val_gap:.2f} grams) - possible overfitting!")
        elif train_val_gap < -10:
            print(f"  [INFO] Validation better than train - good generalization")
        else:
            print(f"  [OK] Train-val gap is reasonable")
        
        return self.results
    
    def create_visualizations(self):
        """Step 5: Create visualizations"""
        print("\n" + "=" * 80)
        print("STEP 5: CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create output directory
        plot_dir = 'PLOTS/BART_PMNS20'
        os.makedirs(plot_dir, exist_ok=True)
        
        fig = plt.figure(figsize=(18, 12))
        
        # Subplot 1: Actual vs Predicted (Test set)
        ax1 = plt.subplot(2, 3, 1)
        y_test, y_pred_test = self.predictions['test']
        ax1.scatter(y_test, y_pred_test, alpha=0.6, s=20)
        ax1.plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Birthweight (grams)')
        ax1.set_ylabel('Predicted Birthweight (grams)')
        ax1.set_title(f'Actual vs Predicted (Test Set)\nR² = {self.results["test"]["R²"]:.4f}, RMSE = {self.results["test"]["RMSE"]:.2f}')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Residuals
        ax2 = plt.subplot(2, 3, 2)
        residuals = y_test - y_pred_test
        ax2.scatter(y_pred_test, residuals, alpha=0.6, s=20)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Birthweight (grams)')
        ax2.set_ylabel('Residuals (grams)')
        ax2.set_title(f'Residuals Plot\nRMSE = {self.results["test"]["RMSE"]:.2f}')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Performance across splits
        ax3 = plt.subplot(2, 3, 3)
        splits = ['train', 'validation', 'test']
        rmse_values = [self.results[split]['RMSE'] for split in splits]
        colors = ['green', 'orange', 'red']
        bars = ax3.bar(splits, rmse_values, color=colors, alpha=0.7)
        ax3.set_ylabel('RMSE (grams)')
        ax3.set_title('RMSE Across Data Splits')
        ax3.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Subplot 4: Prediction intervals (if available from BART)
        ax4 = plt.subplot(2, 3, 4)
        # BART provides uncertainty quantification
        # For now, show prediction distribution
        ax4.hist(y_pred_test, bins=30, alpha=0.7, label='Predictions', edgecolor='black')
        ax4.hist(y_test, bins=30, alpha=0.5, label='Actual', edgecolor='black')
        ax4.set_xlabel('Birthweight (grams)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Error distribution
        ax5 = plt.subplot(2, 3, 5)
        errors = y_test - y_pred_test
        ax5.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax5.axvline(0, color='r', linestyle='--', linewidth=2)
        ax5.set_xlabel('Prediction Error (grams)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Error Distribution')
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: R² comparison
        ax6 = plt.subplot(2, 3, 6)
        r2_values = [self.results[split]['R²'] for split in splits]
        bars = ax6.bar(splits, r2_values, color=colors, alpha=0.7)
        ax6.set_ylabel('R²')
        ax6.set_title('R² Across Data Splits')
        ax6.set_ylim([0, 1])
        ax6.grid(True, alpha=0.3, axis='y')
        for bar, value in zip(bars, r2_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_file = f'{plot_dir}/bart_pmns20_performance.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Visualizations saved to {plot_file}")
        
        return True
    
    def save_results(self):
        """Step 6: Save results"""
        print("\n" + "=" * 80)
        print("STEP 6: SAVING RESULTS")
        print("=" * 80)
        
        # Create output directory
        output_dir = 'Data/processed/BART_PMNS20'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare comprehensive results
        comprehensive_results = {
            'model_type': 'BART_PMNS20',
            'bart_library': self.bart_library,
            'num_features': len(self.features),
            'features': self.features,
            'performance_metrics': self.results,
            'timestamp': timestamp
        }
        
        # Save JSON results
        results_file = f'{output_dir}/bart_pmns20_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Save metrics to CSV
        metrics_data = []
        for split_name, metrics in self.results.items():
            row = {'split': split_name}
            row.update(metrics)
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f'{output_dir}/bart_pmns20_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save model (if pickleable)
        try:
            model_file = f'Models/bart_pmns20_model_{timestamp}.pkl'
            os.makedirs('Models', exist_ok=True)
            joblib.dump(self.model, model_file)
            print(f"  - Model: {model_file}")
        except Exception as e:
            print(f"  [WARNING] Could not save model: {str(e)}")
        
        print(f"[OK] Results saved:")
        print(f"  - Results JSON: {results_file}")
        print(f"  - Metrics CSV: {metrics_file}")
        
        return results_file, metrics_file
    
    def run_complete_analysis(self, n_trees=200, n_burn=200, n_samples=1000, 
                             alpha=0.95, beta=2.0, random_state=42):
        """Run complete BART analysis pipeline"""
        print("=" * 80)
        print("BART MODEL - PMNS 20 VARIABLES")
        print("=" * 80)
        
        # Step 1: Load and prepare features
        self.load_and_prepare_features()
        
        # Step 2: Prepare data splits
        self.prepare_data_splits(random_state=random_state)
        
        # Step 3: Train BART model
        self.train_bart_model(n_trees=n_trees, n_burn=n_burn, n_samples=n_samples,
                              alpha=alpha, beta=beta)
        
        # Step 4: Evaluate model
        self.evaluate_model()
        
        # Step 5: Create visualizations
        self.create_visualizations()
        
        # Step 6: Save results
        self.save_results()
        
        print("\n" + "=" * 80)
        print("BART ANALYSIS COMPLETED!")
        print("=" * 80)
        print(f"\n[FINAL RESULTS]")
        print(f"  Number of features: {len(self.features)}")
        print(f"  Test RMSE: {self.results['test']['RMSE']:.4f} grams")
        print(f"  Test R²: {self.results['test']['R²']:.4f}")
        print(f"  Test MAE: {self.results['test']['MAE']:.4f} grams")
        print("=" * 80)
        
        return {
            'results': self.results,
            'model': self.model,
            'features': self.features
        }


def main():
    """Main function"""
    print("=" * 80)
    print("BART MODEL - PMNS 20 VARIABLES")
    print("=" * 80)
    
    # Configuration
    data_path = 'Data/PMNS_Data.csv'
    
    print(f"\n[Configuration]:")
    print(f"  - Data path: {data_path}")
    print(f"  - Features: PMNS 20 variables")
    print(f"  - BART Library: {BART_LIBRARY}")
    
    if not BART_AVAILABLE:
        print("\n[ERROR] bartpy library not available!")
        print("Please install:")
        print("  pip install bartpy")
        return None
    
    # Initialize BART model
    bart_model = BARTModel(data_path=data_path)
    
    # Run complete analysis
    results = bart_model.run_complete_analysis(
        n_trees=200,
        n_burn=200,
        n_samples=1000,
        alpha=0.95,
        beta=2.0,
        random_state=42
    )
    
    return results


if __name__ == "__main__":
    results = main()

