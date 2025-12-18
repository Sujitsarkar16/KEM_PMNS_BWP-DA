"""
BART (Bayesian Additive Regression Trees) Hyperparameter Optimization
=====================================================================

This script performs hyperparameter optimization for BART model
using randomized search with cross-validation.

Supports:
- Top 25 variables from feature importance ranking
- Comprehensive hyperparameter search space
- Results saving and comparison

Author: Sujit sarkar
Date: 2025-11-11
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import pearsonr
import json
import warnings
import os
import glob
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


class BARTHyperparameterOptimizer:
    """
    BART hyperparameter optimization using GridSearchCV
    """
    
    def __init__(self, data_path, ranking_path='Data/processed/MLE_New/feature_importance_ranking.csv'):
        """
        Initialize optimizer
        
        Parameters:
        -----------
        data_path : str
            Path to data CSV file
        ranking_path : str
            Path to feature importance ranking CSV
        """
        self.data_path = data_path
        self.ranking_path = ranking_path
        self.data = None
        self.features = []
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.best_model = None
        self.search_results = None
        self.results = {}
        self.bart_library = BART_LIBRARY
        
        if not BART_AVAILABLE:
            raise ImportError("BART library not available. Please install a BART implementation.")
    
    def load_and_prepare_features(self):
        """Step 1: Load data and prepare Top 25 features"""
        print("=" * 80)
        print("STEP 1: DATA LOADING AND FEATURE PREPARATION (TOP 25)")
        print("=" * 80)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Load top 25 variables
        print("\n[INFO] Using Top 25 variables from feature importance ranking")
        ranking_df = pd.read_csv(self.ranking_path)
        top_25_vars = ranking_df['variable'].head(25).tolist()
        
        # Check which variables exist in data
        self.features = [var for var in top_25_vars if var in self.data.columns]
        missing_vars = [var for var in top_25_vars if var not in self.data.columns]
        
        if missing_vars:
            print(f"[WARNING] {len(missing_vars)} variables not found in data")
        
        print(f"[OK] Selected {len(self.features)} features from top 25")
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
            X = X.replace([np.inf, -np.inf], np.nan)
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
        
        # Combine train and val for cross-validation
        self.X_cv = pd.concat([X_train, X_val], axis=0)
        self.y_cv = pd.concat([y_train, y_val], axis=0)
        
        print(f"\n[OK] Data splits created:")
        print(f"  - Training set: {X_train.shape[0]} samples")
        print(f"  - Validation set: {X_val.shape[0]} samples")
        print(f"  - Test set: {X_test.shape[0]} samples")
        print(f"  - CV set (train+val): {self.X_cv.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def define_hyperparameter_search_space(self):
        """Step 3: Define hyperparameter search space"""
        print("\n" + "=" * 80)
        print("STEP 3: HYPERPARAMETER SEARCH SPACE DEFINITION")
        print("=" * 80)
        
        # BART hyperparameter grid
        # Note: BART has fewer hyperparameters than XGBoost
        param_grid = {
            'n_trees': [50, 100, 200],  # Number of trees
            'n_burn': [100, 200, 300],  # Burn-in iterations
            'n_samples': [500, 1000],  # Posterior samples
            'alpha': [0.90, 0.95, 0.99],  # Tree depth parameter
            'beta': [1.5, 2.0, 2.5]  # Tree depth parameter
        }
        
        print("\n[Hyperparameter Search Space]:")
        for key, values in param_grid.items():
            print(f"  {key}: {values}")
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        print(f"\n[INFO] Total possible combinations: {total_combinations:,}")
        print(f"[INFO] Using RandomizedSearchCV for optimization")
        
        return param_grid
    
    def create_bart_wrapper(self):
        """Create a wrapper class for BART to work with RandomizedSearchCV"""
        if self.bart_library == 'bartpy':
            from bartpy.sklearnmodel import SklearnModel
            
            class BARTWrapper:
                def __init__(self, n_trees=200, n_burn=200, n_samples=1000, alpha=0.95, beta=2.0):
                    self.n_trees = n_trees
                    self.n_burn = n_burn
                    self.n_samples = n_samples
                    self.alpha = alpha
                    self.beta = beta
                    self.model = None
                
                def get_params(self, deep=True):
                    """Get parameters for this estimator (required by scikit-learn)"""
                    return {
                        'n_trees': self.n_trees,
                        'n_burn': self.n_burn,
                        'n_samples': self.n_samples,
                        'alpha': self.alpha,
                        'beta': self.beta
                    }
                
                def set_params(self, **params):
                    """Set parameters for this estimator (required by scikit-learn)"""
                    for key, value in params.items():
                        setattr(self, key, value)
                    return self
                
                def fit(self, X, y):
                    self.model = SklearnModel(
                        n_trees=self.n_trees,
                        n_burn=self.n_burn,
                        n_samples=self.n_samples,
                        alpha=self.alpha,
                        beta=self.beta
                    )
                    self.model.fit(X, y)
                    return self
                
                def predict(self, X):
                    return self.model.predict(X)
            
            return BARTWrapper
        else:
            raise NotImplementedError(f"RandomizedSearchCV wrapper for {self.bart_library} not implemented")
    
    def perform_grid_search(self, param_grid, cv_folds=5, n_iter=20, random_state=42):
        """Step 4: Perform RandomizedSearchCV"""
        print("\n" + "=" * 80)
        print("STEP 4: RANDOMIZED SEARCH CV")
        print("=" * 80)
        
        # Create BART wrapper
        BARTWrapper = self.create_bart_wrapper()
        base_model = BARTWrapper()
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        
        print(f"\n[INFO] Starting RandomizedSearchCV...")
        print(f"  - Cross-validation folds: {cv_folds}")
        print(f"  - Total parameter combinations: {total_combinations:,}")
        print(f"  - Number of iterations: {n_iter}")
        print(f"  - Estimated model fits: {n_iter * cv_folds} (vs {total_combinations * cv_folds:,} for grid search)")
        
        # Create RMSE scorer
        rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
                                 greater_is_better=False)
        
        # Perform randomized search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring=rmse_scorer,
            n_jobs=1,  # BART may not be thread-safe
            verbose=1,
            random_state=random_state,
            return_train_score=True
        )
        
        print("\n[INFO] Training models (this may take a while)...")
        random_search.fit(self.X_cv.values, self.y_cv.values)
        
        self.search_results = random_search
        self.best_model = random_search.best_estimator_
        
        # Extract results
        best_params = random_search.best_params_
        best_cv_rmse = -random_search.best_score_  # Negative because scorer is negated
        
        print("\n" + "=" * 80)
        print("RANDOMIZED SEARCH RESULTS")
        print("=" * 80)
        print(f"\n[Best Hyperparameters]:")
        for key, value in sorted(best_params.items()):
            print(f"  {key}: {value}")
        print(f"\n[Best CV RMSE]: {best_cv_rmse:.4f} grams")
        
        # Store best parameters
        self.best_hyperparameters = best_params
        self.best_cv_rmse = best_cv_rmse
        
        return random_search
    
    def evaluate_best_model(self):
        """Step 5: Evaluate best model on test set"""
        print("\n" + "=" * 80)
        print("STEP 5: BEST MODEL EVALUATION")
        print("=" * 80)
        
        # Make predictions on all splits
        y_pred_train = self.best_model.predict(self.X_train.values)
        y_pred_val = self.best_model.predict(self.X_val.values)
        y_pred_test = self.best_model.predict(self.X_test.values)
        
        # Calculate metrics for each split
        splits = {
            'train': (self.y_train, y_pred_train),
            'validation': (self.y_val, y_pred_val),
            'test': (self.y_test, y_pred_test)
        }
        
        self.results = {}
        
        for split_name, (y_true, y_pred) in splits.items():
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            correlation, p_value = pearsonr(y_true, y_pred)
            
            self.results[split_name] = {
                'RMSE': float(rmse),
                'MSE': float(mse),
                'MAE': float(mae),
                'R²': float(r2),
                'MAPE': float(mape),
                'Correlation': float(correlation),
                'P-value': float(p_value),
                'Sample_Size': int(len(y_true))
            }
            
            print(f"\n[{split_name.upper()} SET] Performance Metrics:")
            print(f"  - Sample size: {len(y_true)}")
            print(f"  - RMSE: {rmse:.4f} grams")
            print(f"  - MAE: {mae:.4f} grams")
            print(f"  - R²: {r2:.4f}")
            print(f"  - MAPE: {mape:.2f}%")
            print(f"  - Correlation: {correlation:.4f} (p={p_value:.4e})")
        
        # Store predictions
        self.predictions = {
            'train': (self.y_train, y_pred_train),
            'validation': (self.y_val, y_pred_val),
            'test': (self.y_test, y_pred_test)
        }
        
        # Monitor train-validation gap
        train_val_gap = self.results['train']['RMSE'] - self.results['validation']['RMSE']
        test_val_gap = self.results['test']['RMSE'] - self.results['validation']['RMSE']
        
        print("\n" + "=" * 80)
        print("OVERFITTING ANALYSIS")
        print("=" * 80)
        print(f"  - Train-Val RMSE Gap: {train_val_gap:.4f} grams")
        print(f"  - Test-Val RMSE Gap: {test_val_gap:.4f} grams")
        
        if train_val_gap > 20:
            print(f"  [WARNING] Large train-val gap ({train_val_gap:.2f} grams) - possible overfitting!")
        else:
            print(f"  [OK] Train-val gap is reasonable")
        
        return self.results
    
    def create_visualizations(self):
        """Step 6: Create visualizations"""
        print("\n" + "=" * 80)
        print("STEP 6: CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create output directory
        plot_dir = 'PLOTS/BART_Top25_Optimized'
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
        
        # Subplot 4: CV Score distribution
        ax4 = plt.subplot(2, 3, 4)
        cv_results = self.search_results.cv_results_
        test_scores = -cv_results['mean_test_score']  # Negative because scorer is negated
        ax4.hist(test_scores, bins=20, alpha=0.7, edgecolor='black')
        ax4.axvline(self.best_cv_rmse, color='r', linestyle='--', linewidth=2, 
                   label=f'Best: {self.best_cv_rmse:.2f}')
        ax4.set_xlabel('CV RMSE (grams)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('CV Score Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Subplot 5: Hyperparameter importance
        ax5 = plt.subplot(2, 3, 5)
        param_importance = {}
        for param in self.best_hyperparameters.keys():
            param_mask = [param in key for key in cv_results['params']]
            if sum(param_mask) > 1:
                param_scores = -cv_results['mean_test_score'][param_mask]
                param_importance[param] = np.std(param_scores)
        
        if param_importance:
            sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)[:8]
            param_names = [p[0] for p in sorted_params]
            param_values = [p[1] for p in sorted_params]
            ax5.barh(param_names, param_values)
            ax5.set_xlabel('Std Dev of CV Scores')
            ax5.set_title('Hyperparameter Impact')
            ax5.grid(True, alpha=0.3, axis='x')
        
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
        plot_file = f'{plot_dir}/bart_optimized_performance.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Visualizations saved to {plot_file}")
        
        return True
    
    def save_results(self):
        """Step 7: Save results"""
        print("\n" + "=" * 80)
        print("STEP 7: SAVING RESULTS")
        print("=" * 80)
        
        # Create output directory
        output_dir = 'Data/processed/BART_Top25_Optimized'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare comprehensive results
        comprehensive_results = {
            'model_type': 'BART_Optimized_Top25',
            'bart_library': self.bart_library,
            'num_features': len(self.features),
            'features': self.features,
            'best_hyperparameters': self.best_hyperparameters,
            'best_cv_rmse': float(self.best_cv_rmse),
            'performance_metrics': self.results,
            'cv_results_summary': {
                'mean_test_score': float(-self.search_results.best_score_),
                'std_test_score': float(self.search_results.cv_results_['std_test_score'][self.search_results.best_index_]),
                'n_splits': self.search_results.cv.n_splits
            },
            'timestamp': timestamp
        }
        
        # Save JSON results
        results_file = f'{output_dir}/bart_optimized_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Save metrics to CSV
        metrics_data = []
        for split_name, metrics in self.results.items():
            row = {'split': split_name}
            row.update(metrics)
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f'{output_dir}/bart_optimized_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save all CV results
        cv_results_df = pd.DataFrame(self.search_results.cv_results_)
        cv_results_file = f'{output_dir}/bart_cv_search_results_{timestamp}.csv'
        cv_results_df.to_csv(cv_results_file, index=False)
        
        print(f"[OK] Results saved:")
        print(f"  - Results JSON: {results_file}")
        print(f"  - Metrics CSV: {metrics_file}")
        print(f"  - CV Search Results: {cv_results_file}")
        
        return results_file, metrics_file
    
    def run_optimization(self, cv_folds=5, n_iter=20, random_state=42):
        """Run complete hyperparameter optimization pipeline"""
        print("=" * 80)
        print("BART HYPERPARAMETER OPTIMIZATION - TOP 25 VARIABLES")
        print("=" * 80)
        
        # Step 1: Load and prepare features
        self.load_and_prepare_features()
        
        # Step 2: Prepare data splits
        self.prepare_data_splits(random_state=random_state)
        
        # Step 3: Define search space
        param_grid = self.define_hyperparameter_search_space()
        
        # Step 4: Perform randomized search
        self.perform_grid_search(param_grid, cv_folds=cv_folds, n_iter=n_iter, random_state=random_state)
        
        # Step 5: Evaluate best model
        self.evaluate_best_model()
        
        # Step 6: Create visualizations
        self.create_visualizations()
        
        # Step 7: Save results
        self.save_results()
        
        print("\n" + "=" * 80)
        print("HYPERPARAMETER OPTIMIZATION COMPLETED!")
        print("=" * 80)
        print(f"\n[FINAL RESULTS]")
        print(f"  Number of features: {len(self.features)}")
        print(f"  Best CV RMSE: {self.best_cv_rmse:.4f} grams")
        print(f"  Test RMSE: {self.results['test']['RMSE']:.4f} grams")
        print(f"  Test R²: {self.results['test']['R²']:.4f}")
        print(f"  Test MAE: {self.results['test']['MAE']:.4f} grams")
        print("\n[Best Hyperparameters]:")
        for key, value in sorted(self.best_hyperparameters.items()):
            print(f"  {key}: {value}")
        print("=" * 80)
        
        return {
            'best_hyperparameters': self.best_hyperparameters,
            'best_cv_rmse': self.best_cv_rmse,
            'test_metrics': self.results['test'],
            'model': self.best_model
        }


def main():
    """Main function"""
    print("=" * 80)
    print("BART HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Configuration
    data_path = 'Data/processed/cleaned_dataset_with_engineered_features.csv'
    cv_folds = 5  # Cross-validation folds
    n_iter = 20  # Number of random search iterations (reduces from 162 to 20 combinations)
    
    print(f"\n[Configuration]:")
    print(f"  - Use Top 25: True")
    print(f"  - CV folds: {cv_folds}")
    print(f"  - Random search iterations: {n_iter}")
    print(f"  - BART Library: {BART_LIBRARY}")
    
    if not BART_AVAILABLE:
        print("\n[ERROR] BART library not available!")
        return None
    
    # Initialize optimizer
    optimizer = BARTHyperparameterOptimizer(data_path=data_path)
    
    # Run optimization
    results = optimizer.run_optimization(
        cv_folds=cv_folds,
        n_iter=n_iter,
        random_state=42
    )
    
    return results


if __name__ == "__main__":
    results = main()

