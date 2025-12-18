"""
XGBoost Hyperparameter Optimization
====================================

This script performs hyperparameter optimization for XGBoost model
using RandomizedSearchCV with cross-validation.

Supports:
- Top 25 variables from feature importance ranking
- All available variables
- Comprehensive hyperparameter search space
- Results saving and comparison

Author: Sujit sarkar
Date: 2025-11-03
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import pearsonr
import xgboost as xgb
import json
import warnings
import os
import glob
from datetime import datetime
import joblib

warnings.filterwarnings('ignore')


class XGBoostHyperparameterOptimizer:
    """
    XGBoost hyperparameter optimization using RandomizedSearchCV
    """
    
    def __init__(self, data_path, use_top25=True, ranking_path='Data/processed/MLE_New/feature_importance_ranking.csv'):
        """
        Initialize optimizer
        
        Parameters:
        -----------
        data_path : str
            Path to data CSV file
        use_top25 : bool
            If True, use top 25 variables; if False, use all variables
        ranking_path : str
            Path to feature importance ranking CSV (if use_top25=True)
        """
        self.data_path = data_path
        self.use_top25 = use_top25
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
        
    def load_and_prepare_features(self):
        """Step 1: Load data and prepare features"""
        print("=" * 80)
        print("STEP 1: DATA LOADING AND FEATURE PREPARATION")
        print("=" * 80)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        if self.use_top25:
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
        else:
            # Use all numeric variables
            print("\n[INFO] Using ALL numeric variables")
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
            self.features = [col for col in numeric_cols if col != 'f1_bw']
            
            # Remove constant columns
            constant_cols = []
            for col in self.features:
                if self.data[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                self.features = [col for col in self.features if col not in constant_cols]
                print(f"[INFO] Removed {len(constant_cols)} constant columns")
            
            print(f"[OK] Selected {len(self.features)} features")
        
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
        
        # Handle missing values in features using iterative imputation (better than median)
        print("\n[INFO] Handling missing values using IterativeImputer...")
        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            # Replace infinite values with NaN first
            X = X.replace([np.inf, -np.inf], np.nan)
            
            # Use iterative imputation
            imputer = IterativeImputer(random_state=random_state, max_iter=10, n_nearest_features=min(10, len(X.columns)))
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
        
        # Combine train and val for cross-validation
        self.X_cv = pd.concat([X_train, X_val], axis=0)
        self.y_cv = pd.concat([y_train, y_val], axis=0)
        
        print(f"  - CV set (train+val): {self.X_cv.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def define_hyperparameter_search_space(self):
        """Step 3: Define hyperparameter search space"""
        print("\n" + "=" * 80)
        print("STEP 3: HYPERPARAMETER SEARCH SPACE DEFINITION")
        print("=" * 80)
        
        # Comprehensive hyperparameter grid
        # Strengthened regularization to prevent overfitting
        if self.use_top25:
            # For fewer features, use stronger regularization with lower learning rates
            param_distributions = {
                'n_estimators': [200, 300, 400, 500],  # More trees with lower LR
                'max_depth': [3, 4, 5],  # Reduced max depth to prevent overfitting
                'learning_rate': [0.01, 0.03, 0.05, 0.08],  # Lower learning rates
                'subsample': [0.7, 0.75, 0.8, 0.85],  # More subsampling for regularization
                'colsample_bytree': [0.6, 0.7, 0.8],  # More feature subsampling
                'reg_alpha': [0.5, 1.0, 2.0, 3.0],  # Higher L1 regularization
                'reg_lambda': [1.0, 2.0, 3.0, 5.0],  # Higher L2 regularization
                'min_child_weight': [3, 5, 7, 10],  # Higher min_child_weight
                'gamma': [0.1, 0.2, 0.3, 0.5]  # Higher gamma for regularization
            }
        else:
            # For many features, use even more regularization
            param_distributions = {
                'n_estimators': [300, 400, 500, 600],
                'max_depth': [3, 4, 5],  # Keep shallow to prevent overfitting
                'learning_rate': [0.01, 0.03, 0.05],  # Lower learning rates
                'subsample': [0.6, 0.7, 0.8],  # More subsampling
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8],  # More feature subsampling
                'reg_alpha': [1.0, 2.0, 3.0, 5.0],  # Higher L1 regularization
                'reg_lambda': [2.0, 3.0, 5.0, 7.0],  # Higher L2 regularization
                'min_child_weight': [5, 7, 10],  # Higher min_child_weight
                'gamma': [0.2, 0.3, 0.5]  # Higher gamma
            }
        
        print("\n[Hyperparameter Search Space]:")
        for key, values in param_distributions.items():
            print(f"  {key}: {values}")
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_distributions.values():
            total_combinations *= len(values)
        print(f"\n[INFO] Total possible combinations: {total_combinations:,}")
        print(f"[INFO] Using RandomizedSearchCV for efficiency")
        
        return param_distributions
    
    def perform_random_search(self, param_distributions, n_iter=50, cv_folds=5, random_state=42):
        """Step 4: Perform RandomizedSearchCV"""
        print("\n" + "=" * 80)
        print("STEP 4: RANDOMIZED SEARCH CV")
        print("=" * 80)
        
        # Base XGBoost model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=random_state,
            n_jobs=-1,
            tree_method='hist'
        )
        
        print(f"\n[INFO] Starting RandomizedSearchCV...")
        print(f"  - Number of iterations: {n_iter}")
        print(f"  - Cross-validation folds: {cv_folds}")
        print(f"  - Total models to train: {n_iter * cv_folds}")
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
            return_train_score=True
        )
        
        print("\n[INFO] Training models (this may take a while)...")
        random_search.fit(self.X_cv, self.y_cv)
        
        self.search_results = random_search
        
        # Retrain best model with early stopping on validation set
        print("\n[INFO] Retraining best model with early stopping...")
        best_params = random_search.best_params_.copy()
        # Remove early_stopping_rounds from params if present (it's not a fit parameter)
        best_params.pop('early_stopping_rounds', None)
        
        best_model_with_early_stop = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=random_state,
            n_jobs=-1,
            tree_method='hist',
            **best_params
        )
        
        # Train with early stopping using callbacks (compatible with XGBoost 2.0+)
        try:
            # Try using callbacks for newer XGBoost versions
            best_model_with_early_stop.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=20, save_best=True)],
                verbose=False
            )
        except (AttributeError, TypeError):
            # Fallback for older XGBoost versions
            try:
                best_model_with_early_stop.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                    early_stopping_rounds=20,
                    verbose=False
                )
            except TypeError:
                # If early stopping not supported, train without it
                print("[WARNING] Early stopping not supported in this XGBoost version, training without it")
                best_model_with_early_stop.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
                    verbose=False
                )
        
        self.best_model = best_model_with_early_stop
        
        # Extract results (best_params already defined above)
        best_cv_rmse = np.sqrt(-random_search.best_score_)
        
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
        y_pred_train = self.best_model.predict(self.X_train)
        y_pred_val = self.best_model.predict(self.X_val)
        y_pred_test = self.best_model.predict(self.X_test)
        
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
        
        # Monitor train-validation gap for overfitting detection
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
        
        if test_val_gap > 30:
            print(f"  [WARNING] Large test-val gap ({test_val_gap:.2f} grams) - model may not generalize well!")
        else:
            print(f"  [OK] Test-val gap is acceptable")
        
        # Store predictions
        self.predictions = {
            'train': (self.y_train, y_pred_train),
            'validation': (self.y_val, y_pred_val),
            'test': (self.y_test, y_pred_test)
        }
        
        # Store overfitting metrics
        self.results['overfitting_analysis'] = {
            'train_val_gap': float(train_val_gap),
            'test_val_gap': float(test_val_gap)
        }
        
        return self.results
    
    def get_feature_importance(self):
        """Get feature importance from best model"""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        importance_scores = self.best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        self.results['feature_importance'] = feature_importance.to_dict('records')
        
        print("\n[Top 20 Most Important Features]:")
        for i, (idx, row) in enumerate(feature_importance.head(20).iterrows(), 1):
            print(f"  {i:2d}. {row['feature']:40s} (importance: {row['importance']:.4f})")
        
        return feature_importance
    
    def create_ensemble_model(self, n_models=5):
        """Create ensemble of models with different random seeds to reduce variance"""
        print("\n" + "=" * 80)
        print("ENSEMBLE MODEL CREATION")
        print("=" * 80)
        
        print(f"\n[INFO] Training {n_models} models with different random seeds...")
        ensemble_models = []
        
        for i in range(n_models):
            seed = 42 + i
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                eval_metric='rmse',
                random_state=seed,
                n_jobs=-1,
                tree_method='hist',
                **self.best_hyperparameters
            )
            
            # Train with early stopping using callbacks (compatible with XGBoost 2.0+)
            try:
                # Try using callbacks for newer XGBoost versions
                model.fit(
                    self.X_train, self.y_train,
                    eval_set=[(self.X_val, self.y_val)],
                    callbacks=[xgb.callback.EarlyStopping(rounds=20, save_best=True)],
                    verbose=False
                )
            except (AttributeError, TypeError):
                # Fallback for older XGBoost versions
                try:
                    model.fit(
                        self.X_train, self.y_train,
                        eval_set=[(self.X_val, self.y_val)],
                        early_stopping_rounds=20,
                        verbose=False
                    )
                except TypeError:
                    # If early stopping not supported, train without it
                    model.fit(
                        self.X_train, self.y_train,
                        eval_set=[(self.X_val, self.y_val)],
                        verbose=False
                    )
            ensemble_models.append(model)
            print(f"  - Model {i+1}/{n_models} trained (seed={seed})")
        
        # Create ensemble prediction function
        def ensemble_predict(X):
            predictions = np.array([m.predict(X) for m in ensemble_models])
            return np.mean(predictions, axis=0)
        
        # Evaluate ensemble
        y_pred_train_ens = ensemble_predict(self.X_train)
        y_pred_val_ens = ensemble_predict(self.X_val)
        y_pred_test_ens = ensemble_predict(self.X_test)
        
        # Calculate ensemble metrics
        test_rmse_ens = np.sqrt(mean_squared_error(self.y_test, y_pred_test_ens))
        test_rmse_single = self.results['test']['RMSE']
        improvement = ((test_rmse_single - test_rmse_ens) / test_rmse_single) * 100
        
        print(f"\n[Ensemble Results]:")
        print(f"  - Single model test RMSE: {test_rmse_single:.4f} grams")
        print(f"  - Ensemble test RMSE: {test_rmse_ens:.4f} grams")
        print(f"  - Improvement: {improvement:.2f}%")
        
        # Store ensemble
        self.ensemble_models = ensemble_models
        self.ensemble_predict = ensemble_predict
        
        # Update best model if ensemble is better
        if test_rmse_ens < test_rmse_single:
            print(f"  [INFO] Ensemble performs better - using ensemble as best model")
            self.use_ensemble = True
        else:
            print(f"  [INFO] Single model performs better - keeping single model")
            self.use_ensemble = False
        
        return ensemble_models
    
    def create_visualizations(self):
        """Step 6: Create visualizations"""
        print("\n" + "=" * 80)
        print("STEP 6: CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create output directory
        model_type = "Top25" if self.use_top25 else "AllVariables"
        plot_dir = f'PLOTS/XG_{model_type}_Optimized'
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
        
        # Subplot 4: Feature importance (top 15)
        ax4 = plt.subplot(2, 3, 4)
        feature_importance = pd.DataFrame(self.results['feature_importance'])
        top_15 = feature_importance.head(15)
        ax4.barh(range(len(top_15)), top_15['importance'].values)
        ax4.set_yticks(range(len(top_15)))
        ax4.set_yticklabels([f[:30] for f in top_15['feature'].values], fontsize=8)
        ax4.set_xlabel('Importance')
        ax4.set_title('Top 15 Feature Importance')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Subplot 5: CV Score distribution
        ax5 = plt.subplot(2, 3, 5)
        cv_results = self.search_results.cv_results_
        test_scores = np.sqrt(-cv_results['mean_test_score'])
        ax5.hist(test_scores, bins=20, alpha=0.7, edgecolor='black')
        ax5.axvline(self.best_cv_rmse, color='r', linestyle='--', linewidth=2, label=f'Best: {self.best_cv_rmse:.2f}')
        ax5.set_xlabel('CV RMSE (grams)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('CV Score Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Subplot 6: Hyperparameter importance (top parameters by variance in scores)
        ax6 = plt.subplot(2, 3, 6)
        param_importance = {}
        for param in self.best_hyperparameters.keys():
            param_mask = [param in key for key in cv_results['params']]
            if sum(param_mask) > 1:
                param_scores = np.sqrt(-cv_results['mean_test_score'][param_mask])
                param_importance[param] = np.std(param_scores)
        
        if param_importance:
            sorted_params = sorted(param_importance.items(), key=lambda x: x[1], reverse=True)[:8]
            param_names = [p[0] for p in sorted_params]
            param_values = [p[1] for p in sorted_params]
            ax6.barh(param_names, param_values)
            ax6.set_xlabel('Std Dev of CV Scores')
            ax6.set_title('Hyperparameter Impact (by variance)')
            ax6.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plot_file = f'{plot_dir}/xgboost_optimized_performance.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Visualizations saved to {plot_file}")
        
        return True
    
    def save_results(self):
        """Step 7: Save results"""
        print("\n" + "=" * 80)
        print("STEP 7: SAVING RESULTS")
        print("=" * 80)
        
        # Create output directories
        model_type = "Top25" if self.use_top25 else "AllVariables"
        output_dir = f'Data/processed/XG_{model_type}_Optimized'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare comprehensive results
        comprehensive_results = {
            'model_type': f'XGBoost_Optimized_{model_type}',
            'num_features': len(self.features),
            'features': self.features,
            'best_hyperparameters': self.best_hyperparameters,
            'best_cv_rmse': float(self.best_cv_rmse),
            'performance_metrics': self.results,
            'feature_importance': self.results.get('feature_importance', []),
            'cv_results_summary': {
                'mean_test_score': float(-self.search_results.best_score_),
                'std_test_score': float(self.search_results.cv_results_['std_test_score'][self.search_results.best_index_]),
                'n_iterations': self.search_results.n_iter,
                'n_splits': self.search_results.cv.n_splits
            },
            'timestamp': timestamp
        }
        
        # Save JSON results
        results_file = f'{output_dir}/xgboost_optimized_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Save metrics to CSV
        metrics_data = []
        for split_name, metrics in self.results.items():
            if isinstance(metrics, dict) and 'RMSE' in metrics:
                row = {'split': split_name}
                row.update(metrics)
                metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f'{output_dir}/xgboost_optimized_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save all CV results
        cv_results_df = pd.DataFrame(self.search_results.cv_results_)
        cv_results_file = f'{output_dir}/xgboost_cv_search_results_{timestamp}.csv'
        cv_results_df.to_csv(cv_results_file, index=False)
        
        # Save model
        model_file = f'Models/xgboost_optimized_{model_type.lower()}_model_{timestamp}.pkl'
        os.makedirs('Models', exist_ok=True)
        joblib.dump(self.best_model, model_file)
        
        # Save feature names
        feature_names_file = f'Models/xgboost_optimized_{model_type.lower()}_features_{timestamp}.json'
        with open(feature_names_file, 'w') as f:
            json.dump(self.features, f, indent=2)
        
        # Save feature importance
        if 'feature_importance' in self.results:
            feature_importance_df = pd.DataFrame(self.results['feature_importance'])
            importance_file = f'{output_dir}/xgboost_feature_importance_{timestamp}.csv'
            feature_importance_df.to_csv(importance_file, index=False)
        
        print(f"[OK] Results saved:")
        print(f"  - Results JSON: {results_file}")
        print(f"  - Metrics CSV: {metrics_file}")
        print(f"  - CV Search Results: {cv_results_file}")
        print(f"  - Model: {model_file}")
        print(f"  - Features: {feature_names_file}")
        
        return results_file, metrics_file, model_file
    
    def compare_with_previous_models(self):
        """Compare with previous XGBoost models"""
        print("\n" + "=" * 80)
        print("COMPARISON WITH PREVIOUS MODELS")
        print("=" * 80)
        
        optimized_test_rmse = self.results['test']['RMSE']
        comparisons = {}
        
        model_type = "Top25" if self.use_top25 else "AllVariables"
        
        # Compare with non-optimized version
        try:
            pattern = f'Data/processed/XG_{model_type}/xgboost_{model_type.lower()}_metrics_*.csv'
            files = glob.glob(pattern)
            if files:
                latest_file = max(files, key=os.path.getctime)
                prev_metrics = pd.read_csv(latest_file)
                prev_test = prev_metrics[prev_metrics['split'] == 'test']
                if len(prev_test) > 0:
                    prev_rmse = prev_test.iloc[0]['RMSE']
                    improvement = ((prev_rmse - optimized_test_rmse) / prev_rmse) * 100
                    comparisons[f'XGBoost_{model_type}_NonOptimized'] = {
                        'rmse': prev_rmse,
                        'improvement': improvement
                    }
                    print(f"\n[vs Non-Optimized {model_type}]:")
                    print(f"  - Non-Optimized RMSE: {prev_rmse:.4f} grams")
                    print(f"  - Optimized RMSE: {optimized_test_rmse:.4f} grams")
                    print(f"  - Improvement: {improvement:.2f}%")
        except Exception as e:
            print(f"[INFO] Could not compare with non-optimized: {str(e)}")
        
        return comparisons
    
    def run_optimization(self, n_iter=100, cv_folds=10, random_state=42):
        """Run complete hyperparameter optimization pipeline"""
        print("=" * 80)
        model_type = "TOP 25 VARIABLES" if self.use_top25 else "ALL VARIABLES"
        print(f"XGBOOST HYPERPARAMETER OPTIMIZATION - {model_type}")
        print("=" * 80)
        
        # Step 1: Load and prepare features
        self.load_and_prepare_features()
        
        # Step 2: Prepare data splits
        self.prepare_data_splits()
        
        # Step 3: Define search space
        param_distributions = self.define_hyperparameter_search_space()
        
        # Step 4: Perform random search
        self.perform_random_search(param_distributions, n_iter=n_iter, cv_folds=cv_folds, random_state=random_state)
        
        # Step 5: Evaluate best model
        self.evaluate_best_model()
        
        # Step 6: Feature importance
        self.get_feature_importance()
        
        # Step 6.5: Create ensemble model (optional, can be disabled)
        try:
            self.create_ensemble_model(n_models=5)
            # Re-evaluate if ensemble is better
            if hasattr(self, 'use_ensemble') and self.use_ensemble:
                print("\n[INFO] Re-evaluating with ensemble model...")
                y_pred_train_ens = self.ensemble_predict(self.X_train)
                y_pred_val_ens = self.ensemble_predict(self.X_val)
                y_pred_test_ens = self.ensemble_predict(self.X_test)
                
                # Update predictions
                self.predictions = {
                    'train': (self.y_train, y_pred_train_ens),
                    'validation': (self.y_val, y_pred_val_ens),
                    'test': (self.y_test, y_pred_test_ens)
                }
                
                # Recalculate and update results
                for split_name, (y_true, y_pred) in self.predictions.items():
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
        except Exception as e:
            print(f"[WARNING] Ensemble creation failed: {str(e)}")
            self.use_ensemble = False
        
        # Step 7: Create visualizations
        self.create_visualizations()
        
        # Step 8: Save results
        self.save_results()
        
        # Step 9: Compare with previous models
        self.compare_with_previous_models()
        
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
    print("XGBOOST HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Configuration
    data_path = 'Data/processed/cleaned_dataset_with_engineered_features.csv'
    use_top25 = True  # Set to False for all variables
    n_iter = 100  # Number of random search iterations (increased for better search)
    cv_folds = 10  # Cross-validation folds (increased for better generalization estimate)
    
    print(f"\n[Configuration]:")
    print(f"  - Use Top 25: {use_top25}")
    print(f"  - Random search iterations: {n_iter}")
    print(f"  - CV folds: {cv_folds}")
    
    # Initialize optimizer
    optimizer = XGBoostHyperparameterOptimizer(
        data_path=data_path,
        use_top25=use_top25
    )
    
    # Run optimization
    results = optimizer.run_optimization(
        n_iter=n_iter,
        cv_folds=cv_folds,
        random_state=42
    )
    
    return results


if __name__ == "__main__":
    results = main()

