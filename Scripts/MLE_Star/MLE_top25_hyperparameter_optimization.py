"""
Hyperparameter Optimization for MLE Top 25 Model
==================================================

This script performs hyperparameter optimization for the MLE model using
top 25 variables from feature importance ranking.

Hyperparameters optimized:
- max_iter: Maximum iterations for EM algorithm
- tol: Convergence tolerance for EM algorithm
- cov_reg: Covariance matrix regularization parameter
- cov_reg_conditional: Regularization for conditional covariance in prediction

Date: 2024
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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import json
import warnings
import os
import itertools
from datetime import datetime
import sys
try:
    from skopt import gp_minimize  # type: ignore[import-untyped]
    from skopt.space import Real, Integer, Categorical  # type: ignore[import-untyped]
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    # scikit-optimize is optional - install with: pip install scikit-optimize
    # Bayesian optimization will be disabled if not available
    gp_minimize = None  # type: ignore
    Real = None  # type: ignore
    Integer = None  # type: ignore
    Categorical = None  # type: ignore

# Add current directory to path to import OptimizedMLE
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import the base MLE class
from MLE_top_25 import OptimizedMLE

warnings.filterwarnings('ignore')


class OptimizedMLEHyperparameter(OptimizedMLE):
    """
    Extended OptimizedMLE class with hyperparameter support
    """
    
    def __init__(self, data_path, ranking_path='Data/processed/MLE_New/feature_importance_ranking.csv',
                 max_iter=100, tol=1e-4, cov_reg=1e-6, cov_reg_conditional=1e-6,
                 use_shrinkage=False, shrinkage_alpha=0.1,
                 prediction_method='mle', ridge_alpha=1.0,
                 use_feature_scaling=False,
                 em_init_method='nanmean',
                 knn_neighbors=5, elastic_l1_ratio=0.5,
                 early_stopping_patience=10, use_ensemble=False,
                 n_imputations=1, use_robust_scaling=False):
        """Initialize with hyperparameters"""
        super().__init__(data_path, ranking_path)
        # Ensure max_iter is an integer (may be float from JSON)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.cov_reg = float(cov_reg)
        self.cov_reg_conditional = float(cov_reg_conditional)
        
        # Advanced optimization parameters
        self.use_shrinkage = bool(use_shrinkage) if isinstance(use_shrinkage, (bool, int)) else use_shrinkage
        self.shrinkage_alpha = float(shrinkage_alpha)
        self.prediction_method = str(prediction_method)  # 'mle', 'ridge', 'lasso', 'elastic', 'bayesian'
        self.ridge_alpha = float(ridge_alpha)
        self.use_feature_scaling = bool(use_feature_scaling) if isinstance(use_feature_scaling, (bool, int)) else use_feature_scaling
        self.em_init_method = str(em_init_method)  # 'nanmean', 'complete_case', 'median', 'knn'
        self.knn_neighbors = int(knn_neighbors)
        self.elastic_l1_ratio = float(elastic_l1_ratio)
        self.early_stopping_patience = int(early_stopping_patience)
        self.use_ensemble = bool(use_ensemble) if isinstance(use_ensemble, (bool, int)) else use_ensemble
        self.n_imputations = int(n_imputations)
        self.use_robust_scaling = bool(use_robust_scaling) if isinstance(use_robust_scaling, (bool, int)) else use_robust_scaling
        
        self.scaler = None
        self.hyperparameters = {
            'max_iter': self.max_iter,
            'tol': self.tol,
            'cov_reg': self.cov_reg,
            'cov_reg_conditional': self.cov_reg_conditional,
            'use_shrinkage': self.use_shrinkage,
            'shrinkage_alpha': self.shrinkage_alpha,
            'prediction_method': self.prediction_method,
            'ridge_alpha': self.ridge_alpha,
            'use_feature_scaling': self.use_feature_scaling,
            'em_init_method': self.em_init_method,
            'knn_neighbors': self.knn_neighbors,
            'elastic_l1_ratio': self.elastic_l1_ratio,
            'early_stopping_patience': self.early_stopping_patience,
            'use_ensemble': self.use_ensemble,
            'n_imputations': self.n_imputations,
            'use_robust_scaling': self.use_robust_scaling
        }
    
    def _implement_continuous_likelihood(self):
        """Implement multivariate normal likelihood function with hyperparameter"""
        print("\n--- Continuous Likelihood Function ---")
        
        cov_reg = self.cov_reg
        
        def multivariate_normal_log_likelihood(data, mean, cov):
            """Calculate log-likelihood for multivariate normal distribution"""
            try:
                n, p = data.shape
                # Add regularization to covariance matrix
                cov_reg_matrix = cov + np.eye(p) * cov_reg
                
                # Calculate log-likelihood
                log_lik = -0.5 * n * p * np.log(2 * np.pi)
                log_lik -= 0.5 * n * np.log(np.linalg.det(cov_reg_matrix))
                
                # Calculate quadratic form
                diff = data - mean
                inv_cov = np.linalg.inv(cov_reg_matrix)
                quadratic = np.sum(diff @ inv_cov * diff)
                log_lik -= 0.5 * quadratic
                
                return log_lik
            except np.linalg.LinAlgError:
                return -np.inf
        
        self.continuous_likelihood = multivariate_normal_log_likelihood
        print(f"[OK] Multivariate normal likelihood function implemented (cov_reg={cov_reg})")
    
    def _apply_covariance_shrinkage(self, cov_matrix):
        """
        Apply enhanced Ledoit-Wolf style shrinkage with automatic alpha selection
        """
        if not self.use_shrinkage:
            return cov_matrix
        
        n_features = cov_matrix.shape[0]
        
        # Use optimal shrinkage coefficient based on sample size if available
        if hasattr(self, 'data_continuous') and self.data_continuous is not None:
            n_samples = self.data_continuous.shape[0]
            # Optimal shrinkage: balance between sample size and feature count
            optimal_alpha = min(self.shrinkage_alpha, 
                                n_features / (n_samples + n_features))
        else:
            optimal_alpha = self.shrinkage_alpha
        
        # Calculate target (shrink towards identity matrix scaled by trace)
        trace_cov = np.trace(cov_matrix)
        target = (trace_cov / n_features) * np.eye(n_features)
        
        # Enhanced shrinkage: shrink towards identity matrix (better for high dimensions)
        shrunk_cov = (1 - optimal_alpha) * cov_matrix + optimal_alpha * target
        
        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(shrunk_cov)
        if np.any(eigenvals <= 0):
            # Add small regularization if needed
            shrunk_cov += np.eye(n_features) * (abs(np.min(eigenvals)) + 1e-8)
        
        return shrunk_cov
    
    def _improved_em_initialization(self, data):
        """
        Improved initialization strategies for EM algorithm
        """
        if self.em_init_method == 'complete_case':
            # Use only complete cases
            complete_mask = ~np.isnan(data).any(axis=1)
            if complete_mask.sum() > 0:
                complete_data = data[complete_mask]
                mean_init = np.mean(complete_data, axis=0)
                cov_init = np.cov(complete_data.T, rowvar=True, bias=True)
            else:
                mean_init = np.nanmean(data, axis=0)
                cov_init = np.cov(data.T, rowvar=True, bias=True)
                
        elif self.em_init_method == 'median':
            # Median imputation
            data_imputed = data.copy()
            for i in range(data.shape[1]):
                median_val = np.nanmedian(data[:, i])
                if np.isnan(median_val):
                    median_val = 0.0
                data_imputed[np.isnan(data[:, i]), i] = median_val
            mean_init = np.mean(data_imputed, axis=0)
            cov_init = np.cov(data_imputed.T, rowvar=True, bias=True)
            
        elif self.em_init_method == 'knn':
            # KNN imputation for initialization with configurable neighbors
            try:
                imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                data_imputed = imputer.fit_transform(data)
                mean_init = np.mean(data_imputed, axis=0)
                cov_init = np.cov(data_imputed.T, rowvar=True, bias=True)
            except:
                # Fallback to median if KNN fails
                data_imputed = data.copy()
                for i in range(data.shape[1]):
                    median_val = np.nanmedian(data[:, i])
                    if np.isnan(median_val):
                        median_val = 0.0
                    data_imputed[np.isnan(data[:, i]), i] = median_val
                mean_init = np.mean(data_imputed, axis=0)
                cov_init = np.cov(data_imputed.T, rowvar=True, bias=True)
                
        else:  # 'nanmean' - default
            # Standard nanmean initialization
            mean_init = np.nanmean(data, axis=0)
            cov_init = np.cov(data.T, rowvar=True, bias=True)
        
        # Handle NaN in covariance
        if np.any(np.isnan(cov_init)):
            cov_init = np.eye(data.shape[1])
        
        # Apply regularization
        cov_init = cov_init + np.eye(data.shape[1]) * self.cov_reg
        
        return mean_init, cov_init
    
    def _implement_simple_em(self):
        """Implement simple EM algorithm with hyperparameters"""
        print("\n--- Simple EM Algorithm ---")
        
        max_iter = int(self.max_iter)  # Ensure integer
        tol = float(self.tol)
        cov_reg = float(self.cov_reg)
        
        def simple_em(data, max_iter=max_iter, tol=tol):
            """Simple EM algorithm for missing data"""
            # Ensure max_iter is integer (may be passed as float)
            max_iter = int(max_iter)
            
            # Use improved initialization
            mean_init, cov_init = self._improved_em_initialization(data)
            
            current_mean = mean_init.copy()
            current_cov = cov_init.copy()
            
            prev_likelihood = -np.inf
            likelihood_history = []
            best_likelihood = -np.inf
            patience_counter = 0
            
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
                                cov_obs_inv = np.linalg.inv(cov_obs + np.eye(cov_obs.shape[0]) * cov_reg)
                                conditional_mean = mu_miss + cov_miss_obs @ cov_obs_inv @ (data[i][observed_mask] - mu_obs)
                                data_imputed[i][missing_mask] = conditional_mean
                            except:
                                data_imputed[i][missing_mask] = mu_miss
                        else:
                            data_imputed[i][missing_mask] = current_mean[missing_mask]
                
                # M-step: Update parameters
                current_mean = np.mean(data_imputed, axis=0)
                current_cov = np.cov(data_imputed.T, rowvar=True, bias=True) + np.eye(data_imputed.shape[1]) * cov_reg
                
                # Calculate likelihood
                current_likelihood = self.continuous_likelihood(data_imputed, current_mean, current_cov)
                likelihood_history.append(current_likelihood)
                
                # Early stopping: check if likelihood improved
                if current_likelihood > best_likelihood:
                    best_likelihood = current_likelihood
                    patience_counter = 0
                    best_mean = current_mean.copy()
                    best_cov = current_cov.copy()
                else:
                    patience_counter += 1
                
                # Check convergence
                if abs(current_likelihood - prev_likelihood) < tol:
                    break
                
                # Early stopping: stop if no improvement for patience iterations
                if patience_counter >= self.early_stopping_patience and iteration > 10:
                    current_mean = best_mean
                    current_cov = best_cov
                    current_likelihood = best_likelihood
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
        print(f"[OK] Simple EM algorithm implemented (max_iter={max_iter}, tol={tol})")
    
    def _run_em_algorithm(self):
        """Run EM algorithm on prepared data with hyperparameters"""
        print("\n--- Running EM Algorithm ---")
        
        # Run EM algorithm (ensure max_iter is int)
        self.mle_results = self.simple_em(
            self.data_continuous,
            max_iter=int(self.max_iter),
            tol=float(self.tol)
        )
        
        print(f"[OK] EM algorithm completed")
        print(f"  - Iterations: {self.mle_results['iterations']}")
        print(f"  - Converged: {self.mle_results['converged']}")
        print(f"  - Final likelihood: {self.mle_results['likelihood']:.4f}")
    
    def _prepare_mle_data(self):
        """Prepare data for MLE analysis with optional feature scaling"""
        print("\n--- Preparing Data for MLE ---")
        
        # Select key variables for analysis
        analysis_vars = self.key_continuous
        analysis_vars = [var for var in analysis_vars if var in self.data.columns]
        
        # Prepare continuous data
        self.data_continuous = self.data[analysis_vars].values
        self.analysis_vars = analysis_vars
        
        # Apply feature scaling if enabled
        if self.use_feature_scaling:
            print("[INFO] Applying feature scaling to data")
            if self.use_robust_scaling:
                self.scaler = RobustScaler()
                print("[INFO] Using robust scaling (median and IQR)")
            else:
                self.scaler = StandardScaler()
                print("[INFO] Using standard scaling (mean and std)")
            self.data_continuous_scaled = self.scaler.fit_transform(self.data_continuous)
            # Store original data for inverse transform if needed
            self.feature_means = self.scaler.center_ if hasattr(self.scaler, 'center_') else self.scaler.mean_
            self.feature_stds = self.scaler.scale_
            # Use scaled data for EM
            self.data_continuous = self.data_continuous_scaled
        else:
            self.data_continuous_scaled = None
            self.scaler = None
        
        print(f"[OK] Continuous data shape: {self.data_continuous.shape}")
        print(f"[OK] Variables: {analysis_vars}")
        print(f"[TOP 25 MODEL] Using {len(analysis_vars)} variables (target + top 25 predictors)")
        if self.use_feature_scaling:
            print(f"[INFO] Features scaled: mean={self.feature_means[0]:.2f}, std={self.feature_stds[0]:.2f} (example)")
    
    def _ensemble_prediction(self, mean_vector, cov_matrix, X, y_actual=None):
        """
        Ensemble prediction combining multiple methods
        """
        predictions_list = []
        weights = []
        
        # MLE prediction
        try:
            pred_mle, _ = self._mle_prediction_only(mean_vector, cov_matrix, X)
            predictions_list.append(pred_mle)
            weights.append(0.3)
        except:
            pass
        
        # Ridge prediction
        try:
            pred_ridge, _ = self._ridge_prediction_only(mean_vector, cov_matrix, X, y_actual)
            predictions_list.append(pred_ridge)
            weights.append(0.3)
        except:
            pass
        
        # Bayesian prediction
        try:
            pred_bayesian, _ = self._bayesian_prediction_only(mean_vector, cov_matrix, X, y_actual)
            predictions_list.append(pred_bayesian)
            weights.append(0.4)
        except:
            pass
        
        if len(predictions_list) == 0:
            # Fallback to MLE
            pred_mle, _ = self._mle_prediction_only(mean_vector, cov_matrix, X)
            return pred_mle, None
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted average
        ensemble_pred = np.average(predictions_list, axis=0, weights=weights)
        return ensemble_pred, None
    
    def _mle_prediction_only(self, mean_vector, cov_matrix, X):
        """MLE prediction only"""
        mu_y = mean_vector[0]
        mu_x = mean_vector[1:]
        X_centered = X - mu_x
        
        sigma_yx = cov_matrix[0, 1:]
        sigma_xx = cov_matrix[1:, 1:]
        
        if self.use_shrinkage:
            sigma_xx = self._apply_covariance_shrinkage(sigma_xx)
        
        sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * self.cov_reg_conditional)
        beta = sigma_xx_inv @ sigma_yx
        predictions = mu_y + X_centered @ beta
        return predictions, beta
    
    def _ridge_prediction_only(self, mean_vector, cov_matrix, X, y_actual=None):
        """Ridge prediction only"""
        mu_y = mean_vector[0]
        mu_x = mean_vector[1:]
        X_centered = X - mu_x
        
        if y_actual is not None:
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_centered)
            y_centered = y_actual - mu_y
            
            model = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            model.fit(X_scaled, y_centered)
            predictions = mu_y + model.predict(X_scaled)
            return predictions, model.coef_
        else:
            sigma_yx = cov_matrix[0, 1:]
            sigma_xx = cov_matrix[1:, 1:]
            
            if self.use_shrinkage:
                sigma_xx = self._apply_covariance_shrinkage(sigma_xx)
            
            sigma_xx_reg = sigma_xx + self.ridge_alpha * np.eye(sigma_xx.shape[0])
            sigma_xx_inv = np.linalg.inv(sigma_xx_reg + np.eye(sigma_xx.shape[0]) * self.cov_reg_conditional)
            beta = sigma_xx_inv @ sigma_yx
            predictions = mu_y + X_centered @ beta
            return predictions, beta
    
    def _bayesian_prediction_only(self, mean_vector, cov_matrix, X, y_actual=None):
        """Bayesian prediction only"""
        mu_y = mean_vector[0]
        mu_x = mean_vector[1:]
        X_centered = X - mu_x
        
        if y_actual is not None:
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X_centered)
            y_centered = y_actual - mu_y
            
            model = BayesianRidge(
                n_iter=300,
                alpha_1=1e-6,
                alpha_2=1e-6,
                lambda_1=1e-6,
                lambda_2=1e-6,
                fit_intercept=False
            )
            model.fit(X_scaled, y_centered)
            predictions = mu_y + model.predict(X_scaled)
            return predictions, model.coef_
        else:
            # Fallback to Ridge covariance-based
            sigma_yx = cov_matrix[0, 1:]
            sigma_xx = cov_matrix[1:, 1:]
            if self.use_shrinkage:
                sigma_xx = self._apply_covariance_shrinkage(sigma_xx)
            sigma_xx_reg = sigma_xx + self.ridge_alpha * np.eye(sigma_xx.shape[0])
            sigma_xx_inv = np.linalg.inv(sigma_xx_reg + np.eye(sigma_xx.shape[0]) * self.cov_reg_conditional)
            beta = sigma_xx_inv @ sigma_yx
            predictions = mu_y + X_centered @ beta
            return predictions, beta
    
    def _advanced_prediction(self, mean_vector, cov_matrix, X, y_actual=None):
        """
        Advanced prediction methods using regularization
        Returns predictions and optional model coefficients
        """
        # Use ensemble if enabled
        if self.use_ensemble:
            return self._ensemble_prediction(mean_vector, cov_matrix, X, y_actual)
        
        mu_y = mean_vector[0]
        mu_x = mean_vector[1:]
        
        # Center predictors
        X_centered = X - mu_x
        
        if self.prediction_method == 'mle':
            # Original MLE method
            sigma_yx = cov_matrix[0, 1:]
            sigma_xx = cov_matrix[1:, 1:]
            
            # Apply shrinkage if enabled
            if self.use_shrinkage:
                sigma_xx = self._apply_covariance_shrinkage(sigma_xx)
            
            try:
                sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * self.cov_reg_conditional)
                beta = sigma_xx_inv @ sigma_yx
                predictions = mu_y + X_centered @ beta
                return predictions, beta
            except np.linalg.LinAlgError:
                # Fallback
                beta = np.linalg.lstsq(X_centered, y_actual - mu_y if y_actual is not None else None, rcond=None)[0]
                predictions = mu_y + X_centered @ beta
                return predictions, beta
                
        elif self.prediction_method == 'ridge':
            # Ridge regression
            if y_actual is not None:
                # Use actual y values for training
                scaler_X = StandardScaler()
                X_scaled = scaler_X.fit_transform(X_centered)
                y_centered = y_actual - mu_y
                
                model = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
                model.fit(X_scaled, y_centered)
                predictions = mu_y + model.predict(X_scaled)
                return predictions, model.coef_
            else:
                # Use covariance-based estimate with Ridge regularization
                sigma_yx = cov_matrix[0, 1:]
                sigma_xx = cov_matrix[1:, 1:]
                
                if self.use_shrinkage:
                    sigma_xx = self._apply_covariance_shrinkage(sigma_xx)
                
                sigma_xx_reg = sigma_xx + self.ridge_alpha * np.eye(sigma_xx.shape[0])
                sigma_xx_inv = np.linalg.inv(sigma_xx_reg + np.eye(sigma_xx.shape[0]) * self.cov_reg_conditional)
                beta = sigma_xx_inv @ sigma_yx
                predictions = mu_y + X_centered @ beta
                return predictions, beta
                
        elif self.prediction_method == 'lasso':
            # Lasso regression
            if y_actual is not None:
                scaler_X = StandardScaler()
                X_scaled = scaler_X.fit_transform(X_centered)
                y_centered = y_actual - mu_y
                
                model = Lasso(alpha=self.ridge_alpha, fit_intercept=False, max_iter=1000)
                model.fit(X_scaled, y_centered)
                predictions = mu_y + model.predict(X_scaled)
                return predictions, model.coef_
            else:
                # Fallback to Ridge covariance-based
                sigma_yx = cov_matrix[0, 1:]
                sigma_xx = cov_matrix[1:, 1:]
                if self.use_shrinkage:
                    sigma_xx = self._apply_covariance_shrinkage(sigma_xx)
                sigma_xx_reg = sigma_xx + self.ridge_alpha * np.eye(sigma_xx.shape[0])
                sigma_xx_inv = np.linalg.inv(sigma_xx_reg + np.eye(sigma_xx.shape[0]) * self.cov_reg_conditional)
                beta = sigma_xx_inv @ sigma_yx
                predictions = mu_y + X_centered @ beta
                return predictions, beta
                
        elif self.prediction_method == 'elastic':
            # Elastic Net
            if y_actual is not None:
                scaler_X = StandardScaler()
                X_scaled = scaler_X.fit_transform(X_centered)
                y_centered = y_actual - mu_y
                
                model = ElasticNet(alpha=self.ridge_alpha, l1_ratio=self.elastic_l1_ratio, fit_intercept=False, max_iter=1000)
                model.fit(X_scaled, y_centered)
                predictions = mu_y + model.predict(X_scaled)
                return predictions, model.coef_
            else:
                # Fallback to Ridge covariance-based
                sigma_yx = cov_matrix[0, 1:]
                sigma_xx = cov_matrix[1:, 1:]
                if self.use_shrinkage:
                    sigma_xx = self._apply_covariance_shrinkage(sigma_xx)
                sigma_xx_reg = sigma_xx + self.ridge_alpha * np.eye(sigma_xx.shape[0])
                sigma_xx_inv = np.linalg.inv(sigma_xx_reg + np.eye(sigma_xx.shape[0]) * self.cov_reg_conditional)
                beta = sigma_xx_inv @ sigma_yx
                predictions = mu_y + X_centered @ beta
                return predictions, beta
                
        elif self.prediction_method == 'bayesian':
            # Bayesian Ridge Regression
            if y_actual is not None:
                scaler_X = StandardScaler()
                X_scaled = scaler_X.fit_transform(X_centered)
                y_centered = y_actual - mu_y
                
                model = BayesianRidge(
                    n_iter=300,
                    alpha_1=1e-6,
                    alpha_2=1e-6,
                    lambda_1=1e-6,
                    lambda_2=1e-6,
                    fit_intercept=False
                )
                model.fit(X_scaled, y_centered)
                predictions = mu_y + model.predict(X_scaled)
                return predictions, model.coef_
            else:
                # Fallback to Ridge covariance-based
                sigma_yx = cov_matrix[0, 1:]
                sigma_xx = cov_matrix[1:, 1:]
                if self.use_shrinkage:
                    sigma_xx = self._apply_covariance_shrinkage(sigma_xx)
                sigma_xx_reg = sigma_xx + self.ridge_alpha * np.eye(sigma_xx.shape[0])
                sigma_xx_inv = np.linalg.inv(sigma_xx_reg + np.eye(sigma_xx.shape[0]) * self.cov_reg_conditional)
                beta = sigma_xx_inv @ sigma_yx
                predictions = mu_y + X_centered @ beta
                return predictions, beta
        else:
            # Default to MLE
            sigma_yx = cov_matrix[0, 1:]
            sigma_xx = cov_matrix[1:, 1:]
            if self.use_shrinkage:
                sigma_xx = self._apply_covariance_shrinkage(sigma_xx)
            try:
                sigma_xx_inv = np.linalg.inv(sigma_xx + np.eye(sigma_xx.shape[0]) * self.cov_reg_conditional)
                beta = sigma_xx_inv @ sigma_yx
                predictions = mu_y + X_centered @ beta
                return predictions, beta
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(X_centered, y_actual - mu_y if y_actual is not None else None, rcond=None)[0]
                predictions = mu_y + X_centered @ beta
                return predictions, beta
    
    def _calculate_rmse(self):
        """Calculate RMSE with advanced prediction methods"""
        print("\n--- Calculating Performance Metrics ---")
        
        # Get MLE parameters
        mean_vector = self.mle_results['mean']
        cov_matrix = self.mle_results['covariance']
        
        # Apply covariance shrinkage if enabled
        if self.use_shrinkage:
            cov_matrix = self._apply_covariance_shrinkage(cov_matrix)
        
        # Prepare data for evaluation (only complete cases)
        eval_data = self.data[self.analysis_vars].dropna()
        
        # Extract actual birthweight values
        actual_values = eval_data['f1_bw'].values
        
        # Prepare predictor variables (excluding birthweight)
        predictor_vars = [var for var in self.analysis_vars if var != 'f1_bw']
        X = eval_data[predictor_vars].values
        
        # If feature scaling was used, we need to handle it
        # Note: The mean_vector and cov_matrix are from scaled data if scaling was used
        # So X should also be scaled if scaling was used
        # CRITICAL: If scaling was used, predictions will be in scaled space and need inverse transform
        if self.use_feature_scaling and self.scaler is not None:
            # Get indices of predictor vars (excluding target which is first)
            target_idx = 0
            pred_indices = list(range(1, len(self.analysis_vars)))
            
            # Scale X using the scaler (only predictor columns)
            X_scaled = self.scaler.transform(eval_data[self.analysis_vars].values)[:, pred_indices]
            # But we need to work with the scaled mean and cov
            # The mean_vector and cov_matrix are already from scaled data
            X = X_scaled
            # Store flag to inverse transform predictions later
            need_inverse_transform = True
        else:
            need_inverse_transform = False
        
        # Use advanced prediction method
        try:
            # Print hyperparameters being used for debugging
            print(f"[DEBUG] Using prediction_method='{self.prediction_method}', "
                  f"cov_reg_conditional={self.cov_reg_conditional:.2e}, "
                  f"use_shrinkage={self.use_shrinkage}, "
                  f"use_feature_scaling={self.use_feature_scaling}")
            
            predictions, _ = self._advanced_prediction(mean_vector, cov_matrix, X, y_actual=actual_values)
            # Ensure predictions is 1D
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            # CRITICAL FIX: If feature scaling was used, inverse transform predictions back to original scale
            if need_inverse_transform and self.scaler is not None:
                # Create a dummy array with target in first position, then inverse transform
                # We only need to inverse transform the target (birthweight) prediction
                # The scaler was fit on all features, so we need to reconstruct the full feature vector
                # For prediction, we only have the target value, so we'll use the scaler's inverse transform
                # on a dummy array where target is first
                target_mean = self.scaler.mean_[0]
                target_std = self.scaler.scale_[0]
                # Inverse transform: original = scaled * std + mean
                predictions = predictions * target_std + target_mean
        except Exception as e:
            print(f"[WARNING] Advanced prediction failed ({str(e)}), using fallback")
            # Fallback: use simple linear relationship
            mu_y = mean_vector[0]
            mu_x = mean_vector[1:]
            X_centered = X - mu_x
            beta = np.linalg.lstsq(X_centered, actual_values - mu_y, rcond=None)[0]
            predictions = mu_y + X_centered @ beta
            
            # If scaling was used, inverse transform
            if need_inverse_transform and self.scaler is not None:
                target_mean = self.scaler.mean_[0]
                target_std = self.scaler.scale_[0]
                predictions = predictions * target_std + target_mean
        
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
        
        return rmse  # Return RMSE for optimization
    
    def evaluate_on_fold(self, train_indices, val_indices):
        """
        Evaluate model on a single fold
        
        Parameters:
        -----------
        train_indices : array-like
            Training data indices
        val_indices : array-like
            Validation data indices
        
        Returns:
        --------
        float : RMSE on validation set
        """
        # Ensure key_continuous is initialized
        if not hasattr(self, 'key_continuous'):
            # If not initialized, we need to load and analyze
            # But we'll use the subset data
            train_data = self.data.iloc[train_indices]
            
            # Quick initialization with subset
            if hasattr(self, 'top_25_vars'):
                self.key_continuous = ['f1_bw'] + self.top_25_vars
            else:
                # Fallback: try to load ranking
                try:
                    ranking_df = pd.read_csv(self.ranking_path)
                    top_25_vars = ranking_df['variable'].head(25).tolist()
                    self.key_continuous = ['f1_bw'] + top_25_vars
                except:
                    # Last resort: use all numeric columns
                    numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                    self.key_continuous = ['f1_bw'] + [col for col in numeric_cols if col != 'f1_bw'][:25]
        
        # Prepare data
        train_data = self.data.iloc[train_indices]
        val_data = self.data.iloc[val_indices]
        
        # Select key variables
        analysis_vars = self.key_continuous
        analysis_vars = [var for var in analysis_vars if var in self.data.columns]
        
        # Prepare continuous data for training
        train_continuous = train_data[analysis_vars].values
        
        # Handle feature scaling if enabled
        fold_scaler = None
        if self.use_feature_scaling:
            if self.use_robust_scaling:
                fold_scaler = RobustScaler()
            else:
                fold_scaler = StandardScaler()
            train_continuous = fold_scaler.fit_transform(train_continuous)
        
        # Run EM algorithm on training data (ensure max_iter is int)
        mle_results = self.simple_em(train_continuous, max_iter=int(self.max_iter), tol=float(self.tol))
        
        # Prepare validation data (only complete cases)
        val_complete = val_data[analysis_vars].dropna()
        
        if len(val_complete) == 0:
            return np.inf  # Return high error if no complete cases
        
        # Extract actual and predictor values
        actual_values = val_complete['f1_bw'].values
        predictor_vars = [var for var in analysis_vars if var != 'f1_bw']
        X = val_complete[predictor_vars].values
        
        # Scale validation data if scaling was used
        if self.use_feature_scaling and fold_scaler is not None:
            # Scale the full validation data (including target)
            val_scaled = fold_scaler.transform(val_complete[analysis_vars].values)
            # Extract scaled predictors (target is first column)
            X = val_scaled[:, 1:]  # Skip target column
            need_inverse_transform = True
        else:
            need_inverse_transform = False
        
        # Get parameters
        mean_vector = mle_results['mean']
        cov_matrix = mle_results['covariance']
        
        # Apply shrinkage if enabled
        if self.use_shrinkage:
            cov_matrix = self._apply_covariance_shrinkage(cov_matrix)
        
        # Use advanced prediction method
        try:
            predictions, _ = self._advanced_prediction(mean_vector, cov_matrix, X, y_actual=actual_values)
            predictions = predictions.flatten() if predictions.ndim > 1 else predictions
            
            # Inverse transform predictions if scaling was used
            if need_inverse_transform and fold_scaler is not None:
                target_mean = fold_scaler.mean_[0]
                target_std = fold_scaler.scale_[0]
                predictions = predictions * target_std + target_mean
        except Exception as e:
            # Fallback to simple method
            mu_y = mean_vector[0]
            mu_x = mean_vector[1:]
            X_centered = X - mu_x
            beta = np.linalg.lstsq(X_centered, actual_values - mu_y, rcond=None)[0]
            predictions = mu_y + X_centered @ beta
            
            # Inverse transform if scaling was used
            if need_inverse_transform and fold_scaler is not None:
                if hasattr(fold_scaler, 'center_'):
                    # RobustScaler
                    target_center = fold_scaler.center_[0]
                    target_scale = fold_scaler.scale_[0]
                    predictions = predictions * target_scale + target_center
                else:
                    # StandardScaler
                    target_mean = fold_scaler.mean_[0]
                    target_std = fold_scaler.scale_[0]
                    predictions = predictions * target_std + target_mean
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        
        return rmse


def evaluate_hyperparameters(mle_model, data, cv_folds=7, random_state=42, use_multiple_metrics=False):
    """
    Evaluate hyperparameters using cross-validation
    
    Parameters:
    -----------
    mle_model : OptimizedMLEHyperparameter
        MLE model instance with hyperparameters
    data : pd.DataFrame
        Full dataset
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    float : Mean RMSE across folds
    """
    # Initialize data in model - but we'll do minimal setup for CV
    mle_model.data = data
    
    # Only do the necessary setup for CV (skip full analysis)
    # Load variable grouping and top 25 variables
    mle_model.variable_groups = pd.read_csv('Data/processed/variable_grouping_table.csv')
    
    # Load top 25 variables
    ranking_df = pd.read_csv(mle_model.ranking_path)
    mle_model.top_25_vars = ranking_df['variable'].head(25).tolist()
    
    # Set up key variables
    mle_model.key_continuous = ['f1_bw'] + mle_model.top_25_vars
    mle_model.key_continuous = [var for var in mle_model.key_continuous if var in data.columns]
    mle_model.key_categorical = []
    
    # Implement likelihood and EM (these are needed for evaluate_on_fold)
    mle_model._implement_continuous_likelihood()
    mle_model._implement_simple_em()
    
    # Prepare data for cross-validation
    analysis_vars = mle_model.key_continuous
    analysis_vars = [var for var in analysis_vars if var in data.columns]
    
    # Get complete cases for CV
    complete_data = data[analysis_vars].dropna()
    
    if len(complete_data) < cv_folds:
        print(f"[WARNING] Not enough complete cases ({len(complete_data)}) for {cv_folds} folds")
        cv_folds = min(3, len(complete_data) // 2)
    
    # Create KFold splitter
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Get original indices of complete cases
    complete_indices = complete_data.index.values
    
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(complete_data)):
        train_original_indices = complete_indices[train_idx]
        val_original_indices = complete_indices[val_idx]
        
        try:
            rmse = mle_model.evaluate_on_fold(train_original_indices, val_original_indices)
            rmse_scores.append(rmse)
            
            if use_multiple_metrics:
                # Calculate additional metrics if needed
                # This would require modifying evaluate_on_fold to return more metrics
                pass
        except Exception as e:
            print(f"[WARNING] Fold {fold_idx + 1} failed: {str(e)}")
            rmse_scores.append(np.inf)
    
    mean_rmse = np.mean(rmse_scores) if rmse_scores else np.inf
    
    # Use robust metrics (median) if multiple metrics enabled
    if use_multiple_metrics and len(rmse_scores) > 3:
        # Use trimmed mean (remove outliers)
        sorted_rmse = np.sort(rmse_scores)
        trimmed_rmse = sorted_rmse[1:-1]  # Remove min and max
        mean_rmse = np.mean(trimmed_rmse) if len(trimmed_rmse) > 0 else mean_rmse
    
    return mean_rmse


def grid_search_hyperparameters(data_path, ranking_path, param_grid, cv_folds=5, verbose=True):
    """
    Perform grid search for hyperparameter optimization
    
    Parameters:
    -----------
    data_path : str
        Path to data CSV file
    ranking_path : str
        Path to feature importance ranking CSV
    param_grid : dict
        Dictionary of hyperparameter names to lists of values
    cv_folds : int
        Number of cross-validation folds
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    dict : Best hyperparameters and results
    """
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION - GRID SEARCH")
    print("=" * 80)
    
    # Load data once
    data = pd.read_csv(data_path)
    print(f"\n[OK] Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"\n[INFO] Total hyperparameter combinations: {len(combinations)}")
    print(f"[INFO] Cross-validation folds: {cv_folds}")
    
    results = []
    
    for idx, combination in enumerate(combinations):
        params = dict(zip(param_names, combination))
        
        if verbose:
            print(f"\n[{idx + 1}/{len(combinations)}] Testing: {params}")
        
        # Create model with hyperparameters
        mle_model = OptimizedMLEHyperparameter(
            data_path=data_path,
            ranking_path=ranking_path,
            **params
        )
        
        # Evaluate using cross-validation
        try:
            mean_rmse = evaluate_hyperparameters(mle_model, data, cv_folds=cv_folds)
            
            result = params.copy()
            result['mean_rmse'] = float(mean_rmse)
            results.append(result)
            
            if verbose:
                print(f"  → Mean RMSE: {mean_rmse:.4f} grams")
        except Exception as e:
            print(f"[ERROR] Failed for {params}: {str(e)}")
            result = params.copy()
            result['mean_rmse'] = np.inf
            results.append(result)
    
    # Find best hyperparameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['mean_rmse'].idxmin()
    best_params = results_df.loc[best_idx].to_dict()
    best_rmse = best_params['mean_rmse']
    
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS")
    print("=" * 80)
    print(f"\nBest Hyperparameters:")
    for key in param_names:
        print(f"  {key}: {best_params[key]}")
    print(f"\nBest Mean RMSE: {best_rmse:.4f} grams")
    
    return {
        'best_hyperparameters': {k: best_params[k] for k in param_names},
        'best_rmse': best_rmse,
        'all_results': results_df.to_dict('records')
    }


def random_search_hyperparameters(data_path, ranking_path, param_distributions, n_iter=20, 
                                  cv_folds=5, random_state=42, verbose=True):
    """
    Perform random search for hyperparameter optimization
    
    Parameters:
    -----------
    data_path : str
        Path to data CSV file
    ranking_path : str
        Path to feature importance ranking CSV
    param_distributions : dict
        Dictionary of hyperparameter names to lists of values
    n_iter : int
        Number of random combinations to try
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    dict : Best hyperparameters and results
    """
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION - RANDOM SEARCH")
    print("=" * 80)
    
    # Load data once
    data = pd.read_csv(data_path)
    print(f"\n[OK] Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Set random seed
    np.random.seed(random_state)
    
    # Generate random combinations
    param_names = list(param_distributions.keys())
    param_values = list(param_distributions.values())
    
    print(f"\n[INFO] Random search iterations: {n_iter}")
    print(f"[INFO] Cross-validation folds: {cv_folds}")
    
    results = []
    
    for idx in range(n_iter):
        # Randomly select one value for each parameter
        combination = []
        for values in param_values:
            if isinstance(values, list):
                # Handle boolean lists specially
                if all(isinstance(v, bool) for v in values):
                    choice = np.random.choice(values)
                else:
                    choice = np.random.choice(values)
                combination.append(choice)
            else:
                combination.append(values)
        params = dict(zip(param_names, combination))
        
        if verbose:
            print(f"\n[{idx + 1}/{n_iter}] Testing: {params}")
        
        # Create model with hyperparameters
        mle_model = OptimizedMLEHyperparameter(
            data_path=data_path,
            ranking_path=ranking_path,
            **params
        )
        
        # Evaluate using cross-validation
        try:
            mean_rmse = evaluate_hyperparameters(mle_model, data, cv_folds=cv_folds)
            
            result = params.copy()
            result['mean_rmse'] = float(mean_rmse)
            results.append(result)
            
            if verbose:
                print(f"  → Mean RMSE: {mean_rmse:.4f} grams")
        except Exception as e:
            print(f"[ERROR] Failed for {params}: {str(e)}")
            result = params.copy()
            result['mean_rmse'] = np.inf
            results.append(result)
    
    # Find best hyperparameters
    results_df = pd.DataFrame(results)
    best_idx = results_df['mean_rmse'].idxmin()
    best_params = results_df.loc[best_idx].to_dict()
    best_rmse = best_params['mean_rmse']
    
    print("\n" + "=" * 80)
    print("RANDOM SEARCH RESULTS")
    print("=" * 80)
    print(f"\nBest Hyperparameters:")
    for key in param_names:
        print(f"  {key}: {best_params[key]}")
    print(f"\nBest Mean RMSE: {best_rmse:.4f} grams")
    
    return {
        'best_hyperparameters': {k: best_params[k] for k in param_names},
        'best_rmse': best_rmse,
        'all_results': results_df.to_dict('records')
    }


def bayesian_optimization_hyperparameters(data_path, ranking_path, n_calls=50, 
                                         cv_folds=7, random_state=42, verbose=True):
    """
    Perform Bayesian optimization for hyperparameter tuning using Gaussian Process
    
    Parameters:
    -----------
    data_path : str
        Path to data CSV file
    ranking_path : str
        Path to feature importance ranking CSV
    n_calls : int
        Number of optimization iterations
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random state for reproducibility
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    dict : Best hyperparameters and results
    """
    if not SKOPT_AVAILABLE:
        print("[ERROR] scikit-optimize not available. Falling back to random search.")
        return None
    
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION - BAYESIAN OPTIMIZATION")
    print("=" * 80)
    
    # Load data once
    data = pd.read_csv(data_path)
    print(f"\n[OK] Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Define search space for Bayesian optimization
    space = [
        Integer(150, 300, name='max_iter'),
        Real(1e-6, 1e-4, prior='log-uniform', name='tol'),
        Real(1e-7, 1e-4, prior='log-uniform', name='cov_reg'),
        Real(1e-7, 1e-4, prior='log-uniform', name='cov_reg_conditional'),
        Categorical([True, False], name='use_shrinkage'),
        Real(0.05, 0.3, name='shrinkage_alpha'),
        Categorical(['mle', 'ridge', 'lasso', 'elastic', 'bayesian'], name='prediction_method'),
        Real(0.1, 10.0, prior='log-uniform', name='ridge_alpha'),
        Categorical([True, False], name='use_feature_scaling'),
        Categorical(['nanmean', 'complete_case', 'median', 'knn'], name='em_init_method'),
        Integer(3, 7, name='knn_neighbors'),
        Real(0.3, 0.7, name='elastic_l1_ratio'),
        Integer(5, 15, name='early_stopping_patience'),
        Categorical([True, False], name='use_ensemble'),
        Categorical([True, False], name='use_robust_scaling')
    ]
    
    def objective(params):
        """Objective function for Bayesian optimization"""
        # Convert params to dict
        param_dict = {
            'max_iter': int(params[0]),
            'tol': float(params[1]),
            'cov_reg': float(params[2]),
            'cov_reg_conditional': float(params[3]),
            'use_shrinkage': bool(params[4]),
            'shrinkage_alpha': float(params[5]),
            'prediction_method': str(params[6]),
            'ridge_alpha': float(params[7]),
            'use_feature_scaling': bool(params[8]),
            'em_init_method': str(params[9]),
            'knn_neighbors': int(params[10]),
            'elastic_l1_ratio': float(params[11]),
            'early_stopping_patience': int(params[12]),
            'use_ensemble': bool(params[13]),
            'use_robust_scaling': bool(params[14])
        }
        
        try:
            # Create model with hyperparameters
            mle_model = OptimizedMLEHyperparameter(
                data_path=data_path,
                ranking_path=ranking_path,
                **param_dict
            )
            
            # Evaluate using cross-validation
            mean_rmse = evaluate_hyperparameters(mle_model, data, cv_folds=cv_folds, 
                                                 random_state=random_state)
            
            if verbose:
                print(f"  → Mean RMSE: {mean_rmse:.4f} grams")
            
            return mean_rmse
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed: {str(e)}")
            return np.inf
    
    print(f"\n[INFO] Bayesian optimization iterations: {n_calls}")
    print(f"[INFO] Cross-validation folds: {cv_folds}")
    
    # Run Bayesian optimization
    result = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        random_state=random_state,
        verbose=verbose
    )
    
    # Extract best parameters
    best_params = {
        'max_iter': int(result.x[0]),
        'tol': float(result.x[1]),
        'cov_reg': float(result.x[2]),
        'cov_reg_conditional': float(result.x[3]),
        'use_shrinkage': bool(result.x[4]),
        'shrinkage_alpha': float(result.x[5]),
        'prediction_method': str(result.x[6]),
        'ridge_alpha': float(result.x[7]),
        'use_feature_scaling': bool(result.x[8]),
        'em_init_method': str(result.x[9]),
        'knn_neighbors': int(result.x[10]),
        'elastic_l1_ratio': float(result.x[11]),
        'early_stopping_patience': int(result.x[12]),
        'use_ensemble': bool(result.x[13]),
        'use_robust_scaling': bool(result.x[14])
    }
    
    best_rmse = float(result.fun)
    
    print("\n" + "=" * 80)
    print("BAYESIAN OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nBest Mean RMSE: {best_rmse:.4f} grams")
    
    return {
        'best_hyperparameters': best_params,
        'best_rmse': best_rmse,
        'optimization_result': result
    }


def main():
    """Main function for hyperparameter optimization"""
    print("=" * 80)
    print("MLE TOP 25 - HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Define data paths
    data_path = 'Data/processed/cleaned_dataset_with_engineered_features.csv'
    ranking_path = 'Data/processed/MLE_New/feature_importance_ranking.csv'
    
    # Define enhanced hyperparameter search space (including advanced optimizations)
    param_grid = {
        'max_iter': [150, 200, 250, 300],
        'tol': [1e-6, 1e-5, 1e-4],
        'cov_reg': [1e-7, 1e-6, 1e-5, 1e-4],
        'cov_reg_conditional': [1e-7, 1e-6, 1e-5, 1e-4],
        'use_shrinkage': [True, False],
        'shrinkage_alpha': [0.05, 0.1, 0.15, 0.2, 0.3],
        'prediction_method': ['mle', 'ridge', 'lasso', 'elastic', 'bayesian'],
        'ridge_alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
        'use_feature_scaling': [True, False],
        'em_init_method': ['nanmean', 'complete_case', 'median', 'knn'],
        'knn_neighbors': [3, 5, 7],
        'elastic_l1_ratio': [0.3, 0.5, 0.7],
        'early_stopping_patience': [5, 10, 15],
        'use_ensemble': [True, False],
        'use_robust_scaling': [True, False]
    }
    
    print("\nHyperparameter Search Space:")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")
    
    # Calculate total combinations for grid search (for info)
    total_combinations = 1
    for values in param_grid.values():
        total_combinations *= len(values)
    print(f"\n[INFO] Total possible combinations (grid search): {total_combinations}")
    print(f"[INFO] This would be computationally expensive - using random search instead")
    
    # Choose search strategy
    # Options: 'random', 'grid', 'bayesian'
    search_strategy = 'random'  # Options: 'random', 'grid', 'bayesian'
    
    if search_strategy == 'bayesian':
        print("\n[INFO] Using Bayesian Optimization (Gaussian Process)")
        results = bayesian_optimization_hyperparameters(
            data_path=data_path,
            ranking_path=ranking_path,
            n_calls=50,  # Number of optimization iterations
            cv_folds=7,
            random_state=42,
            verbose=True
        )
    elif search_strategy == 'random':
        print("\n[INFO] Using Random Search (recommended for large parameter space)")
        results = random_search_hyperparameters(
            data_path=data_path,
            ranking_path=ranking_path,
            param_distributions=param_grid,
            n_iter=100,  # Increased for better coverage
            cv_folds=7,  # Increased CV folds for more stable estimates
            random_state=42,
            verbose=True
        )
    else:  # grid search
        print("\n[INFO] Using Grid Search (exhaustive)")
        results = grid_search_hyperparameters(
            data_path=data_path,
            ranking_path=ranking_path,
            param_grid=param_grid,
            cv_folds=7,  # Increased CV folds
            verbose=True
        )
    
    # Save results
    output_dir = 'Data/processed/MLE_Top25'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save best hyperparameters
    best_hyperparams_file = os.path.join(output_dir, f'best_hyperparameters_{timestamp}.json')
    with open(best_hyperparams_file, 'w') as f:
        json.dump(results['best_hyperparameters'], f, indent=2)
    
    # Save all results
    all_results_df = pd.DataFrame(results['all_results'])
    all_results_file = os.path.join(output_dir, f'hyperparameter_search_results_{timestamp}.csv')
    all_results_df.to_csv(all_results_file, index=False)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  - Best hyperparameters: {best_hyperparams_file}")
    print(f"  - All results: {all_results_file}")
    
    # Train final model with best hyperparameters
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 80)
    
    best_model = OptimizedMLEHyperparameter(
        data_path=data_path,
        ranking_path=ranking_path,
        **results['best_hyperparameters']
    )
    
    # Run complete analysis
    best_model.load_and_analyze_data()
    best_model.select_probability_models()
    best_model.implement_likelihood_functions()
    best_model.implement_optimization_methods()
    best_model.implement_em_algorithm()
    final_results = best_model.run_complete_mle_analysis()
    
    # Save final model results with hyperparameters
    final_results_file = os.path.join(output_dir, f'mle_top25_optimized_results_{timestamp}.json')
    final_results_summary = {
        'hyperparameters': results['best_hyperparameters'],
        'cv_mean_rmse': results['best_rmse'],
        'final_test_metrics': best_model.metrics,
        'mle_results': {
            'converged': bool(final_results['converged']),
            'iterations': int(final_results['iterations']),
            'final_likelihood': float(final_results['likelihood'])
        }
    }
    
    with open(final_results_file, 'w') as f:
        json.dump(final_results_summary, f, indent=2)
    
    print(f"\n[OK] Final optimized model results saved to: {final_results_file}")
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBest Hyperparameters:")
    for key, value in results['best_hyperparameters'].items():
        print(f"  {key}: {value}")
    print(f"\nCross-Validation RMSE: {results['best_rmse']:.4f} grams")
    print(f"\nFinal Test Metrics:")
    print(f"  RMSE: {best_model.metrics['RMSE']:.4f} grams")
    print(f"  MAE: {best_model.metrics['MAE']:.4f} grams")
    print(f"  R²: {best_model.metrics['R²']:.4f}")
    print(f"  Correlation: {best_model.metrics['Correlation']:.4f}")
    print("=" * 80)
    
    return results, best_model


if __name__ == "__main__":
    results, model = main()

