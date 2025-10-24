"""
Phase 4: Model Optimization and Validation
==========================================

This script implements:
1. Hyperparameter tuning for XGBoost and Random Forest
2. Cross-validation for robust model validation
3. Feature selection to identify most important features
4. Model comparison and selection

Expected improvements: 10-20% additional RMSE reduction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from xgboost import XGBRegressor
import joblib
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/phase4_optimization_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def load_data():
    """Load the engineered dataset from Phase 2"""
    try:
        data = pd.read_csv('Data/processed/MLE_Improved/phase2_engineered_dataset.csv')
        logging.info(f"Loaded dataset with shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def prepare_ml_data(data):
    """Prepare data for machine learning models"""
    # Load feature names from Phase 2
    with open('Data/processed/MLE_Improved/phase2_feature_details.json', 'r') as f:
        feature_details = json.load(f)
    
    # Get final features (excluding target and ID columns)
    target_col = 'f1_bw'
    id_cols = ['f0_id', 'f1_id'] if 'f0_id' in data.columns else ['f1_id']
    
    feature_cols = [col for col in data.columns if col not in [target_col] + id_cols]
    
    # Prepare feature matrix and target
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    # Handle infinite values and missing values
    # Replace infinite values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Check for columns with all NaN values and remove them
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        logging.info(f"Removing {len(all_nan_cols)} columns with all NaN values: {all_nan_cols[:5]}...")
        X = X.drop(columns=all_nan_cols)
        feature_cols = [col for col in feature_cols if col not in all_nan_cols]
    
    # Handle missing values with iterative imputation
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    imputer = IterativeImputer(random_state=42, max_iter=10)
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=feature_cols)
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    logging.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler

def hyperparameter_tuning_xgboost(X_train, y_train, X_val, y_val):
    """Perform hyperparameter tuning for XGBoost"""
    logging.info("Starting XGBoost hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 6, 9, 12],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    }
    
    # Use RandomizedSearchCV for efficiency
    xgb_model = XGBRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = random_search.best_params_
    best_score = -random_search.best_score_
    
    logging.info(f"Best XGBoost parameters: {best_params}")
    logging.info(f"Best CV RMSE: {np.sqrt(best_score):.2f}")
    
    # Evaluate on validation set
    best_xgb = random_search.best_estimator_
    y_pred_val = best_xgb.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)
    
    logging.info(f"XGBoost Validation - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
    
    return best_xgb, best_params, val_rmse

def hyperparameter_tuning_random_forest(X_train, y_train, X_val, y_val):
    """Perform hyperparameter tuning for Random Forest"""
    logging.info("Starting Random Forest hyperparameter tuning...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9],
        'bootstrap': [True, False]
    }
    
    # Use RandomizedSearchCV for efficiency
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    # Fit the model
    random_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = random_search.best_params_
    best_score = -random_search.best_score_
    
    logging.info(f"Best Random Forest parameters: {best_params}")
    logging.info(f"Best CV RMSE: {np.sqrt(best_score):.2f}")
    
    # Evaluate on validation set
    best_rf = random_search.best_estimator_
    y_pred_val = best_rf.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)
    
    logging.info(f"Random Forest Validation - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
    
    return best_rf, best_params, val_rmse

def cross_validation_analysis(X_train, y_train, models):
    """Perform cross-validation analysis for all models"""
    logging.info("Starting cross-validation analysis...")
    
    cv_results = {}
    
    for name, model in models.items():
        logging.info(f"Performing CV for {name}...")
        
        # 5-fold cross-validation
        scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        rmse_scores = np.sqrt(-scores)
        
        cv_results[name] = {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'scores': rmse_scores.tolist()
        }
        
        logging.info(f"{name} - CV RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")
    
    return cv_results

def feature_selection_analysis(X_train, y_train, X_val, y_val, feature_cols, k_values=[10, 15, 20, 25, 30]):
    """Perform feature selection analysis"""
    logging.info("Starting feature selection analysis...")
    
    feature_selection_results = {}
    
    for k in k_values:
        logging.info(f"Selecting top {k} features...")
        
        # SelectKBest with f_regression
        selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        
        # Get selected feature names
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Train XGBoost with selected features
        xgb_selected = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_selected.fit(X_train_selected, y_train)
        
        # Evaluate
        y_pred_val = xgb_selected.predict(X_val_selected)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
        val_mae = mean_absolute_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)
        
        feature_selection_results[k] = {
            'selected_features': selected_features,
            'val_rmse': val_rmse,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'feature_importance': xgb_selected.feature_importances_.tolist()
        }
        
        logging.info(f"Top {k} features - Val RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, R²: {val_r2:.4f}")
    
    # Find optimal number of features
    best_k = min(feature_selection_results.keys(), 
                key=lambda x: feature_selection_results[x]['val_rmse'])
    
    logging.info(f"Optimal number of features: {best_k}")
    
    return feature_selection_results, best_k

def create_optimization_visualizations(cv_results, feature_selection_results, best_k):
    """Create visualizations for optimization results"""
    logging.info("Creating optimization visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cross-validation results
    model_names = list(cv_results.keys())
    mean_rmse = [cv_results[name]['mean_rmse'] for name in model_names]
    std_rmse = [cv_results[name]['std_rmse'] for name in model_names]
    
    bars = axes[0, 0].bar(model_names, mean_rmse, yerr=std_rmse, capsize=5, alpha=0.7)
    axes[0, 0].set_ylabel('RMSE (grams)')
    axes[0, 0].set_title('Cross-Validation RMSE Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, mean, std in zip(bars, mean_rmse, std_rmse):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + std,
                       f'{mean:.1f}±{std:.1f}', ha='center', va='bottom')
    
    # 2. Feature selection results
    k_values = list(feature_selection_results.keys())
    rmse_values = [feature_selection_results[k]['val_rmse'] for k in k_values]
    
    axes[0, 1].plot(k_values, rmse_values, marker='o', linewidth=2, markersize=8)
    axes[0, 1].axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={best_k}')
    axes[0, 1].set_xlabel('Number of Features')
    axes[0, 1].set_ylabel('Validation RMSE (grams)')
    axes[0, 1].set_title('Feature Selection Analysis')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Add value labels
    for k, rmse in zip(k_values, rmse_values):
        axes[0, 1].annotate(f'{rmse:.1f}', (k, rmse), textcoords="offset points", 
                           xytext=(0,10), ha='center')
    
    # 3. Feature importance (top 15 from best model)
    best_features = feature_selection_results[best_k]['selected_features']
    best_importance = feature_selection_results[best_k]['feature_importance']
    
    # Create DataFrame for sorting
    importance_df = pd.DataFrame({
        'feature': best_features,
        'importance': best_importance
    }).sort_values('importance', ascending=True).tail(15)
    
    axes[1, 0].barh(range(len(importance_df)), importance_df['importance'])
    axes[1, 0].set_yticks(range(len(importance_df)))
    axes[1, 0].set_yticklabels(importance_df['feature'])
    axes[1, 0].set_xlabel('Feature Importance')
    axes[1, 0].set_title(f'Top 15 Feature Importance (k={best_k})')
    
    # 4. Model comparison summary
    # Load previous phase results for comparison
    try:
        with open('Data/processed/MLE_Improved/phase3_summary.json', 'r') as f:
            phase3_results = json.load(f)
        
        # Compare with previous phase
        phase3_rmse = phase3_results['model_performance']['XGBoost']['test_rmse']
        current_best_rmse = min(rmse_values)
        
        comparison_data = ['Phase 3 XGBoost', 'Phase 4 Optimized']
        comparison_rmse = [phase3_rmse, current_best_rmse]
        colors = ['lightblue', 'lightgreen']
        
        bars = axes[1, 1].bar(comparison_data, comparison_rmse, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('RMSE (grams)')
        axes[1, 1].set_title('Phase 3 vs Phase 4 Performance')
        
        # Add value labels
        for bar, value in zip(bars, comparison_rmse):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # Add improvement percentage
        improvement = ((phase3_rmse - current_best_rmse) / phase3_rmse) * 100
        axes[1, 1].text(0.5, max(comparison_rmse) * 0.8, 
                       f'Improvement: {improvement:.1f}%', 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
    except FileNotFoundError:
        axes[1, 1].text(0.5, 0.5, 'Phase 3 results not found', ha='center', va='center')
        axes[1, 1].set_title('Phase Comparison (Data Not Available)')
    
    plt.tight_layout()
    plt.savefig('PLOTS/MLE_Improved/phase4_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Optimization visualizations saved to PLOTS/MLE_Improved/phase4_optimization_analysis.png")

def save_optimization_results(best_xgb, best_rf, cv_results, feature_selection_results, best_k, scaler):
    """Save optimization results and models"""
    logging.info("Saving optimization results...")
    
    # Save optimized models
    joblib.dump(best_xgb, 'Models/optimized_xgboost_model.pkl')
    joblib.dump(best_rf, 'Models/optimized_random_forest_model.pkl')
    joblib.dump(scaler, 'Models/optimization_scaler.pkl')
    
    # Save feature selection results
    best_features = feature_selection_results[best_k]['selected_features']
    with open('Models/optimized_feature_names.json', 'w') as f:
        json.dump(best_features, f, indent=2)
    
    # Create comprehensive results summary
    results = {
        'phase': 'Phase 4: Model Optimization and Validation',
        'optimization_date': datetime.now().isoformat(),
        'cross_validation_results': cv_results,
        'feature_selection_results': {
            'optimal_k': best_k,
            'selected_features': best_features,
            'performance_by_k': {str(k): {
                'val_rmse': feature_selection_results[k]['val_rmse'],
                'val_mae': feature_selection_results[k]['val_mae'],
                'val_r2': feature_selection_results[k]['val_r2']
            } for k in feature_selection_results.keys()}
        },
        'best_models': {
            'xgboost': {
                'model_file': 'Models/optimized_xgboost_model.pkl',
                'cv_rmse': cv_results.get('XGBoost', {}).get('mean_rmse', 'N/A')
            },
            'random_forest': {
                'model_file': 'Models/optimized_random_forest_model.pkl',
                'cv_rmse': cv_results.get('Random Forest', {}).get('mean_rmse', 'N/A')
            }
        }
    }
    
    # Save results
    with open('Data/processed/MLE_Improved/phase4_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info("Optimization results saved successfully!")

def main():
    """Main function to run Phase 4 optimization"""
    logging.info("Starting Phase 4: Model Optimization and Validation")
    
    try:
        # Load data
        data = load_data()
        
        # Prepare ML data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, scaler = prepare_ml_data(data)
        
        # Hyperparameter tuning
        best_xgb, xgb_params, xgb_val_rmse = hyperparameter_tuning_xgboost(X_train, y_train, X_val, y_val)
        best_rf, rf_params, rf_val_rmse = hyperparameter_tuning_random_forest(X_train, y_train, X_val, y_val)
        
        # Cross-validation analysis
        models = {
            'XGBoost': best_xgb,
            'Random Forest': best_rf
        }
        cv_results = cross_validation_analysis(X_train, y_train, models)
        
        # Feature selection analysis
        feature_selection_results, best_k = feature_selection_analysis(
            X_train, y_train, X_val, y_val, feature_cols
        )
        
        # Create visualizations
        create_optimization_visualizations(cv_results, feature_selection_results, best_k)
        
        # Save results
        save_optimization_results(best_xgb, best_rf, cv_results, feature_selection_results, best_k, scaler)
        
        # Final summary
        best_model_name = min(cv_results.keys(), key=lambda x: cv_results[x]['mean_rmse'])
        best_cv_rmse = cv_results[best_model_name]['mean_rmse']
        
        logging.info("="*60)
        logging.info("PHASE 4 OPTIMIZATION COMPLETED SUCCESSFULLY")
        logging.info("="*60)
        logging.info(f"Best model: {best_model_name}")
        logging.info(f"Best CV RMSE: {best_cv_rmse:.2f} grams")
        logging.info(f"Optimal features: {best_k}")
        logging.info(f"Selected features: {len(feature_selection_results[best_k]['selected_features'])}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Error in Phase 4 optimization: {e}")
        raise

if __name__ == "__main__":
    main()
