

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import xgboost as xgb
import joblib
import json
import warnings
import logging
import time
from datetime import datetime
from tqdm import tqdm
import sys
warnings.filterwarnings('ignore')

# Configure logging
def setup_logging():
    """Setup comprehensive logging system"""
    # Create logs directory if it doesn't exist
    import os
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('phase3_ml_models')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(f'logs/phase3_ml_models_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_logging()

class CustomEnsemble:
    """Custom ensemble model that handles different scaling for different models"""
    def __init__(self, rf_model, xgb_model, nn_model, nn_scaler):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.nn_model = nn_model
        self.nn_scaler = nn_scaler
    
    def predict(self, X):
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        X_nn = self.nn_scaler.transform(X)
        nn_pred = self.nn_model.predict(X_nn)
        
        # Simple average ensemble
        return (rf_pred + xgb_pred + nn_pred) / 3

class ProgressTracker:
    """Track progress and timing for different phases"""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = time.time()
        self.step_times = []
        self.pbar = tqdm(total=total_steps, desc=description, 
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    def update(self, step_description="", increment=1):
        """Update progress with optional description"""
        self.current_step += increment
        self.step_times.append(time.time())
        
        if step_description:
            self.pbar.set_description(f"{self.description}: {step_description}")
        
        self.pbar.update(increment)
        
        # Log progress
        elapsed = time.time() - self.start_time
        logger.info(f"Progress: {self.current_step}/{self.total_steps} - {step_description} (Elapsed: {elapsed:.1f}s)")
    
    def finish(self, final_description="Completed"):
        """Finish progress tracking"""
        self.pbar.set_description(f"{self.description}: {final_description}")
        self.pbar.close()
        
        total_time = time.time() - self.start_time
        logger.info(f"{self.description} completed in {total_time:.2f} seconds")
        
        return total_time

def log_phase_start(phase_name, description=""):
    """Log the start of a new phase"""
    logger.info("="*60)
    logger.info(f"STARTING: {phase_name}")
    if description:
        logger.info(f"Description: {description}")
    logger.info("="*60)

def log_phase_end(phase_name, duration, results_summary=""):
    """Log the end of a phase"""
    logger.info("="*60)
    logger.info(f"COMPLETED: {phase_name}")
    logger.info(f"Duration: {duration:.2f} seconds")
    if results_summary:
        logger.info(f"Results: {results_summary}")
    logger.info("="*60)

def load_phase2_results():
    """Load Phase 2 results and engineered dataset"""
    log_phase_start("Phase 3: Advanced Machine Learning Models", 
                   "Implementing Random Forest, XGBoost, Neural Network, and Ensemble models")
    
    logger.info("Loading Phase 2 engineered dataset...")
    start_time = time.time()
    
    # Load engineered dataset
    data = pd.read_csv('Data/processed/MLE_Improved/phase2_engineered_dataset.csv')
    logger.info(f"Engineered dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Load Phase 2 summary
    with open('Data/processed/MLE_Improved/phase2_summary.json', 'r') as f:
        phase2_summary = json.load(f)
    
    logger.info(f"Original features: {phase2_summary['original_features']}")
    logger.info(f"Engineered features: {phase2_summary['engineered_features']}")
    logger.info(f"Total features: {phase2_summary['total_features']}")
    
    duration = time.time() - start_time
    logger.info(f"Data loading completed in {duration:.2f} seconds")
    
    return data, phase2_summary

def prepare_data_for_ml(data):
    """Step 3.1: Prepare data for machine learning models"""
    log_phase_start("Step 3.1: Data Preparation", "Preparing data for ML models")
    
    # Initialize progress tracker
    progress = ProgressTracker(6, "Data Preparation")
    
    # Define target variable
    progress.update("Defining target variable and features")
    target_var = 'f1_bw'
    if target_var not in data.columns:
        raise ValueError(f"Target variable {target_var} not found in dataset")
    
    # Select features (exclude target and non-predictive columns)
    exclude_cols = [target_var, 'Unnamed: 0', 'row_index', 'row_index.1', 'LBW_flag']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    logger.info(f"Target variable: {target_var}")
    logger.info(f"Feature columns: {len(feature_cols)}")
    
    # Prepare feature matrix and target vector
    progress.update("Creating feature matrix and target vector")
    X = data[feature_cols].copy()
    y = data[target_var].copy()
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    
    # Handle missing data and infinite values
    progress.update("Handling missing data and infinite values")
    missing_before = X.isnull().sum().sum()
    inf_before = np.isinf(X).sum().sum()
    logger.info(f"Missing values before cleaning: {missing_before}")
    logger.info(f"Infinite values before cleaning: {inf_before}")
    
    # Replace infinite values with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    if missing_before > 0 or inf_before > 0:
        logger.info("Applying iterative imputation...")
        imputer = IterativeImputer(random_state=42, max_iter=10)
        X_imputed = imputer.fit_transform(X)
        X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        
        missing_after = X.isnull().sum().sum()
        inf_after = np.isinf(X).sum().sum()
        logger.info(f"Missing values after imputation: {missing_after}")
        logger.info(f"Infinite values after imputation: {inf_after}")
    else:
        logger.info("No missing or infinite values found")
    
    # Split data into train/validation/test sets
    progress.update("Splitting data into train/validation/test sets")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=None
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=None
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features for models that need it
    progress.update("Scaling features for ML models")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    # Finish progress tracking
    duration = progress.finish("Data preparation completed successfully!")
    log_phase_end("Step 3.1: Data Preparation", duration, 
                 f"Prepared {len(feature_cols)} features for {X_train.shape[0]} training samples")
    
    return (X_train, X_val, X_test, y_train, y_val, y_test, 
            X_train_scaled, X_val_scaled, X_test_scaled, scaler, feature_cols)

def calculate_metrics(y_true, y_pred, set_name):
    """Calculate comprehensive metrics for model evaluation"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    
    print(f"{set_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%, Corr: {correlation:.4f}")
    
    return {
        'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape, 'correlation': correlation
    }

def implement_random_forest(X_train, X_val, X_test, y_train, y_val, y_test):
    """Step 3.2: Implement Random Forest model"""
    log_phase_start("Step 3.2: Random Forest Model", "Training Random Forest with 100 estimators")
    
    # Initialize progress tracker
    progress = ProgressTracker(4, "Random Forest Training")
    
    # Random Forest model
    progress.update("Initializing Random Forest model")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("Random Forest parameters:")
    logger.info(f"  - n_estimators: {rf_model.n_estimators}")
    logger.info(f"  - max_depth: {rf_model.max_depth}")
    logger.info(f"  - min_samples_split: {rf_model.min_samples_split}")
    logger.info(f"  - min_samples_leaf: {rf_model.min_samples_leaf}")
    
    # Training with progress tracking
    progress.update("Training Random Forest model (this may take a while...)")
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"Random Forest training completed in {training_time:.2f} seconds")
    
    # Make predictions
    progress.update("Making predictions on train/validation/test sets")
    y_pred_train = rf_model.predict(X_train)
    y_pred_val = rf_model.predict(X_val)
    y_pred_test = rf_model.predict(X_test)
    
    # Calculate metrics
    progress.update("Calculating performance metrics")
    train_metrics = calculate_metrics(y_train, y_pred_train, "Random Forest Training")
    val_metrics = calculate_metrics(y_val, y_pred_val, "Random Forest Validation")
    test_metrics = calculate_metrics(y_test, y_pred_test, "Random Forest Test")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 Most Important Features (Random Forest):")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        logger.info(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    # Finish progress tracking
    duration = progress.finish("Random Forest training completed!")
    log_phase_end("Step 3.2: Random Forest Model", duration, 
                 f"Test RMSE: {test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.4f}")
    
    return rf_model, train_metrics, val_metrics, test_metrics, feature_importance

def implement_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    """Step 3.3: Implement XGBoost model"""
    log_phase_start("Step 3.3: XGBoost Model", "Training XGBoost with gradient boosting")
    
    # Initialize progress tracker
    progress = ProgressTracker(4, "XGBoost Training")
    
    # XGBoost model
    progress.update("Initializing XGBoost model")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    logger.info("XGBoost parameters:")
    logger.info(f"  - n_estimators: {xgb_model.n_estimators}")
    logger.info(f"  - max_depth: {xgb_model.max_depth}")
    logger.info(f"  - learning_rate: {xgb_model.learning_rate}")
    logger.info(f"  - subsample: {xgb_model.subsample}")
    logger.info(f"  - colsample_bytree: {xgb_model.colsample_bytree}")
    
    # Training with progress tracking
    progress.update("Training XGBoost model (gradient boosting in progress...)")
    start_time = time.time()
    xgb_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"XGBoost training completed in {training_time:.2f} seconds")
    
    # Make predictions
    progress.update("Making predictions on train/validation/test sets")
    y_pred_train = xgb_model.predict(X_train)
    y_pred_val = xgb_model.predict(X_val)
    y_pred_test = xgb_model.predict(X_test)
    
    # Calculate metrics
    progress.update("Calculating performance metrics")
    train_metrics = calculate_metrics(y_train, y_pred_train, "XGBoost Training")
    val_metrics = calculate_metrics(y_val, y_pred_val, "XGBoost Validation")
    test_metrics = calculate_metrics(y_test, y_pred_test, "XGBoost Test")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("Top 10 Most Important Features (XGBoost):")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
        logger.info(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
    
    # Finish progress tracking
    duration = progress.finish("XGBoost training completed!")
    log_phase_end("Step 3.3: XGBoost Model", duration, 
                 f"Test RMSE: {test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.4f}")
    
    return xgb_model, train_metrics, val_metrics, test_metrics, feature_importance

def implement_neural_network(X_train, X_val, X_test, y_train, y_val, y_test):
    """Step 3.4: Implement Neural Network model"""
    log_phase_start("Step 3.4: Neural Network Model", "Training Multi-layer Perceptron with 3 hidden layers")
    
    # Initialize progress tracker
    progress = ProgressTracker(5, "Neural Network Training")
    
    # Scale data for neural network (0-1 range)
    progress.update("Scaling data for neural network (MinMax scaling)")
    nn_scaler = MinMaxScaler()
    X_train_nn = nn_scaler.fit_transform(X_train)
    X_val_nn = nn_scaler.transform(X_val)
    X_test_nn = nn_scaler.transform(X_test)
    logger.info("Data scaled to [0,1] range for neural network")
    
    # Neural Network model
    progress.update("Initializing Neural Network model")
    nn_model = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    
    logger.info("Neural Network architecture:")
    logger.info(f"  - Hidden layers: {nn_model.hidden_layer_sizes}")
    logger.info(f"  - Activation: {nn_model.activation}")
    logger.info(f"  - Solver: {nn_model.solver}")
    logger.info(f"  - Alpha (L2 regularization): {nn_model.alpha}")
    logger.info(f"  - Max iterations: {nn_model.max_iter}")
    
    # Training with progress tracking
    progress.update("Training Neural Network model (backpropagation in progress...)")
    start_time = time.time()
    nn_model.fit(X_train_nn, y_train)
    training_time = time.time() - start_time
    logger.info(f"Neural Network training completed in {training_time:.2f} seconds")
    logger.info(f"Final loss: {nn_model.loss_:.6f}")
    logger.info(f"Number of iterations: {nn_model.n_iter_}")
    
    # Make predictions
    progress.update("Making predictions on train/validation/test sets")
    y_pred_train = nn_model.predict(X_train_nn)
    y_pred_val = nn_model.predict(X_val_nn)
    y_pred_test = nn_model.predict(X_test_nn)
    
    # Calculate metrics
    progress.update("Calculating performance metrics")
    train_metrics = calculate_metrics(y_train, y_pred_train, "Neural Network Training")
    val_metrics = calculate_metrics(y_val, y_pred_val, "Neural Network Validation")
    test_metrics = calculate_metrics(y_test, y_pred_test, "Neural Network Test")
    
    # Finish progress tracking
    duration = progress.finish("Neural Network training completed!")
    log_phase_end("Step 3.4: Neural Network Model", duration, 
                 f"Test RMSE: {test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.4f}")
    
    return nn_model, train_metrics, val_metrics, test_metrics, nn_scaler

def implement_ensemble_model(rf_model, xgb_model, nn_model, nn_scaler, X_train, X_val, X_test, y_train, y_val, y_test):
    """Step 3.5: Implement Ensemble model"""
    log_phase_start("Step 3.5: Ensemble Model", "Creating ensemble from Random Forest, XGBoost, and Neural Network")
    
    # Initialize progress tracker
    progress = ProgressTracker(4, "Ensemble Training")
    
    # Create ensemble model
    progress.update("Creating custom ensemble model")
    # For neural network, we need to use scaled data
    X_train_nn = nn_scaler.transform(X_train)
    X_val_nn = nn_scaler.transform(X_val)
    X_test_nn = nn_scaler.transform(X_test)
    
    # Create a custom ensemble that handles different scaling
    custom_ensemble = CustomEnsemble(rf_model, xgb_model, nn_model, nn_scaler)
    logger.info("Custom ensemble created with equal weights for all three models")
    logger.info("Ensemble components: Random Forest + XGBoost + Neural Network")
    
    # Make predictions
    progress.update("Making ensemble predictions on all datasets")
    y_pred_train = custom_ensemble.predict(X_train)
    y_pred_val = custom_ensemble.predict(X_val)
    y_pred_test = custom_ensemble.predict(X_test)
    
    # Calculate metrics
    progress.update("Calculating ensemble performance metrics")
    train_metrics = calculate_metrics(y_train, y_pred_train, "Ensemble Training")
    val_metrics = calculate_metrics(y_val, y_pred_val, "Ensemble Validation")
    test_metrics = calculate_metrics(y_test, y_pred_test, "Ensemble Test")
    
    # Finish progress tracking
    duration = progress.finish("Ensemble model completed!")
    log_phase_end("Step 3.5: Ensemble Model", duration, 
                 f"Test RMSE: {test_metrics['rmse']:.2f}, R²: {test_metrics['r2']:.4f}")
    
    return custom_ensemble, train_metrics, val_metrics, test_metrics

def create_model_comparison_visualizations(models_results):
    """Create visualizations comparing all models"""
    log_phase_start("Model Comparison Visualizations", "Creating comprehensive model performance charts")
    
    logger.info("Creating model comparison visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract model names and metrics
    model_names = list(models_results.keys())
    rmse_values = [models_results[name]['test_metrics']['rmse'] for name in model_names]
    mae_values = [models_results[name]['test_metrics']['mae'] for name in model_names]
    r2_values = [models_results[name]['test_metrics']['r2'] for name in model_names]
    
    # 1. RMSE comparison
    bars1 = axes[0, 0].bar(model_names, rmse_values, color=['blue', 'green', 'orange', 'red'])
    axes[0, 0].set_title('RMSE Comparison (Test Set)')
    axes[0, 0].set_ylabel('RMSE (grams)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars1, rmse_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # 2. MAE comparison
    bars2 = axes[0, 1].bar(model_names, mae_values, color=['blue', 'green', 'orange', 'red'])
    axes[0, 1].set_title('MAE Comparison (Test Set)')
    axes[0, 1].set_ylabel('MAE (grams)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars2, mae_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # 3. R² comparison
    bars3 = axes[0, 2].bar(model_names, r2_values, color=['blue', 'green', 'orange', 'red'])
    axes[0, 2].set_title('R² Comparison (Test Set)')
    axes[0, 2].set_ylabel('R² Score')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars3, r2_values):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Training vs Test RMSE
    train_rmse = [models_results[name]['train_metrics']['rmse'] for name in model_names]
    test_rmse = [models_results[name]['test_metrics']['rmse'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, train_rmse, width, label='Training', alpha=0.8)
    axes[1, 0].bar(x + width/2, test_rmse, width, label='Test', alpha=0.8)
    axes[1, 0].set_title('Training vs Test RMSE')
    axes[1, 0].set_ylabel('RMSE (grams)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].legend()
    
    # 5. Model performance radar chart (simplified)
    metrics = ['RMSE', 'MAE', 'R²', 'MAPE', 'Correlation']
    best_model_idx = np.argmin(rmse_values)
    best_model = model_names[best_model_idx]
    
    # Normalize metrics for radar chart (lower is better for RMSE, MAE, MAPE)
    normalized_metrics = []
    for i, name in enumerate(model_names):
        model_metrics = models_results[name]['test_metrics']
        normalized = [
            1 - (model_metrics['rmse'] / max(rmse_values)),  # RMSE (inverted)
            1 - (model_metrics['mae'] / max(mae_values)),    # MAE (inverted)
            model_metrics['r2'],                             # R² (higher better)
            1 - (model_metrics['mape'] / 100),              # MAPE (inverted)
            model_metrics['correlation']                     # Correlation
        ]
        normalized_metrics.append(normalized)
    
    # Plot top 2 models
    for i, name in enumerate(model_names[:2]):
        axes[1, 1].plot(metrics, normalized_metrics[i], marker='o', label=name)
    
    axes[1, 1].set_title('Model Performance Comparison')
    axes[1, 1].set_ylabel('Normalized Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Improvement over baseline
    baseline_rmse = 388.42  # From original MLE
    improvements = [(baseline_rmse - rmse) / baseline_rmse * 100 for rmse in rmse_values]
    
    bars6 = axes[1, 2].bar(model_names, improvements, color=['blue', 'green', 'orange', 'red'])
    axes[1, 2].set_title('Improvement over Baseline MLE')
    axes[1, 2].set_ylabel('Improvement (%)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars6, improvements):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('PLOTS/MLE_Improved/phase3_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved to PLOTS/MLE_Improved/phase3_model_comparison.png")
    log_phase_end("Model Comparison Visualizations", 0, "6 comprehensive charts created")

def save_phase3_results(models_results, scaler, feature_cols):
    """Save Phase 3 results for next phases"""
    log_phase_start("Saving Phase 3 Results", "Saving models, scalers, and performance metrics")
    
    logger.info("Saving Phase 3 results...")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('Data/processed/MLE_Improved', exist_ok=True)
    os.makedirs('Models', exist_ok=True)
    
    # Save models
    for model_name, results in models_results.items():
        if 'model' in results:
            model_path = f'Models/{model_name.lower().replace(" ", "_")}_model.pkl'
            joblib.dump(results['model'], model_path)
            logger.info(f"Saved {model_name} model to {model_path}")
    
    # Save scaler
    joblib.dump(scaler, 'Models/feature_scaler.pkl')
    logger.info("Saved feature scaler to Models/feature_scaler.pkl")
    
    # Save feature names
    with open('Models/feature_names.json', 'w') as f:
        json.dump(feature_cols, f)
    logger.info("Saved feature names to Models/feature_names.json")
    
    # Save performance results
    performance_summary = {
        'phase': 'Phase 3: Advanced ML Models',
        'models_trained': len(models_results),
        'baseline_rmse': 388.42,
        'model_performance': {}
    }
    
    for model_name, results in models_results.items():
        performance_summary['model_performance'][model_name] = {
            'test_rmse': results['test_metrics']['rmse'],
            'test_mae': results['test_metrics']['mae'],
            'test_r2': results['test_metrics']['r2'],
            'improvement_percent': ((388.42 - results['test_metrics']['rmse']) / 388.42) * 100
        }
    
    with open('Data/processed/MLE_Improved/phase3_summary.json', 'w') as f:
        json.dump(performance_summary, f, indent=2)
    
    logger.info("Phase 3 results saved successfully!")
    log_phase_end("Saving Phase 3 Results", 0, f"Saved {len(models_results)} models and performance metrics")

def main():
    """Main function to run Phase 3"""
    # Initialize overall progress tracking
    overall_start_time = time.time()
    overall_progress = ProgressTracker(7, "Phase 3: Advanced ML Models")
    
    try:
        # Load Phase 2 results
        overall_progress.update("Loading Phase 2 results")
        data, phase2_summary = load_phase2_results()
        
        # Step 3.1: Prepare data for ML
        overall_progress.update("Preparing data for ML models")
        (X_train, X_val, X_test, y_train, y_val, y_test, 
         X_train_scaled, X_val_scaled, X_test_scaled, scaler, feature_cols) = prepare_data_for_ml(data)
        
        # Step 3.2: Implement Random Forest
        overall_progress.update("Training Random Forest model")
        rf_model, rf_train_metrics, rf_val_metrics, rf_test_metrics, rf_importance = implement_random_forest(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Step 3.3: Implement XGBoost
        overall_progress.update("Training XGBoost model")
        xgb_model, xgb_train_metrics, xgb_val_metrics, xgb_test_metrics, xgb_importance = implement_xgboost(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Step 3.4: Implement Neural Network
        overall_progress.update("Training Neural Network model")
        nn_model, nn_train_metrics, nn_val_metrics, nn_test_metrics, nn_scaler = implement_neural_network(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Step 3.5: Implement Ensemble
        overall_progress.update("Creating Ensemble model")
        ensemble_model, ensemble_train_metrics, ensemble_val_metrics, ensemble_test_metrics = implement_ensemble_model(
            rf_model, xgb_model, nn_model, nn_scaler, X_train, X_val, X_test, y_train, y_val, y_test
        )
        
        # Compile results
        overall_progress.update("Compiling model results")
        models_results = {
            'Random Forest': {
                'model': rf_model,
                'train_metrics': rf_train_metrics,
                'val_metrics': rf_val_metrics,
                'test_metrics': rf_test_metrics,
                'feature_importance': rf_importance
            },
            'XGBoost': {
                'model': xgb_model,
                'train_metrics': xgb_train_metrics,
                'val_metrics': xgb_val_metrics,
                'test_metrics': xgb_test_metrics,
                'feature_importance': xgb_importance
            },
            'Neural Network': {
                'model': nn_model,
                'train_metrics': nn_train_metrics,
                'val_metrics': nn_val_metrics,
                'test_metrics': nn_test_metrics,
                'nn_scaler': nn_scaler
            },
            'Ensemble': {
                'model': ensemble_model,
                'train_metrics': ensemble_train_metrics,
                'val_metrics': ensemble_val_metrics,
                'test_metrics': ensemble_test_metrics
            }
        }
        
        # Create visualizations
        overall_progress.update("Creating model comparison visualizations")
        create_model_comparison_visualizations(models_results)
        
        # Save results
        overall_progress.update("Saving Phase 3 results")
        save_phase3_results(models_results, scaler, feature_cols)
        
        # Find best model
        best_model_name = min(models_results.keys(), 
                             key=lambda x: models_results[x]['test_metrics']['rmse'])
        best_rmse = models_results[best_model_name]['test_metrics']['rmse']
        improvement = ((388.42 - best_rmse) / 388.42) * 100
        
        # Final progress update
        total_time = overall_progress.finish("Phase 3 completed successfully!")
        
        # Final summary
        logger.info("="*60)
        logger.info("PHASE 3 COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        logger.info(f"Models trained: {len(models_results)}")
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best RMSE: {best_rmse:.2f} grams")
        logger.info(f"Improvement over baseline: {improvement:.1f}%")
        logger.info("Ready for Phase 4: Model Optimization")
        logger.info("="*60)
        
        # Print final summary to console
        print("\n" + "="*60)
        print("PHASE 3 COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Models trained: {len(models_results)}")
        print(f"Best model: {best_model_name}")
        print(f"Best RMSE: {best_rmse:.2f} grams")
        print(f"Improvement: {improvement:.1f}%")
        print(f"Ready for Phase 4: Model Optimization")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error in Phase 3 execution: {str(e)}")
        logger.error("Phase 3 failed!")
        overall_progress.finish("Phase 3 failed due to error")
        raise

if __name__ == "__main__":
    main()
