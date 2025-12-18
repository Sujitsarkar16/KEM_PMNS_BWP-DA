"""
Random Forest Baseline - PMNS Variables (20 Features from Previous Work)
=========================================================================

This script implements a baseline Random Forest model using the 20 features
identified from previous PMNS research work.

Features by Category:
- Obstetric History (1): Mother's parity at visit 1
- Body Size (4): Weight prepregnancy, Fundal height, Abdominal circumference, Weight at visit 2
- Dietary Intake (4): Green chilli, Lunch calories, Food scores
- Blood Pressure (2): Pulse readings at visit 2
- Biochemistry (3): Fasting glucose, Red cell folate, GLV score
- Delivery Outcome (2): Placental weight, Gestational age at delivery
- Paternal Features (2): Head circumference, Platelet count
- Outcomes (1): Child sex
- Elder Child Age (1): Age of elder child

Author: Sujit Sarkar
Date: 2025-12-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
import json
import warnings
import os
from datetime import datetime
import joblib
warnings.filterwarnings('ignore')


class RFPMNSBaseline:
    """
    Random Forest baseline model using 20 PMNS features from previous research
    """
    
    def __init__(self, data_path='e:/KEM/Project/Data/PMNS_Data.csv'):
        """Initialize RF baseline implementation"""
        self.data_path = data_path
        self.data = None
        self.model = None
        self.results = {}
        
        # PMNS Features from previous work (20 features)
        self.selected_features = [
            # Obstetric History (1)
            'f0_m_parity_v1',
            
            # Body Size (4)
            'f0_m_wt_prepreg',
            'f0_m_fundal_ht_v2',
            'f0_m_abd_cir_v2',
            'f0_m_wt_v2',
            
            # Dietary Intake (4)
            'f0_m_r4_v2',
            'f0_m_lunch_cal_v1',
            'f0_m_p_sc_v1',
            'f0_m_o_sc_v1',
            
            # Blood Pressure (2)
            'f0_m_pulse_r1_v2',
            'f0_m_pulse_r2_v2',
            
            # Biochemistry (3)
            'f0_m_glu_f_v2',
            'f0_m_rcf_v2',
            'f0_m_g_sc_v1',
            
            # Delivery Outcome (2)
            'f0_m_plac_wt',
            'f0_m_GA_Del',
            
            # Paternal Features (2)
            'f0_f_head_cir_ini',
            'f0_f_plt_ini',
            
            # Outcomes (1)
            'f1_sex',
            
            # Elder Child Age (1)
            'f0_m_age_eld_child'
        ]
        
        self.target = 'f1_bw'
        
    def load_and_prepare_data(self):
        """Step 1: Load and prepare data"""
        print("=" * 80)
        print("STEP 1: DATA LOADING AND PREPARATION")
        print("=" * 80)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Verify features exist
        available_features = [f for f in self.selected_features if f in self.data.columns]
        missing_features = [f for f in self.selected_features if f not in self.data.columns]
        
        if missing_features:
            print(f"[WARNING] {len(missing_features)} features not found")
        
        self.selected_features = available_features
        print(f"[OK] Using {len(self.selected_features)} features")
        
        return self.data
    
    def prepare_data_splits(self, test_size=0.2, val_size=0.2, random_state=42):
        """Step 2: Prepare train/validation/test splits"""
        print("\n" + "=" * 80)
        print("STEP 2: DATA SPLITTING (60/20/20)")
        print("=" * 80)
        
        # Prepare X and y
        X = self.data[self.selected_features].values
        y = self.data[self.target].values
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
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
        print(f"  - Training:   {X_train.shape[0]} samples")
        print(f"  - Validation: {X_val.shape[0]} samples")
        print(f"  - Test:       {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_baseline_model(self, random_state=42):
        """Step 3: Train baseline Random Forest"""
        print("\n" + "=" * 80)
        print("STEP 3: TRAINING BASELINE RANDOM FOREST")
        print("=" * 80)
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': random_state,
            'n_jobs': -1
        }
        
        print("\n[Configuration] Default parameters:")
        for key, value in default_params.items():
            print(f"  {key}: {value}")
        
        # Train model
        print("\n[Training] Fitting Random Forest...")
        self.model = RandomForestRegressor(**default_params)
        self.model.fit(self.X_train, self.y_train)
        
        print(f"[OK] Model trained successfully!")
        print(f"  - Number of trees: {self.model.n_estimators}")
        print(f"  - Max depth: {self.model.max_depth}")
        
        return self.model
    
    def perform_cross_validation(self, cv_folds=5, random_state=42):
        """Step 4: Perform k-fold cross-validation"""
        print("\n" + "=" * 80)
        print("STEP 4: CROSS-VALIDATION")
        print("=" * 80)
        
        print(f"\n[INFO] Running {cv_folds}-fold cross-validation...")
        
        # Combine train and validation for CV
        X_cv = np.vstack([self.X_train, self.X_val])
        y_cv = np.concatenate([self.y_train, self.y_val])
        
        # Perform CV
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        cv_scores = -cross_val_score(
            self.model, X_cv, y_cv,
            cv=kfold,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        cv_rmse_scores = np.sqrt(cv_scores)
        
        self.cv_results = {
            'cv_rmse_mean': float(np.mean(cv_rmse_scores)),
            'cv_rmse_std': float(np.std(cv_rmse_scores)),
            'cv_rmse_scores': cv_rmse_scores.tolist(),
            'n_folds': cv_folds
        }
        
        print(f"[OK] Cross-validation completed:")
        print(f"  - Mean CV RMSE: {self.cv_results['cv_rmse_mean']:.4f} ± {self.cv_results['cv_rmse_std']:.4f}")
        print(f"  - Min CV RMSE:  {np.min(cv_rmse_scores):.4f}")
        print(f"  - Max CV RMSE:  {np.max(cv_rmse_scores):.4f}")
        
        return self.cv_results
    
    def evaluate_model(self):
        """Step 5: Evaluate on all splits"""
        print("\n" + "=" * 80)
        print("STEP 5: MODEL EVALUATION")
        print("=" * 80)
        
        results = {}
        
        for split_name, X, y in [
            ('train', self.X_train, self.y_train),
            ('validation', self.X_val, self.y_val),
            ('test', self.X_test, self.y_test)
        ]:
            # Predictions
            y_pred = self.model.predict(X)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            mape = np.mean(np.abs((y - y_pred) / y)) * 100
            correlation, p_value = pearsonr(y, y_pred)
            
            results[split_name] = {
                'RMSE': float(rmse),
                'MAE': float(mae),
                'R²': float(r2),
                'MAPE': float(mape),
                'Correlation': float(correlation),
                'P-value': float(p_value),
                'Sample_Size': int(len(y))
            }
            
            print(f"\n[{split_name.upper()}] Performance:")
            print(f"  - RMSE:        {rmse:.4f} grams")
            print(f"  - MAE:         {mae:.4f} grams")
            print(f"  - R²:          {r2:.4f}")
            print(f"  - Correlation: {correlation:.4f}")
        
        self.results = results
        return results
    
    def calculate_feature_importance(self):
        """Step 6: Calculate feature importance"""
        print("\n" + "=" * 80)
        print("STEP 6: FEATURE IMPORTANCE")
        print("=" * 80)
        
        # Get feature importance
        importances = self.model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        print("\n[Top 10 Most Important Features]:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']:35s}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_results(self):
        """Step 7: Save all results"""
        print("\n" + "=" * 80)
        print("STEP 7: SAVING RESULTS")
        print("=" * 80)
        
        # Create output directory
        output_dir = 'e:/KEM/Project/PMNS_Variables/Results_PMNS'
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare comprehensive results
        comprehensive_results = {
            'model_type': 'RandomForest_Baseline_PMNS_Variables',
            'num_features': len(self.selected_features),
            'features': self.selected_features,
            'hyperparameters': {
                'n_estimators': int(self.model.n_estimators),
                'max_depth': self.model.max_depth,
                'min_samples_split': int(self.model.min_samples_split),
                'min_samples_leaf': int(self.model.min_samples_leaf),
                'max_features': self.model.max_features
            },
            'cross_validation': self.cv_results,
            'performance_metrics': self.results,
            'timestamp': timestamp
        }
        
        # Save JSON
        results_file = f'{output_dir}/rf_pmns_baseline_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        # Save metrics CSV
        metrics_data = []
        for split in ['train', 'validation', 'test']:
            row = {'split': split}
            row.update(self.results[split])
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f'{output_dir}/rf_pmns_baseline_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        
        # Save feature importance
        importance_file = f'{output_dir}/rf_pmns_baseline_importance_{timestamp}.csv'
        self.feature_importance.to_csv(importance_file, index=False)
        
        # Save model
        model_file = f'{output_dir}/rf_pmns_baseline_model_{timestamp}.pkl'
        joblib.dump(self.model, model_file)
        
        print(f"[OK] Results saved:")
        print(f"  - Results JSON: {results_file}")
        print(f"  - Metrics CSV:  {metrics_file}")
        print(f"  - Importance:   {importance_file}")
        print(f"  - Model PKL:    {model_file}")
        
        return results_file, metrics_file
    
    def run_baseline(self):
        """Run complete baseline pipeline"""
        print("=" * 80)
        print("RANDOM FOREST BASELINE - PMNS VARIABLES (20 FEATURES)")
        print("=" * 80)
        
        # Pipeline
        self.load_and_prepare_data()
        self.prepare_data_splits()
        self.train_baseline_model()
        self.perform_cross_validation()
        self.evaluate_model()
        self.calculate_feature_importance()
        self.save_results()
        
        # Final summary
        print("\n" + "=" * 80)
        print("BASELINE RANDOM FOREST TRAINING COMPLETED!")
        print("=" * 80)
        print(f"\n[FINAL RESULTS]")
        print(f"  Number of features: {len(self.selected_features)}")
        print(f"  Number of trees:    {self.model.n_estimators}")
        print(f"  CV RMSE:            {self.cv_results['cv_rmse_mean']:.4f} ± {self.cv_results['cv_rmse_std']:.4f}")
        print(f"  Test RMSE:          {self.results['test']['RMSE']:.4f} grams")
        print(f"  Test R²:            {self.results['test']['R²']:.4f}")
        print(f"  Test MAE:           {self.results['test']['MAE']:.4f} grams")
        print(f"  Test Correlation:   {self.results['test']['Correlation']:.4f}")
        print("=" * 80)
        
        return {
            'cv_results': self.cv_results,
            'test_metrics': self.results['test'],
            'feature_importance': self.feature_importance
        }


def main():
    """Main function"""
    print("=" * 80)
    print("RANDOM FOREST BASELINE - PMNS VARIABLES")
    print("=" * 80)
    
    # Configuration
    data_path = 'e:/KEM/Project/Data/PMNS_Data.csv'
    
    print(f"\n[Configuration]:")
    print(f"  - Data source: {data_path}")
    print(f"  - Model: Random Forest (default parameters)")
    print(f"  - Data split: 60% train, 20% validation, 20% test")
    print(f"  - Cross-validation: 5-fold")
    
    # Initialize and run
    baseline = RFPMNSBaseline(data_path=data_path)
    results = baseline.run_baseline()
    
    return results


if __name__ == "__main__":
    results = main()
