"""
Run All Top 25 Models
======================

This script runs all baseline and optimized models for the Top 25 features:
1. MLE Baseline & Optimized
2. Random Forest Baseline & Optimized
3. XGBoost Baseline & Optimized

It then summarizes the results.

Author: Sujit Sarkar
Date: 2025-12-06
"""

import sys
import os
import subprocess
import glob
import json
import pandas as pd
from datetime import datetime

# Define paths
BASE_DIR = 'e:/KEM/Project/Top 25 features'
MLE_DIR = os.path.join(BASE_DIR, 'MLE_top25_engineered')
RF_DIR = os.path.join(BASE_DIR, 'RF_top25_engineered')
XGB_DIR = os.path.join(BASE_DIR, 'XG_Boost_top_25')
RESULTS_DIR = os.path.join(BASE_DIR, 'Results')

# Define scripts to run
scripts = [
    (MLE_DIR, 'mle_top25_baseline.py', 'MLE Baseline'),
    (MLE_DIR, 'mle_top25_optimized.py', 'MLE Optimized'),
    (RF_DIR, 'rf_top25_baseline.py', 'Random Forest Baseline'),
    (RF_DIR, 'rf_top25_optimized.py', 'Random Forest Optimized'),
    (XGB_DIR, 'xgboost_top25_baseline.py', 'XGBoost Baseline'),
    (XGB_DIR, 'xgboost_top25_optimized.py', 'XGBoost Optimized')
]

def run_script(directory, script_name, description):
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"Script: {os.path.join(directory, script_name)}")
    print(f"{'='*80}\n")
    
    try:
        # Run the script
        result = subprocess.run(
            ['python', script_name],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"[OK] {description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed with error:")
        print(e.stderr)
        return False

def collect_results():
    print(f"\n{'='*80}")
    print(f"COLLECTING RESULTS")
    print(f"{'='*80}\n")
    
    results_summary = []
    
    # Process all JSON result files in the Results directory
    json_files = glob.glob(os.path.join(RESULTS_DIR, '*_results_*.json'))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            model_type = data.get('model_type', 'Unknown')
            timestamp = data.get('timestamp', '')
            
            # Extract metrics
            metrics = data.get('performance_metrics', {})
            test_metrics = metrics.get('test', {})
            
            rmse = test_metrics.get('RMSE', float('nan'))
            mae = test_metrics.get('MAE', float('nan'))
            r2 = test_metrics.get('R²', float('nan'))
            
            results_summary.append({
                'Model': model_type,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'Timestamp': timestamp,
                'File': os.path.basename(file_path)
            })
            
        except Exception as e:
            print(f"[WARNING] Could not process {file_path}: {e}")
    
    # Create DataFrame and sort
    if results_summary:
        df = pd.DataFrame(results_summary)
        df = df.sort_values(by='RMSE')
        
        # Save summary CSV
        output_path = os.path.join(RESULTS_DIR, 'top25_models_summary.csv')
        df.to_csv(output_path, index=False)
        
        print("\n[TOP 25 MODELS RESULTS TABLE]")
        print(df[['Model', 'RMSE', 'MAE', 'R2']].to_string(index=False))
        print(f"\n[OK] Summary saved to {output_path}")
    else:
        print("[WARNING] No results found.")

def main():
    print("STARTING TOP 25 MODELS EXECUTION PIPELINE")
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run all scripts
    success_count = 0
    for directory, script, desc in scripts:
        if run_script(directory, script, desc):
            success_count += 1
            
    print(f"\n\nExecution completed. {success_count}/{len(scripts)} scripts ran successfully.")
    
    # Collect and display results
    collect_results()

if __name__ == "__main__":
    main()
