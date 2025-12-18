"""
Master Script - Run All Advanced Stacking Models
Executes the complete pipeline from feature engineering to final comparison
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 80)
    print(f" {description}")
    print("=" * 80)
    print(f"Running: {script_name}")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )
        print(f"\n✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed!")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error in {description}")
        print(f"Error: {e}")
        return False

def main():
    print("=" * 80)
    print("ADVANCED STACKING MODELS - FULL PIPELINE")
    print("SOTA Approach for Birth Weight Prediction")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Pipeline steps
    pipeline = [
        ("feature_engineering.py", "Step 1: Feature Engineering (Power Features)"),
        ("catboost_baseline.py", "Step 2: CatBoost Baseline Model"),
        ("stacking_ensemble.py", "Step 3: Stacking Ensemble (XGBoost + CatBoost + Linear)"),
        ("comparison_report.py", "Step 4: Generate Comparison Report")
    ]
    
    results = []
    
    # Execute pipeline
    for script, description in pipeline:
        success = run_script(script, description)
        results.append((description, success))
        
        if not success:
            print("\n" + "!" * 80)
            print("PIPELINE STOPPED DUE TO ERROR")
            print("!" * 80)
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    for description, success in results:
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"{status}: {description}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n" + "🎉" * 40)
        print("\n✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("\n📊 Check the Results folder for:")
        print("   • Model metrics JSON files")
        print("   • Feature importance CSVs")
        print("   • COMPARISON_REPORT.md (comprehensive analysis)")
        print("\n" + "🎉" * 40)
    else:
        print("\n⚠ Some steps failed. Please check the errors above.")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
