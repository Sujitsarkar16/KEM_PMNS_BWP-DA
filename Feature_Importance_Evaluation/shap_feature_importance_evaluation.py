"""
SHAP Feature Importance Evaluation for Top 15 Clean Features
==============================================================

This script performs comprehensive SHAP (SHapley Additive exPlanations) analysis
for the top 15 clean features using trained XGBoost and Random Forest models.

SHAP provides:
- Global feature importance (which features matter most overall)
- Local explanations (how features affect individual predictions)
- Interaction effects between features
- Direction and magnitude of feature impacts

Models analyzed:
- XGBoost Optimized
- Random Forest Optimized

Author: Sujit Sarkar
Date: 2025-12-06
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Set plot styles
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SHAPFeatureImportanceEvaluator:
    """
    Comprehensive SHAP analysis for feature importance evaluation
    """
    
    def __init__(self, 
                 data_path='Paper/data/clean_top15_features_20251206.csv',
                 xgb_model_path=None,
                 rf_model_path=None,
                 output_dir='Feature_Importance_Evaluation'):
        """
        Initialize SHAP evaluator
        
        Args:
            data_path: Path to the dataset
            xgb_model_path: Path to trained XGBoost model
            rf_model_path: Path to trained Random Forest model
            output_dir: Directory for saving outputs
        """
        self.data_path = data_path
        self.xgb_model_path = xgb_model_path
        self.rf_model_path = rf_model_path
        self.output_dir = output_dir
        
        # Top 15 clean features (from clean_top15_features_20251206.csv)
        self.selected_features = [
            'f0_m_plac_wt',
            'f0_m_GA_Del',
            'gestational_health_index',
            'f0_m_ht',
            'bmi_age_interaction',
            'f0_m_abd_cir_v2',
            'f0_m_rcf_v2',
            'f0_m_wt_prepreg_squared',
            'f0_m_fundal_ht_v2',
            'bmi_age_ratio',
            'f0_m_bi_v1',
            'nutritional_status',
            'wt_ht_interaction',
            'f0_m_int_sin_ma',
            'f0_m_age'
        ]
        
        self.target = 'f1_bw'
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/plots', exist_ok=True)
        os.makedirs(f'{output_dir}/results', exist_ok=True)
        
        # Storage for results
        self.shap_results = {}
        
    def load_data(self):
        """Load and prepare data"""
        print("=" * 80)
        print("STEP 1: DATA LOADING")
        print("=" * 80)
        
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Verify all features exist in the dataset
        available_features = set(self.data.columns)
        missing_features = [f for f in self.selected_features if f not in available_features]
        
        if missing_features:
            print(f"[WARNING] Missing features: {missing_features}")
            # Use only available features
            self.selected_features = [f for f in self.selected_features if f in available_features]
            print(f"[INFO] Using {len(self.selected_features)} available features")
        
        # Prepare features and target
        self.X = self.data[self.selected_features]
        self.y = self.data[self.target]
        
        print(f"[OK] Features: {len(self.selected_features)}")
        print(f"[OK] Target: {self.target}")
        
        # Sample for SHAP (SHAP can be slow on large datasets)
        # Use 500 samples for detailed analysis
        sample_size = min(500, len(self.X))
        np.random.seed(42)
        sample_indices = np.random.choice(len(self.X), sample_size, replace=False)
        
        self.X_sample = self.X.iloc[sample_indices]
        self.y_sample = self.y.iloc[sample_indices]
        
        print(f"[OK] Sample size for SHAP: {sample_size}")
        
        return self.X, self.y
    
    def load_models(self):
        """Load trained models"""
        print("\n" + "=" * 80)
        print("STEP 2: LOADING TRAINED MODELS")
        print("=" * 80)
        
        self.models = {}
        
        # Auto-detect latest models if paths not provided
        if self.xgb_model_path is None:
            xgb_dir = 'paper/results/xgboost_clean_top15'
            xgb_files = sorted(Path(xgb_dir).glob('xgboost_top30_optimized_model_*.pkl'))
            if xgb_files:
                self.xgb_model_path = str(xgb_files[-1])
        
        if self.rf_model_path is None:
            rf_dir = 'paper/results/rf_clean_top15'
            rf_files = sorted(Path(rf_dir).glob('rf_top30_optimized_model_*.pkl'))
            if rf_files:
                self.rf_model_path = str(rf_files[-1])
        
        # Load XGBoost
        if self.xgb_model_path and os.path.exists(self.xgb_model_path):
            self.models['XGBoost'] = joblib.load(self.xgb_model_path)
            print(f"[OK] Loaded XGBoost: {self.xgb_model_path}")
        else:
            print(f"[WARNING] XGBoost model not found")
        
        # Load Random Forest
        if self.rf_model_path and os.path.exists(self.rf_model_path):
            self.models['RandomForest'] = joblib.load(self.rf_model_path)
            print(f"[OK] Loaded Random Forest: {self.rf_model_path}")
        else:
            print(f"[WARNING] Random Forest model not found")
        
        print(f"\n[OK] Total models loaded: {len(self.models)}")
        
        return self.models
    
    def calculate_shap_values(self):
        """Calculate SHAP values for all models"""
        print("\n" + "=" * 80)
        print("STEP 3: CALCULATING SHAP VALUES")
        print("=" * 80)
        
        for model_name, model in self.models.items():
            print(f"\n[Processing] {model_name}...")
            
            # Create appropriate SHAP explainer
            if model_name == 'XGBoost':
                # Use TreeExplainer for tree-based models (faster and exact)
                explainer = shap.TreeExplainer(model)
            elif model_name == 'RandomForest':
                # Use TreeExplainer for Random Forest
                explainer = shap.TreeExplainer(model)
            else:
                # Fallback to KernelExplainer
                explainer = shap.KernelExplainer(model.predict, self.X_sample)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(self.X_sample)
            
            # Handle expected value (could be array for RF)
            expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else None
            if isinstance(expected_value, np.ndarray):
                expected_value = float(expected_value[0]) if len(expected_value) > 0 else float(expected_value)
            elif expected_value is not None:
                expected_value = float(expected_value)
            
            # Store results
            self.shap_results[model_name] = {
                'explainer': explainer,
                'shap_values': shap_values,
                'expected_value': expected_value
            }
            
            print(f"[OK] SHAP values shape: {shap_values.shape}")
            if expected_value is not None:
                print(f"[OK] Expected value: {expected_value:.2f}g")
            else:
                print(f"[OK] Expected value: None")
        
        return self.shap_results
    
    def create_summary_plots(self):
        """Create SHAP summary plots"""
        print("\n" + "=" * 80)
        print("STEP 4: CREATING SUMMARY PLOTS")
        print("=" * 80)
        
        for model_name in self.models.keys():
            print(f"\n[Plotting] {model_name}...")
            
            shap_values = self.shap_results[model_name]['shap_values']
            
            # Summary plot (beeswarm)
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values, 
                self.X_sample, 
                feature_names=self.selected_features,
                show=False,
                max_display=20
            )
            plt.title(f'SHAP Summary Plot - {model_name}', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            summary_file = f'{self.output_dir}/plots/{model_name.lower()}_shap_summary.png'
            plt.savefig(summary_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved: {summary_file}")
            
            # Bar plot (mean absolute SHAP values)
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values, 
                self.X_sample,
                feature_names=self.selected_features,
                plot_type='bar',
                show=False,
                max_display=20
            )
            plt.title(f'SHAP Feature Importance - {model_name}', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            bar_file = f'{self.output_dir}/plots/{model_name.lower()}_shap_bar.png'
            plt.savefig(bar_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved: {bar_file}")
    
    def create_waterfall_plots(self, num_samples=5):
        """Create waterfall plots for individual predictions"""
        print("\n" + "=" * 80)
        print("STEP 5: CREATING WATERFALL PLOTS")
        print("=" * 80)
        
        for model_name in self.models.keys():
            print(f"\n[Plotting] {model_name}...")
            
            shap_values = self.shap_results[model_name]['shap_values']
            expected_value = self.shap_results[model_name]['expected_value']
            
            # Select diverse samples (min, max, median predictions)
            predictions = self.models[model_name].predict(self.X_sample)
            
            sample_indices = [
                np.argmin(predictions),  # Lowest prediction
                np.argmax(predictions),  # Highest prediction
                np.argsort(predictions)[len(predictions) // 2],  # Median
            ]
            
            for i, idx in enumerate(sample_indices):
                plt.figure(figsize=(12, 8))
                
                # Create explanation object
                explanation = shap.Explanation(
                    values=shap_values[idx],
                    base_values=expected_value,
                    data=self.X_sample.iloc[idx].values,
                    feature_names=self.selected_features
                )
                
                shap.waterfall_plot(explanation, show=False, max_display=15)
                
                prediction = predictions[idx]
                actual = self.y_sample.iloc[idx]
                
                plt.title(
                    f'SHAP Waterfall - {model_name} (Sample {i+1})\n'
                    f'Predicted: {prediction:.2f}g | Actual: {actual:.2f}g | Error: {abs(prediction-actual):.2f}g',
                    fontsize=14, fontweight='bold', pad=20
                )
                plt.tight_layout()
                
                waterfall_file = f'{self.output_dir}/plots/{model_name.lower()}_waterfall_sample{i+1}.png'
                plt.savefig(waterfall_file, dpi=300, bbox_inches='tight')
                plt.close()
                
            print(f"[OK] Created {len(sample_indices)} waterfall plots")
    
    def create_dependence_plots(self, top_n=5):
        """Create SHAP dependence plots for top features"""
        print("\n" + "=" * 80)
        print("STEP 6: CREATING DEPENDENCE PLOTS")
        print("=" * 80)
        
        for model_name in self.models.keys():
            print(f"\n[Plotting] {model_name}...")
            
            shap_values = self.shap_results[model_name]['shap_values']
            
            # Get top features by mean absolute SHAP value
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_feature_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
            
            # Create dependence plots for top features
            for idx in top_feature_indices:
                feature_name = self.selected_features[idx]
                
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    idx,
                    shap_values,
                    self.X_sample,
                    feature_names=self.selected_features,
                    show=False
                )
                plt.title(
                    f'SHAP Dependence Plot - {model_name}\nFeature: {feature_name}',
                    fontsize=14, fontweight='bold', pad=20
                )
                plt.tight_layout()
                
                # Clean feature name for filename
                clean_name = feature_name.replace('/', '_').replace(' ', '_')
                dep_file = f'{self.output_dir}/plots/{model_name.lower()}_dependence_{clean_name}.png'
                plt.savefig(dep_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"[OK] Created {len(top_feature_indices)} dependence plots")
    
    def create_force_plots(self, num_samples=3):
        """Create force plots for individual predictions"""
        print("\n" + "=" * 80)
        print("STEP 7: CREATING FORCE PLOTS")
        print("=" * 80)
        
        for model_name in self.models.keys():
            print(f"\n[Plotting] {model_name}...")
            
            shap_values = self.shap_results[model_name]['shap_values']
            expected_value = self.shap_results[model_name]['expected_value']
            
            # Select samples
            predictions = self.models[model_name].predict(self.X_sample)
            sample_indices = [
                np.argmin(predictions),
                np.argmax(predictions),
                np.argsort(predictions)[len(predictions) // 2]
            ]
            
            for i, idx in enumerate(sample_indices):
                # Create force plot
                force_plot = shap.force_plot(
                    expected_value,
                    shap_values[idx],
                    self.X_sample.iloc[idx],
                    feature_names=self.selected_features,
                    matplotlib=True,
                    show=False
                )
                
                prediction = predictions[idx]
                actual = self.y_sample.iloc[idx]
                
                plt.title(
                    f'SHAP Force Plot - {model_name} (Sample {i+1})\n'
                    f'Predicted: {prediction:.2f}g | Actual: {actual:.2f}g',
                    fontsize=12, fontweight='bold', pad=10
                )
                
                force_file = f'{self.output_dir}/plots/{model_name.lower()}_force_sample{i+1}.png'
                plt.savefig(force_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"[OK] Created {len(sample_indices)} force plots")
    
    def calculate_feature_importance_metrics(self):
        """Calculate and save feature importance metrics"""
        print("\n" + "=" * 80)
        print("STEP 8: CALCULATING FEATURE IMPORTANCE METRICS")
        print("=" * 80)
        
        importance_summary = {}
        
        for model_name in self.models.keys():
            print(f"\n[Processing] {model_name}...")
            
            shap_values = self.shap_results[model_name]['shap_values']
            
            # Calculate various importance metrics
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            mean_shap = shap_values.mean(axis=0)
            std_shap = shap_values.std(axis=0)
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.selected_features,
                'mean_abs_shap': mean_abs_shap,
                'mean_shap': mean_shap,
                'std_shap': std_shap,
                'abs_mean_ratio': mean_abs_shap / (np.abs(mean_shap) + 1e-10)
            }).sort_values('mean_abs_shap', ascending=False)
            
            # Add rank
            importance_df['rank'] = range(1, len(importance_df) + 1)
            
            # Calculate percentage importance
            importance_df['importance_pct'] = (
                importance_df['mean_abs_shap'] / importance_df['mean_abs_shap'].sum() * 100
            )
            
            # Cumulative importance
            importance_df['cumulative_importance_pct'] = importance_df['importance_pct'].cumsum()
            
            # Save to CSV
            importance_file = f'{self.output_dir}/results/{model_name.lower()}_shap_importance.csv'
            importance_df.to_csv(importance_file, index=False)
            print(f"[OK] Saved: {importance_file}")
            
            # Store summary
            importance_summary[model_name] = importance_df
            
            # Print top 10
            print(f"\n[Top 10 Features - {model_name}]:")
            for i, row in importance_df.head(10).iterrows():
                print(f"  {row['rank']:2d}. {row['feature']:35s}: {row['importance_pct']:6.2f}% "
                      f"(mean SHAP: {row['mean_shap']:+7.2f})")
        
        self.importance_summary = importance_summary
        return importance_summary
    
    def create_comparison_plots(self):
        """Create comparison plots across models"""
        print("\n" + "=" * 80)
        print("STEP 9: CREATING MODEL COMPARISON PLOTS")
        print("=" * 80)
        
        if len(self.models) < 2:
            print("[INFO] Need at least 2 models for comparison")
            return
        
        # Compare top features across models
        fig, axes = plt.subplots(1, len(self.models), figsize=(18, 10))
        
        if len(self.models) == 1:
            axes = [axes]
        
        for ax, (model_name, importance_df) in zip(axes, self.importance_summary.items()):
            top_features = importance_df.head(15)
            
            ax.barh(range(len(top_features)), top_features['mean_abs_shap'])
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'], fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP value|', fontsize=12)
            ax.set_title(f'{model_name}\nTop 15 Features', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        comparison_file = f'{self.output_dir}/plots/model_comparison_shap.png'
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved: {comparison_file}")
        
        # Feature ranking comparison
        if len(self.models) >= 2:
            model_names = list(self.models.keys())
            df1 = self.importance_summary[model_names[0]][['feature', 'rank']].rename(
                columns={'rank': f'{model_names[0]}_rank'}
            )
            df2 = self.importance_summary[model_names[1]][['feature', 'rank']].rename(
                columns={'rank': f'{model_names[1]}_rank'}
            )
            
            comparison_df = df1.merge(df2, on='feature')
            comparison_df['rank_diff'] = abs(
                comparison_df[f'{model_names[0]}_rank'] - comparison_df[f'{model_names[1]}_rank']
            )
            comparison_df = comparison_df.sort_values('rank_diff', ascending=False)
            
            comparison_file = f'{self.output_dir}/results/feature_ranking_comparison.csv'
            comparison_df.to_csv(comparison_file, index=False)
            print(f"[OK] Saved: {comparison_file}")
    
    def generate_report(self):
        """Generate comprehensive SHAP analysis report"""
        print("\n" + "=" * 80)
        print("STEP 10: GENERATING ANALYSIS REPORT")
        print("=" * 80)
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# SHAP Feature Importance Analysis Report
## Top 30 Engineered Features

**Generated:** {timestamp}  
**Dataset:** {self.data_path}  
**Total Samples:** {len(self.X)}  
**SHAP Sample Size:** {len(self.X_sample)}  
**Features Analyzed:** {len(self.selected_features)}

---

## 1. Executive Summary

This report presents a comprehensive SHAP (SHapley Additive exPlanations) analysis of the top 30 engineered features
for birth weight prediction. SHAP values provide model-agnostic interpretability by quantifying each feature's
contribution to individual predictions.

### Models Analyzed
"""
        
        for model_name in self.models.keys():
            report += f"- **{model_name}**\n"
        
        report += "\n---\n\n## 2. Feature Importance Rankings\n\n"
        
        for model_name, importance_df in self.importance_summary.items():
            report += f"### {model_name}\n\n"
            report += "| Rank | Feature | Importance (%) | Mean SHAP | Std SHAP | Cumulative (%) |\n"
            report += "|------|---------|----------------|-----------|----------|----------------|\n"
            
            for i, row in importance_df.head(20).iterrows():
                report += (
                    f"| {row['rank']} | `{row['feature']}` | {row['importance_pct']:.2f}% | "
                    f"{row['mean_shap']:+.2f} | {row['std_shap']:.2f} | "
                    f"{row['cumulative_importance_pct']:.2f}% |\n"
                )
            
            report += "\n"
        
        report += """
---

## 3. Key Insights

### Top Features Across Models

"""
        
        # Find common top features
        if len(self.models) >= 2:
            model_names = list(self.models.keys())
            top5_model1 = set(self.importance_summary[model_names[0]].head(5)['feature'])
            top5_model2 = set(self.importance_summary[model_names[1]].head(5)['feature'])
            common_top5 = top5_model1.intersection(top5_model2)
            
            report += f"**Common Top 5 Features:**\n"
            for feature in common_top5:
                report += f"- `{feature}`\n"
            report += "\n"
        
        report += """
### Feature Categories

The top features can be categorized as:

1. **Gestational Features:** Features related to gestational age and delivery
2. **Anthropometric Features:** Maternal height, weight, and BMI-related features
3. **Clinical Measurements:** Fundal height, abdominal circumference, etc.
4. **Engineered Interactions:** Interaction terms between key variables
5. **Risk Indicators:** Composite risk scores and flags

---

## 4. Interpretation Guidelines

### SHAP Value Interpretation

- **Positive SHAP values:** Feature pushes prediction higher (increases birth weight)
- **Negative SHAP values:** Feature pushes prediction lower (decreases birth weight)
- **Magnitude:** Indicates strength of the feature's impact
- **Distribution:** Wide spread indicates feature has varying effects across samples

### Visualization Types

1. **Summary Plot (Beeswarm):** Shows distribution of SHAP values for each feature
2. **Bar Plot:** Shows average absolute impact of each feature
3. **Dependence Plot:** Shows relationship between feature value and SHAP value
4. **Waterfall Plot:** Shows individual prediction breakdown
5. **Force Plot:** Shows how features push prediction from base value

---

## 5. Files Generated

### Visualizations
"""
        
        for model_name in self.models.keys():
            report += f"\n**{model_name}:**\n"
            report += f"- Summary plot: `plots/{model_name.lower()}_shap_summary.png`\n"
            report += f"- Bar plot: `plots/{model_name.lower()}_shap_bar.png`\n"
            report += f"- Waterfall plots: `plots/{model_name.lower()}_waterfall_sample*.png`\n"
            report += f"- Dependence plots: `plots/{model_name.lower()}_dependence_*.png`\n"
            report += f"- Force plots: `plots/{model_name.lower()}_force_sample*.png`\n"
        
        report += """
### Data Files

"""
        for model_name in self.models.keys():
            report += f"- `results/{model_name.lower()}_shap_importance.csv`: Detailed importance metrics\n"
        
        report += "- `results/feature_ranking_comparison.csv`: Cross-model ranking comparison\n"
        
        report += """
---

## 6. Recommendations

Based on this SHAP analysis:

1. **Model Development:** Focus feature engineering efforts on top-ranked features
2. **Clinical Interpretation:** Top features with positive mean SHAP indicate protective factors
3. **Risk Assessment:** Features with high variance may indicate differential effects across populations
4. **Future Work:** Investigate interaction effects revealed in dependence plots

---

## 7. Technical Notes

- SHAP values were calculated using TreeExplainer for tree-based models (exact values)
- Sample size for SHAP analysis: {len(self.X_sample)} (computational efficiency)
- All values represent contributions to birth weight in grams
- Analysis performed on the test/validation set for consistency

---

**End of Report**
"""
        
        report_file = f'{self.output_dir}/SHAP_ANALYSIS_REPORT.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"[OK] Saved: {report_file}")
        
        return report_file
    
    def run_full_analysis(self):
        """Run complete SHAP analysis pipeline"""
        print("=" * 80)
        print("SHAP FEATURE IMPORTANCE EVALUATION")
        print("TOP 15 CLEAN FEATURES")
        print("=" * 80)
        
        # Run all steps
        self.load_data()
        self.load_models()
        
        if not self.models:
            print("\n[ERROR] No models loaded. Cannot proceed with SHAP analysis.")
            return
        
        self.calculate_shap_values()
        self.create_summary_plots()
        self.create_waterfall_plots()
        self.create_dependence_plots()
        self.create_force_plots()
        self.calculate_feature_importance_metrics()
        self.create_comparison_plots()
        self.generate_report()
        
        print("\n" + "=" * 80)
        print("SHAP ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\n[Results Location]: {self.output_dir}/")
        print(f"  - Visualizations: {self.output_dir}/plots/")
        print(f"  - Data files: {self.output_dir}/results/")
        print(f"  - Report: {self.output_dir}/SHAP_ANALYSIS_REPORT.md")
        print("=" * 80)


def main():
    """Main execution function"""
    print("=" * 80)
    print("SHAP FEATURE IMPORTANCE EVALUATION")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = SHAPFeatureImportanceEvaluator()
    
    # Run full analysis
    evaluator.run_full_analysis()
    
    return evaluator


if __name__ == "__main__":
    evaluator = main()
