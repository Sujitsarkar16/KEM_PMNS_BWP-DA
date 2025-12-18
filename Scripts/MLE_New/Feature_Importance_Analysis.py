"""
Comprehensive Feature Importance Analysis for Birthweight Prediction
====================================================================

This script analyzes feature importance using multiple methods:
1. Correlation Analysis (Simple, fast)
2. Random Forest Feature Importance (Model-based)
3. XGBoost Feature Importance (Model-based)
4. Mutual Information (Information-theoretic)
5. Statistical Tests (f_regression - F-statistic)

Author: Sujit sarkar
Date: 2025-10-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, mutual_info_regression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis using multiple methods
    """
    
    def __init__(self, data_path):
        """Initialize analyzer"""
        self.data_path = data_path
        self.data = None
        self.importance_results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for analysis"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        print(f"[OK] Loaded dataset: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        
        # Load variable grouping
        self.variable_groups = pd.read_csv('Data/processed/variable_grouping_table.csv')
        
        # Identify continuous and categorical variables
        continuous_vars = self.variable_groups[self.variable_groups['Type'] == 'Continuous']['Variable'].tolist()
        categorical_vars = self.variable_groups[self.variable_groups['Type'] == 'Categorical']['Variable'].tolist()
        
        # Filter to available variables
        self.continuous_vars = [var for var in continuous_vars if var in self.data.columns]
        self.categorical_vars = [var for var in categorical_vars if var in self.data.columns]
        
        print(f"[OK] Continuous variables: {len(self.continuous_vars)}")
        print(f"[OK] Categorical variables: {len(self.categorical_vars)}")
        
        return self.data
    
    def analyze_correlation(self):
        """Method 1: Correlation Analysis (Simple and fast)"""
        print("\n" + "=" * 80)
        print("METHOD 1: CORRELATION ANALYSIS")
        print("=" * 80)
        
        correlations = {}
        
        # Analyze correlation between each continuous variable and birthweight
        if 'f1_bw' not in self.data.columns:
            print("[ERROR] Birthweight (f1_bw) not found in data")
            return correlations
        
        for var in self.continuous_vars:
            if var == 'f1_bw':
                continue
            
            # Calculate Pearson correlation
            data_clean = self.data[[var, 'f1_bw']].dropna()
            if len(data_clean) > 10:  # Need at least 10 observations
                corr, p_value = stats.pearsonr(data_clean[var], data_clean['f1_bw'])
                if not np.isnan(corr):
                    correlations[var] = {
                        'correlation': abs(corr),
                        'p_value': p_value,
                        'sample_size': len(data_clean)
                    }
        
        # Sort by absolute correlation
        correlations = dict(sorted(correlations.items(), key=lambda x: x[1]['correlation'], reverse=True))
        
        self.importance_results['correlation'] = correlations
        
        print(f"[OK] Analyzed {len(correlations)} variables")
        print("\nTop 10 variables by correlation:")
        for i, (var, result) in enumerate(list(correlations.items())[:10], 1):
            print(f"  {i}. {var}: r={result['correlation']:.4f} (p={result['p_value']:.4e})")
        
        return correlations
    
    def analyze_random_forest(self):
        """Method 2: Random Forest Feature Importance"""
        print("\n" + "=" * 80)
        print("METHOD 2: RANDOM FOREST FEATURE IMPORTANCE")
        print("=" * 80)
        
        # Prepare data
        feature_vars = [var for var in self.continuous_vars if var != 'f1_bw']
        feature_vars = [var for var in feature_vars if self.data[var].notna().sum() > 10]
        
        # Use only features with <50% missing data
        feature_vars = [var for var in feature_vars 
                       if (self.data[var].isna().sum() / len(self.data)) < 0.5]
        
        print(f"[OK] Using {len(feature_vars)} features for Random Forest analysis")
        
        # Prepare X and y
        X = self.data[feature_vars].copy()
        y = self.data['f1_bw'].copy()
        
        # Handle missing values (simple mean imputation for RF)
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Get feature importance
        rf_importance = {}
        for var, importance in zip(feature_vars, rf.feature_importances_):
            rf_importance[var] = float(importance)
        
        # Sort by importance
        rf_importance = dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True))
        
        self.importance_results['random_forest'] = rf_importance
        
        # Evaluate model
        rf_pred = rf.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - rf_pred) ** 2))
        r2 = rf.score(X_test, y_test)
        
        print(f"[OK] Model trained: RMSE={rmse:.2f}, R²={r2:.4f}")
        print("\nTop 10 variables by Random Forest importance:")
        for i, (var, importance) in enumerate(list(rf_importance.items())[:10], 1):
            print(f"  {i}. {var}: {importance:.4f}")
        
        return rf_importance
    
    def analyze_xgboost(self):
        """Method 3: XGBoost Feature Importance"""
        print("\n" + "=" * 80)
        print("METHOD 3: XGBOOST FEATURE IMPORTANCE")
        print("=" * 80)
        
        # Prepare data (same features as RF)
        feature_vars = [var for var in self.continuous_vars if var != 'f1_bw']
        feature_vars = [var for var in feature_vars if self.data[var].notna().sum() > 10]
        feature_vars = [var for var in feature_vars 
                       if (self.data[var].isna().sum() / len(self.data)) < 0.5]
        
        print(f"[OK] Using {len(feature_vars)} features for XGBoost analysis")
        
        # Prepare X and y
        X = self.data[feature_vars].copy()
        y = self.data['f1_bw'].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost
        xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        
        # Get feature importance
        xgb_importance = {}
        for var, importance in zip(feature_vars, xgb.feature_importances_):
            xgb_importance[var] = float(importance)
        
        # Sort by importance
        xgb_importance = dict(sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True))
        
        self.importance_results['xgboost'] = xgb_importance
        
        # Evaluate model
        xgb_pred = xgb.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - xgb_pred) ** 2))
        r2 = xgb.score(X_test, y_test)
        
        print(f"[OK] Model trained: RMSE={rmse:.2f}, R²={r2:.4f}")
        print("\nTop 10 variables by XGBoost importance:")
        for i, (var, importance) in enumerate(list(xgb_importance.items())[:10], 1):
            print(f"  {i}. {var}: {importance:.4f}")
        
        return xgb_importance
    
    def analyze_mutual_information(self):
        """Method 4: Mutual Information"""
        print("\n" + "=" * 80)
        print("METHOD 4: MUTUAL INFORMATION")
        print("=" * 80)
        
        # Prepare data
        feature_vars = [var for var in self.continuous_vars if var != 'f1_bw']
        feature_vars = [var for var in feature_vars if self.data[var].notna().sum() > 10]
        feature_vars = [var for var in feature_vars 
                       if (self.data[var].isna().sum() / len(self.data)) < 0.5]
        
        print(f"[OK] Using {len(feature_vars)} features for Mutual Information analysis")
        
        # Prepare X and y
        X = self.data[feature_vars].fillna(self.data[feature_vars].mean())
        y = self.data['f1_bw']
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        mi_importance = {}
        for var, score in zip(feature_vars, mi_scores):
            mi_importance[var] = float(score)
        
        # Sort by MI score
        mi_importance = dict(sorted(mi_importance.items(), key=lambda x: x[1], reverse=True))
        
        self.importance_results['mutual_information'] = mi_importance
        
        print(f"[OK] Analyzed {len(mi_importance)} variables")
        print("\nTop 10 variables by Mutual Information:")
        for i, (var, score) in enumerate(list(mi_importance.items())[:10], 1):
            print(f"  {i}. {var}: {score:.4f}")
        
        return mi_importance
    
    def analyze_statistical_tests(self):
        """Method 5: Statistical Tests (F-regression)"""
        print("\n" + "=" * 80)
        print("METHOD 5: STATISTICAL TESTS (F-REGRESSION)")
        print("=" * 80)
        
        # Prepare data
        feature_vars = [var for var in self.continuous_vars if var != 'f1_bw']
        feature_vars = [var for var in feature_vars if self.data[var].notna().sum() > 10]
        feature_vars = [var for var in feature_vars 
                       if (self.data[var].isna().sum() / len(self.data)) < 0.5]
        
        print(f"[OK] Using {len(feature_vars)} features for F-test analysis")
        
        # Prepare X and y
        X = self.data[feature_vars].fillna(self.data[feature_vars].mean())
        y = self.data['f1_bw']
        
        # Calculate F-statistics
        f_scores, p_values = f_regression(X, y)
        
        f_importance = {}
        for var, score, p_val in zip(feature_vars, f_scores, p_values):
            f_importance[var] = {
                'f_score': float(score),
                'p_value': float(p_val)
            }
        
        # Sort by F-score
        f_importance = dict(sorted(f_importance.items(), key=lambda x: x[1]['f_score'], reverse=True))
        
        self.importance_results['f_regression'] = f_importance
        
        print(f"[OK] Analyzed {len(f_importance)} variables")
        print("\nTop 10 variables by F-statistic:")
        for i, (var, result) in enumerate(list(f_importance.items())[:10], 1):
            print(f"  {i}. {var}: F={result['f_score']:.2f}, p={result['p_value']:.4e}")
        
        return f_importance
    
    def combine_all_methods(self):
        """Combine all methods to get a comprehensive ranking"""
        print("\n" + "=" * 80)
        print("COMBINING ALL METHODS")
        print("=" * 80)
        
        # Get all unique variables from all methods
        all_vars = set()
        for method_results in self.importance_results.values():
            all_vars.update(method_results.keys())
        
        # Calculate combined scores
        combined_scores = {}
        
        for var in all_vars:
            scores = []
            
            # Correlation score (normalize to 0-1)
            if var in self.importance_results.get('correlation', {}):
                corr = self.importance_results['correlation'][var]['correlation']
                scores.append(('correlation', corr))
            
            # RF importance (already 0-1)
            if var in self.importance_results.get('random_forest', {}):
                scores.append(('rf', self.importance_results['random_forest'][var]))
            
            # XGB importance (already 0-1)
            if var in self.importance_results.get('xgboost', {}):
                scores.append(('xgb', self.importance_results['xgboost'][var]))
            
            # MI score (normalize to 0-1)
            if var in self.importance_results.get('mutual_information', {}):
                mi = self.importance_results['mutual_information'][var]
                scores.append(('mi', mi))
            
            # F-score (normalize to 0-1)
            if var in self.importance_results.get('f_regression', {}):
                f = self.importance_results['f_regression'][var]['f_score']
                scores.append(('f_test', f))
            
            # Calculate average score
            if scores:
                avg_score = np.mean([s[1] for s in scores])
                combined_scores[var] = {
                    'combined_score': avg_score,
                    'num_methods': len(scores),
                    'individual_scores': scores
                }
        
        # Sort by combined score
        combined_scores = dict(sorted(combined_scores.items(), 
                                     key=lambda x: x[1]['combined_score'], 
                                     reverse=True))
        
        self.importance_results['combined'] = combined_scores
        
        print(f"[OK] Combined scores for {len(combined_scores)} variables")
        print("\nTop 20 MOST IMPORTANT FEATURES for Birthweight Prediction:")
        print("-" * 80)
        
        for i, (var, result) in enumerate(list(combined_scores.items())[:20], 1):
            print(f"{i:2d}. {var:30s} | Score: {result['combined_score']:.4f} | Methods: {result['num_methods']}")
        
        return combined_scores
    
    def create_visualizations(self):
        """Create visualizations of feature importance"""
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        
        import os
        os.makedirs('PLOTS/MLE_New', exist_ok=True)
        
        # Create a comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Get combined scores
        combined = self.importance_results['combined']
        top_vars = list(combined.keys())[:20]
        top_scores = [combined[var]['combined_score'] for var in top_vars]
        
        # Create bar plot
        ax = plt.subplot(2, 2, 1)
        bars = ax.barh(range(len(top_vars)), top_scores, color='steelblue')
        ax.set_yticks(range(len(top_vars)))
        ax.set_yticklabels([var.replace('f0_m_', '').replace('f1_', '') for var in top_vars], fontsize=9)
        ax.set_xlabel('Combined Importance Score')
        ax.set_title('Top 20 Most Important Features for Birthweight Prediction')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            ax.text(score, i, f' {score:.3f}', va='center')
        
        # Comparison across methods
        ax2 = plt.subplot(2, 2, 2)
        methods = ['correlation', 'random_forest', 'xgboost', 'mutual_information']
        top_5_vars = top_vars[:5]
        
        method_names = []
        for method in methods:
            if method in self.importance_results:
                method_names.append(method.replace('_', ' ').title())
        
        x = np.arange(len(top_5_vars))
        width = 0.2
        
        for i, method in enumerate(methods):
            if method in self.importance_results:
                scores = []
                for var in top_5_vars:
                    if var in self.importance_results[method]:
                        if isinstance(self.importance_results[method][var], dict):
                            # Extract numeric value
                            val = self.importance_results[method][var].get('correlation', 
                                    self.importance_results[method][var].get('f_score', 0))
                        else:
                            val = self.importance_results[method][var]
                        scores.append(val)
                    else:
                        scores.append(0)
                
                ax2.bar(x + i * width, scores, width, label=method.replace('_', ' ').title(), alpha=0.8)
        
        ax2.set_ylabel('Importance Score')
        ax2.set_title('Top 5 Features: Comparison Across Methods')
        ax2.set_xticks(x + width * 1.5)
        ax2.set_xticklabels([var.replace('f0_m_', '').replace('f1_', '') for var in top_5_vars], 
                           rotation=45, ha='right', fontsize=8)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Method agreement heatmap
        ax3 = plt.subplot(2, 2, 3)
        top_10_vars = top_vars[:10]
        method_ranks = {}
        
        for method in methods:
            if method in self.importance_results:
                rankings = []
                for var in top_10_vars:
                    if var in self.importance_results[method]:
                        rankings.append(1)
                    else:
                        rankings.append(0)
                method_ranks[method] = rankings
        
        if method_ranks:
            heatmap_data = np.array(list(method_ranks.values()))
            sns.heatmap(heatmap_data, ax=ax3, cmap='YlOrRd', cbar=False, 
                       yticklabels=[m.replace('_', ' ').title() for m in method_ranks.keys()],
                       xticklabels=[v.replace('f0_m_', '').replace('f1_', '') for v in top_10_vars],
                       fmt='d')
            ax3.set_title('Feature Presence Across Methods (Top 10)')
            ax3.set_xlabel('Features')
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Category distribution
        ax4 = plt.subplot(2, 2, 4)
        
        # Categorize features
        categories = {
            'Gestational': ['GA_Del', 'GA_V1', 'GA_V2'],
            'Anthropometric': ['age', 'ht', 'wt', 'bmi', 'fundal', 'abd_cir', 'bi_v'],
            'Health Markers': ['hb', 'glu', 'bp', 'r9', 'rcf'],
            'Placental': ['plac_wt'],
            'Other': []
        }
        
        category_counts = {}
        for var in top_10_vars:
            categorized = False
            for cat, keywords in categories.items():
                if any(kw in var for kw in keywords):
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    categorized = True
                    break
            if not categorized:
                category_counts['Other'] = category_counts.get('Other', 0) + 1
        
        ax4.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%',
               startangle=90)
        ax4.set_title('Top 10 Features by Category')
        
        plt.tight_layout()
        plt.savefig('PLOTS/MLE_New/feature_importance_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("[OK] Visualizations saved to PLOTS/MLE_New/feature_importance_comprehensive.png")
    
    def save_results(self):
        """Save results to JSON and CSV"""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        import os
        os.makedirs('Data/processed/MLE_New', exist_ok=True)
        
        # Save detailed JSON results
        with open('Data/processed/MLE_New/feature_importance_results.json', 'w') as f:
            json.dump(self.importance_results, f, indent=2)
        
        # Save combined scores to CSV
        combined = self.importance_results['combined']
        results_df = pd.DataFrame([
            {
                'variable': var,
                'combined_score': result['combined_score'],
                'num_methods': result['num_methods']
            }
            for var, result in combined.items()
        ])
        results_df.to_csv('Data/processed/MLE_New/feature_importance_ranking.csv', index=False)
        
        print("[OK] Results saved:")
        print("     - Data/processed/MLE_New/feature_importance_results.json")
        print("     - Data/processed/MLE_New/feature_importance_ranking.csv")

def main():
    """Main function to run feature importance analysis"""
    print("=" * 80)
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("FOR BIRTHWEIGHT PREDICTION")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer('Data/processed/cleaned_dataset_with_engineered_features.csv')
    
    # Load data
    analyzer.load_and_prepare_data()
    
    # Run all analysis methods
    analyzer.analyze_correlation()
    analyzer.analyze_random_forest()
    analyzer.analyze_xgboost()
    analyzer.analyze_mutual_information()
    analyzer.analyze_statistical_tests()
    
    # Combine methods
    analyzer.combine_all_methods()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS COMPLETED!")
    print("=" * 80)
    print("\nKey findings saved to:")
    print("  - Data/processed/MLE_New/feature_importance_results.json")
    print("  - Data/processed/MLE_New/feature_importance_ranking.csv")
    print("  - PLOTS/MLE_New/feature_importance_comprehensive.png")
    print("=" * 80)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()
