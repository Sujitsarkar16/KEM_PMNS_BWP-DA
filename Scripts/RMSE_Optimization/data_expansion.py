"""
Phase 1: Data Expansion and Variable Selection
==============================================

This script implements Phase 1 of the RMSE improvement plan:
- Step 1.1: Identify all available variables
- Step 1.2: Assess data quality for each variable
- Step 1.3: Select and filter variables

Target: Expand from 4 variables to 20+ variables
Expected Impact: 20-35% RMSE reduction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load the cleaned dataset and explore its structure"""
    print("="*60)
    print("PHASE 1: DATA EXPANSION AND VARIABLE SELECTION")
    print("="*60)
    
    # Load data
    data = pd.read_csv('Data/processed/cleaned_dataset_with_engineered_features.csv')
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Check current MLE results
    mle_metrics = pd.read_csv('Data/processed/mle_performance_metrics.csv')
    print(f"\nCurrent MLE Performance:")
    print(f"RMSE: {mle_metrics['RMSE'].iloc[0]:.2f} grams")
    print(f"MAE: {mle_metrics['MAE'].iloc[0]:.2f} grams")
    print(f"R²: {mle_metrics['R²'].iloc[0]:.4f}")
    print(f"Sample Size: {mle_metrics['Sample_Size'].iloc[0]}")
    
    return data

def identify_all_variables(data):
    """Step 1.1: Identify all available variables for birthweight prediction"""
    print("\n" + "="*50)
    print("STEP 1.1: IDENTIFY ALL AVAILABLE VARIABLES")
    print("="*50)
    
    # Define variable categories based on the implementation plan
    variable_categories = {
        'MATERNAL_DEMOGRAPHICS': [
            'f0_m_age', 'f0_m_edu', 'f0_f_edu', 'f0_occ_hou_head', 
            'f0_socio_eco_sc', 'f0_caste_fly'
        ],
        'MATERNAL_ANTHROPOMETRY': [
            'f0_m_ht', 'f0_m_wt_prepreg', 'f0_m_bmi_prepreg',
            'f0_m_waist_prepreg', 'f0_m_hip_prepreg',
            'f0_m_tr_prepreg', 'f0_m_bi_prepreg', 'f0_m_ss_prepreg',
            'f0_m_su_prepreg', 'f0_m_ma_prepreg'
        ],
        'PREGNANCY_HISTORY': [
            'f0_m_gravida_v1', 'f0_m_parity_v1', 'f0_m_abor_v1',
            'f0_m_liv_male_v1', 'f0_m_liv_female_v1',
            'f0_m_still_birth_v1', 'f0_m_neo_death_v1'
        ],
        'MATERNAL_HEALTH_V1': [
            'f0_m_hb_v1', 'f0_m_glu_f_v1', 'f0_m_sys_bp_r1_v1',
            'f0_m_dia_bp_r1_v1', 'f0_m_wbc_v1', 'f0_m_rbc_v1',
            'f0_m_plt_v1', 'f0_m_hct_v1', 'f0_m_b12_v1', 'f0_m_fer_v1'
        ],
        'MATERNAL_HEALTH_V2': [
            'f0_m_hb_v2', 'f0_m_glu_f_v2', 'f0_m_sys_bp_r1_v2',
            'f0_m_dia_bp_r1_v2', 'f0_m_wbc_v2', 'f0_m_rbc_v2',
            'f0_m_plt_v2', 'f0_m_hct_v2', 'f0_m_b12_v2', 'f0_m_fer_v2'
        ],
        'GESTATIONAL_FACTORS': [
            'f0_m_GA_V1', 'f0_m_GA_V2', 'f0_m_GA_Del', 'f0_m_plac_wt'
        ],
        'CHILD_FACTORS': [
            'f1_sex', 'f0_m_del_mode'
        ],
        'NUTRITIONAL_FACTORS': [
            'f0_m_totcal_v1', 'f0_m_totcal_v2', 'f0_m_totpro_v1', 'f0_m_totpro_v2',
            'f0_m_totfat_v1', 'f0_m_totfat_v2', 'f0_m_totiron_v1', 'f0_m_totiron_v2'
        ]
    }
    
    # Check which variables exist in the dataset
    available_vars = {}
    missing_vars = {}
    
    for category, vars_list in variable_categories.items():
        available = []
        missing = []
        
        for var in vars_list:
            if var in data.columns:
                available.append(var)
            else:
                missing.append(var)
        
        available_vars[category] = available
        missing_vars[category] = missing
    
    # Print results
    print("Variable Availability Check:")
    print("-" * 40)
    for category, vars_list in available_vars.items():
        print(f"\n{category}:")
        print(f"  Available: {len(vars_list)}/{len(variable_categories[category])}")
        if vars_list:
            print(f"  Variables: {vars_list}")
        if missing_vars[category]:
            print(f"  Missing: {missing_vars[category]}")
    
    # Get all available variables
    all_available = []
    for vars_list in available_vars.values():
        all_available.extend(vars_list)
    
    print(f"\nTotal Available Variables: {len(all_available)}")
    print(f"Target: 20+ variables")
    print(f"Current MLE used: 4 variables")
    
    return available_vars, all_available

def assess_data_quality(data, variables):
    """Step 1.2: Assess data quality for each variable"""
    print("\n" + "="*50)
    print("STEP 1.2: DATA QUALITY ASSESSMENT")
    print("="*50)
    
    quality_metrics = {}
    
    for var in variables:
        if var in data.columns:
            # Basic statistics
            total_count = len(data)
            non_null_count = data[var].count()
            null_count = total_count - non_null_count
            missing_pct = (null_count / total_count) * 100
            
            # Data type
            dtype = data[var].dtype
            
            # For numeric variables
            if pd.api.types.is_numeric_dtype(data[var]):
                # Outlier detection using IQR
                Q1 = data[var].quantile(0.25)
                Q3 = data[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[var] < lower_bound) | (data[var] > upper_bound)]
                outlier_pct = (len(outliers) / non_null_count) * 100 if non_null_count > 0 else 0
                
                # Skewness
                skewness = data[var].skew()
                
                quality_metrics[var] = {
                    'missing_pct': missing_pct,
                    'outlier_pct': outlier_pct,
                    'skewness': abs(skewness),
                    'dtype': dtype,
                    'mean': data[var].mean(),
                    'std': data[var].std(),
                    'min': data[var].min(),
                    'max': data[var].max()
                }
            else:
                # For categorical variables
                unique_count = data[var].nunique()
                most_common_pct = (data[var].value_counts().iloc[0] / non_null_count) * 100 if non_null_count > 0 else 0
                
                quality_metrics[var] = {
                    'missing_pct': missing_pct,
                    'unique_count': unique_count,
                    'most_common_pct': most_common_pct,
                    'dtype': dtype
                }
    
    # Create quality summary
    quality_df = pd.DataFrame(quality_metrics).T
    
    # Filter criteria
    print("\nData Quality Summary:")
    print("-" * 40)
    print(f"Variables with <50% missing data: {len(quality_df[quality_df['missing_pct'] < 50])}")
    print(f"Variables with <30% missing data: {len(quality_df[quality_df['missing_pct'] < 30])}")
    print(f"Variables with <10% missing data: {len(quality_df[quality_df['missing_pct'] < 10])}")
    
    # Show variables by missing data percentage
    print("\nMissing Data Analysis:")
    print("-" * 30)
    missing_analysis = quality_df[['missing_pct']].sort_values('missing_pct')
    print(missing_analysis.head(20))
    
    return quality_df

def select_variables(quality_df, target_missing_threshold=50):
    """Step 1.3: Select variables based on quality criteria"""
    print("\n" + "="*50)
    print("STEP 1.3: VARIABLE SELECTION AND FILTERING")
    print("="*50)
    
    # Filter variables based on missing data threshold
    selected_vars = quality_df[quality_df['missing_pct'] < target_missing_threshold].index.tolist()
    
    print(f"Variables selected (missing < {target_missing_threshold}%): {len(selected_vars)}")
    
    # Categorize selected variables
    selected_categories = {
        'DEMOGRAPHICS': [var for var in selected_vars if var.startswith(('f0_m_age', 'f0_m_edu', 'f0_f_edu', 'f0_socio_eco_sc', 'f0_caste_fly'))],
        'ANTHROPOMETRY': [var for var in selected_vars if var.startswith(('f0_m_ht', 'f0_m_wt', 'f0_m_bmi', 'f0_m_waist', 'f0_m_hip', 'f0_m_tr', 'f0_m_bi', 'f0_m_ss'))],
        'PREGNANCY_HISTORY': [var for var in selected_vars if var.startswith(('f0_m_gravida', 'f0_m_parity', 'f0_m_abor', 'f0_m_liv', 'f0_m_still', 'f0_m_neo'))],
        'HEALTH_MARKERS': [var for var in selected_vars if var.startswith(('f0_m_hb', 'f0_m_glu', 'f0_m_sys_bp', 'f0_m_dia_bp', 'f0_m_wbc', 'f0_m_rbc', 'f0_m_plt', 'f0_m_hct', 'f0_m_b12', 'f0_m_fer'))],
        'GESTATIONAL': [var for var in selected_vars if var.startswith(('f0_m_GA', 'f0_m_plac'))],
        'CHILD_FACTORS': [var for var in selected_vars if var.startswith(('f1_sex', 'f0_m_del'))],
        'NUTRITIONAL': [var for var in selected_vars if var.startswith(('f0_m_totcal', 'f0_m_totpro', 'f0_m_totfat', 'f0_m_totiron'))]
    }
    
    print("\nSelected Variables by Category:")
    print("-" * 40)
    for category, vars_list in selected_categories.items():
        print(f"{category}: {len(vars_list)} variables")
        if vars_list:
            print(f"  {vars_list}")
    
    # Check correlation with target variable
    print("\nCorrelation with Birthweight (f1_bw):")
    print("-" * 40)
    
    # Load data for correlation analysis
    data = pd.read_csv('Data/processed/cleaned_dataset_with_engineered_features.csv')
    
    correlations = {}
    for var in selected_vars:
        if var in data.columns and 'f1_bw' in data.columns:
            corr = data[var].corr(data['f1_bw'])
            if not pd.isna(corr):
                correlations[var] = corr
    
    # Sort by absolute correlation
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("Top 20 variables by correlation with birthweight:")
    for var, corr in sorted_correlations[:20]:
        print(f"  {var}: {corr:.4f}")
    
    return selected_vars, selected_categories, correlations

def create_quality_visualizations(quality_df, selected_vars, correlations):
    """Create visualizations for data quality assessment"""
    print("\nCreating data quality visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Missing data distribution
    missing_pct = quality_df['missing_pct'].sort_values(ascending=False)
    axes[0, 0].bar(range(len(missing_pct)), missing_pct.values)
    axes[0, 0].axhline(y=50, color='r', linestyle='--', label='50% threshold')
    axes[0, 0].axhline(y=30, color='orange', linestyle='--', label='30% threshold')
    axes[0, 0].set_xlabel('Variables')
    axes[0, 0].set_ylabel('Missing Data Percentage')
    axes[0, 0].set_title('Missing Data Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Selected vs All variables
    selected_count = len(selected_vars)
    total_count = len(quality_df)
    axes[0, 1].pie([selected_count, total_count - selected_count], 
                   labels=['Selected', 'Excluded'], 
                   autopct='%1.1f%%',
                   colors=['lightgreen', 'lightcoral'])
    axes[0, 1].set_title(f'Variable Selection\n({selected_count}/{total_count} selected)')
    
    # 3. Correlation with birthweight
    if correlations:
        corr_values = list(correlations.values())
        corr_vars = list(correlations.keys())
        
        # Sort by absolute correlation
        sorted_pairs = sorted(zip(corr_vars, corr_values), key=lambda x: abs(x[1]), reverse=True)
        top_vars = [pair[0] for pair in sorted_pairs[:15]]
        top_corrs = [pair[1] for pair in sorted_pairs[:15]]
        
        axes[1, 0].barh(range(len(top_vars)), top_corrs)
        axes[1, 0].set_yticks(range(len(top_vars)))
        axes[1, 0].set_yticklabels(top_vars, fontsize=8)
        axes[1, 0].set_xlabel('Correlation with Birthweight')
        axes[1, 0].set_title('Top 15 Variables by Correlation')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Missing data vs Correlation
    if correlations:
        missing_corr_data = []
        corr_data = []
        
        for var in selected_vars:
            if var in quality_df.index and var in correlations:
                missing_corr_data.append(quality_df.loc[var, 'missing_pct'])
                corr_data.append(abs(correlations[var]))
        
        if missing_corr_data and corr_data:
            axes[1, 1].scatter(missing_corr_data, corr_data, alpha=0.6)
            axes[1, 1].set_xlabel('Missing Data Percentage')
            axes[1, 1].set_ylabel('Absolute Correlation with Birthweight')
            axes[1, 1].set_title('Missing Data vs Correlation')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('PLOTS/MLE_Improved/phase1_data_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to PLOTS/MLE_Improved/phase1_data_quality_analysis.png")

def save_phase1_results(selected_vars, selected_categories, correlations, quality_df):
    """Save Phase 1 results for next phases"""
    print("\nSaving Phase 1 results...")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('Data/processed/MLE_Improved', exist_ok=True)
    os.makedirs('PLOTS/MLE_Improved', exist_ok=True)
    
    # Save selected variables
    selected_vars_df = pd.DataFrame({
        'variable': selected_vars,
        'missing_pct': [quality_df.loc[var, 'missing_pct'] for var in selected_vars],
        'correlation_with_bw': [correlations.get(var, 0) for var in selected_vars]
    })
    selected_vars_df.to_csv('Data/processed/MLE_Improved/phase1_selected_variables.csv', index=False)
    
    # Save quality metrics
    quality_df.to_csv('Data/processed/MLE_Improved/phase1_quality_metrics.csv')
    
    # Save categories
    categories_df = pd.DataFrame([
        {'category': cat, 'variables': ', '.join(vars_list), 'count': len(vars_list)}
        for cat, vars_list in selected_categories.items()
    ])
    categories_df.to_csv('Data/processed/MLE_Improved/phase1_variable_categories.csv', index=False)
    
    # Save summary
    summary = {
        'phase': 'Phase 1: Data Expansion and Variable Selection',
        'total_variables_available': len(quality_df),
        'variables_selected': len(selected_vars),
        'selection_criteria': 'Missing data < 50%',
        'categories': {cat: len(vars_list) for cat, vars_list in selected_categories.items()},
        'expected_impact': '20-35% RMSE reduction'
    }
    
    import json
    with open('Data/processed/MLE_Improved/phase1_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Phase 1 results saved successfully!")
    print(f"Selected {len(selected_vars)} variables for next phases")

def main():
    """Main function to run Phase 1"""
    # Load and explore data
    data = load_and_explore_data()
    
    # Step 1.1: Identify all variables
    available_vars, all_available = identify_all_variables(data)
    
    # Step 1.2: Assess data quality
    quality_df = assess_data_quality(data, all_available)
    
    # Step 1.3: Select variables
    selected_vars, selected_categories, correlations = select_variables(quality_df)
    
    # Create visualizations
    create_quality_visualizations(quality_df, selected_vars, correlations)
    
    # Save results
    save_phase1_results(selected_vars, selected_categories, correlations, quality_df)
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"✅ Identified {len(all_available)} available variables")
    print(f"✅ Selected {len(selected_vars)} high-quality variables")
    print(f"✅ Categorized variables into {len(selected_categories)} groups")
    print(f"✅ Ready for Phase 2: Feature Engineering")
    print("="*60)

if __name__ == "__main__":
    main()
