"""
Phase 2: Feature Engineering
============================

This script implements Phase 2 of the RMSE improvement plan:
- Step 2.1: Create interaction terms
- Step 2.2: Create composite scores
- Step 2.3: Create ratio features
- Step 2.4: Handle categorical variables

Target: Create meaningful engineered features
Expected Impact: 10-20% RMSE reduction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

def load_phase1_results():
    """Load Phase 1 results and data"""
    print("="*60)
    print("PHASE 2: FEATURE ENGINEERING")
    print("="*60)
    
    # Load data
    data = pd.read_csv('Data/processed/cleaned_dataset_with_engineered_features.csv')
    print(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Load Phase 1 results
    selected_vars_df = pd.read_csv('Data/processed/MLE_Improved/phase1_selected_variables.csv')
    selected_vars = selected_vars_df['variable'].tolist()
    print(f"Phase 1 selected variables: {len(selected_vars)}")
    
    # Load categories
    categories_df = pd.read_csv('Data/processed/MLE_Improved/phase1_variable_categories.csv')
    print(f"Variable categories: {len(categories_df)}")
    
    return data, selected_vars, categories_df

def create_interaction_terms(data, selected_vars):
    """Step 2.1: Create interaction terms to capture non-linear relationships"""
    print("\n" + "="*50)
    print("STEP 2.1: CREATE INTERACTION TERMS")
    print("="*50)
    
    interaction_terms = {}
    
    # BMI × Height interaction
    if 'f0_m_bmi_prepreg' in data.columns and 'f0_m_ht' in data.columns:
        data['bmi_height_interaction'] = data['f0_m_bmi_prepreg'] * data['f0_m_ht']
        interaction_terms['bmi_height_interaction'] = 'BMI × Height'
        print("+ Created BMI x Height interaction")
    
    # Age × Parity interaction
    if 'f0_m_age' in data.columns and 'f0_m_parity_v1' in data.columns:
        data['age_parity_interaction'] = data['f0_m_age'] * data['f0_m_parity_v1']
        interaction_terms['age_parity_interaction'] = 'Age × Parity'
        print("+ Created Age x Parity interaction")
    
    # Education × Socioeconomic interaction
    if 'f0_m_edu' in data.columns and 'f0_socio_eco_sc' in data.columns:
        data['edu_socio_interaction'] = data['f0_m_edu'] * data['f0_socio_eco_sc']
        interaction_terms['edu_socio_interaction'] = 'Education × Socioeconomic'
        print("+ Created Education × Socioeconomic interaction")
    
    # Hemoglobin × Gestational age interaction
    if 'f0_m_hb_v1' in data.columns and 'f0_m_GA_V1' in data.columns:
        data['hb_ga_interaction'] = data['f0_m_hb_v1'] * data['f0_m_GA_V1']
        interaction_terms['hb_ga_interaction'] = 'Hemoglobin × Gestational Age'
        print("+ Created Hemoglobin × Gestational Age interaction")
    
    # Weight × Height interaction
    if 'f0_m_wt_prepreg' in data.columns and 'f0_m_ht' in data.columns:
        data['wt_ht_interaction'] = data['f0_m_wt_prepreg'] * data['f0_m_ht']
        interaction_terms['wt_ht_interaction'] = 'Weight × Height'
        print("+ Created Weight × Height interaction")
    
    # Additional meaningful interactions
    # BMI × Age interaction
    if 'f0_m_bmi_prepreg' in data.columns and 'f0_m_age' in data.columns:
        data['bmi_age_interaction'] = data['f0_m_bmi_prepreg'] * data['f0_m_age']
        interaction_terms['bmi_age_interaction'] = 'BMI × Age'
        print("+ Created BMI × Age interaction")
    
    # Parity × Gravida interaction
    if 'f0_m_parity_v1' in data.columns and 'f0_m_gravida_v1' in data.columns:
        data['parity_gravida_interaction'] = data['f0_m_parity_v1'] * data['f0_m_gravida_v1']
        interaction_terms['parity_gravida_interaction'] = 'Parity × Gravida'
        print("+ Created Parity × Gravida interaction")
    
    # Hemoglobin × BMI interaction
    if 'f0_m_hb_v1' in data.columns and 'f0_m_bmi_prepreg' in data.columns:
        data['hb_bmi_interaction'] = data['f0_m_hb_v1'] * data['f0_m_bmi_prepreg']
        interaction_terms['hb_bmi_interaction'] = 'Hemoglobin × BMI'
        print("+ Created Hemoglobin × BMI interaction")
    
    print(f"\nTotal interaction terms created: {len(interaction_terms)}")
    return data, interaction_terms

def create_composite_scores(data, selected_vars):
    """Step 2.2: Create composite scores to combine related variables"""
    print("\n" + "="*50)
    print("STEP 2.2: CREATE COMPOSITE SCORES")
    print("="*50)
    
    composite_scores = {}
    
    # Maternal Health Index (normalize to 0-100)
    health_components = ['f0_m_hb_v1', 'f0_m_glu_f_v1', 'f0_m_sys_bp_r1_v1']
    available_health = [col for col in health_components if col in data.columns]
    
    if len(available_health) >= 2:
        # Normalize each component to 0-100 scale
        for col in available_health:
            if data[col].std() > 0:  # Avoid division by zero
                data[f'{col}_normalized'] = ((data[col] - data[col].min()) / 
                                           (data[col].max() - data[col].min())) * 100
        
        # Create composite score
        normalized_cols = [f'{col}_normalized' for col in available_health]
        data['maternal_health_index'] = data[normalized_cols].mean(axis=1)
        composite_scores['maternal_health_index'] = f'Maternal Health Index ({len(available_health)} components)'
        print(f"+ Created Maternal Health Index with {len(available_health)} components")
    
    # Nutritional Status Score
    nutrition_components = ['f0_m_bmi_prepreg', 'f0_m_wt_prepreg', 'f0_m_ht']
    available_nutrition = [col for col in nutrition_components if col in data.columns]
    
    if len(available_nutrition) >= 2:
        # Normalize and create composite
        for col in available_nutrition:
            if data[col].std() > 0:
                data[f'{col}_normalized'] = ((data[col] - data[col].min()) / 
                                           (data[col].max() - data[col].min())) * 100
        
        normalized_cols = [f'{col}_normalized' for col in available_nutrition]
        data['nutritional_status'] = data[normalized_cols].mean(axis=1)
        composite_scores['nutritional_status'] = f'Nutritional Status ({len(available_nutrition)} components)'
        print(f"+ Created Nutritional Status Score with {len(available_nutrition)} components")
    
    # Pregnancy Risk Score (higher = more risk)
    risk_components = ['f0_m_age', 'f0_m_parity_v1', 'f0_m_abor_v1']
    available_risk = [col for col in risk_components if col in data.columns]
    
    if len(available_risk) >= 2:
        # Normalize and sum (higher values = more risk)
        for col in available_risk:
            if data[col].std() > 0:
                data[f'{col}_normalized'] = ((data[col] - data[col].min()) / 
                                           (data[col].max() - data[col].min())) * 100
        
        normalized_cols = [f'{col}_normalized' for col in available_risk]
        data['pregnancy_risk_score'] = data[normalized_cols].sum(axis=1)
        composite_scores['pregnancy_risk_score'] = f'Pregnancy Risk Score ({len(available_risk)} components)'
        print(f"+ Created Pregnancy Risk Score with {len(available_risk)} components")
    
    # Anthropometric Index
    anthro_components = ['f0_m_waist_prepreg', 'f0_m_hip_prepreg']
    available_anthro = [col for col in anthro_components if col in data.columns]
    
    if len(available_anthro) >= 2:
        # Normalize and create composite
        for col in available_anthro:
            if data[col].std() > 0:
                data[f'{col}_normalized'] = ((data[col] - data[col].min()) / 
                                           (data[col].max() - data[col].min())) * 100
        
        normalized_cols = [f'{col}_normalized' for col in available_anthro]
        data['anthropometric_index'] = data[normalized_cols].mean(axis=1)
        composite_scores['anthropometric_index'] = f'Anthropometric Index ({len(available_anthro)} components)'
        print(f"+ Created Anthropometric Index with {len(available_anthro)} components")
    
    # Gestational Health Index
    ga_components = ['f0_m_GA_V1', 'f0_m_GA_V2', 'f0_m_GA_Del']
    available_ga = [col for col in ga_components if col in data.columns]
    
    if len(available_ga) >= 2:
        # Normalize and create composite
        for col in available_ga:
            if data[col].std() > 0:
                data[f'{col}_normalized'] = ((data[col] - data[col].min()) / 
                                           (data[col].max() - data[col].min())) * 100
        
        normalized_cols = [f'{col}_normalized' for col in available_ga]
        data['gestational_health_index'] = data[normalized_cols].mean(axis=1)
        composite_scores['gestational_health_index'] = f'Gestational Health Index ({len(available_ga)} components)'
        print(f"+ Created Gestational Health Index with {len(available_ga)} components")
    
    print(f"\nTotal composite scores created: {len(composite_scores)}")
    return data, composite_scores

def create_ratio_features(data, selected_vars):
    """Step 2.3: Create ratio features for meaningful relationships"""
    print("\n" + "="*50)
    print("STEP 2.3: CREATE RATIO FEATURES")
    print("="*50)
    
    ratio_features = {}
    
    # Waist-to-Hip ratio
    if 'f0_m_waist_prepreg' in data.columns and 'f0_m_hip_prepreg' in data.columns:
        data['waist_hip_ratio'] = data['f0_m_waist_prepreg'] / data['f0_m_hip_prepreg']
        ratio_features['waist_hip_ratio'] = 'Waist-to-Hip Ratio'
        print("+ Created Waist-to-Hip ratio")
    
    # Weight-to-Height ratio
    if 'f0_m_wt_prepreg' in data.columns and 'f0_m_ht' in data.columns:
        data['wt_ht_ratio'] = data['f0_m_wt_prepreg'] / data['f0_m_ht']
        ratio_features['wt_ht_ratio'] = 'Weight-to-Height Ratio'
        print("+ Created Weight-to-Height ratio")
    
    # Hemoglobin-to-Gestational age ratio
    if 'f0_m_hb_v1' in data.columns and 'f0_m_GA_V1' in data.columns:
        data['hb_ga_ratio'] = data['f0_m_hb_v1'] / data['f0_m_GA_V1']
        ratio_features['hb_ga_ratio'] = 'Hemoglobin-to-Gestational Age Ratio'
        print("+ Created Hemoglobin-to-Gestational Age ratio")
    
    # BMI-to-Age ratio
    if 'f0_m_bmi_prepreg' in data.columns and 'f0_m_age' in data.columns:
        data['bmi_age_ratio'] = data['f0_m_bmi_prepreg'] / data['f0_m_age']
        ratio_features['bmi_age_ratio'] = 'BMI-to-Age Ratio'
        print("+ Created BMI-to-Age ratio")
    
    # Parity-to-Gravida ratio
    if 'f0_m_parity_v1' in data.columns and 'f0_m_gravida_v1' in data.columns:
        # Avoid division by zero
        data['parity_gravida_ratio'] = np.where(data['f0_m_gravida_v1'] > 0, 
                                               data['f0_m_parity_v1'] / data['f0_m_gravida_v1'], 0)
        ratio_features['parity_gravida_ratio'] = 'Parity-to-Gravida Ratio'
        print("+ Created Parity-to-Gravida ratio")
    
    # Height-to-Age ratio
    if 'f0_m_ht' in data.columns and 'f0_m_age' in data.columns:
        data['ht_age_ratio'] = data['f0_m_ht'] / data['f0_m_age']
        ratio_features['ht_age_ratio'] = 'Height-to-Age Ratio'
        print("+ Created Height-to-Age ratio")
    
    # Hemoglobin-to-BMI ratio
    if 'f0_m_hb_v1' in data.columns and 'f0_m_bmi_prepreg' in data.columns:
        data['hb_bmi_ratio'] = data['f0_m_hb_v1'] / data['f0_m_bmi_prepreg']
        ratio_features['hb_bmi_ratio'] = 'Hemoglobin-to-BMI Ratio'
        print("+ Created Hemoglobin-to-BMI ratio")
    
    # Placental weight-to-Birth weight ratio
    if 'f0_m_plac_wt' in data.columns and 'f1_bw' in data.columns:
        data['plac_bw_ratio'] = data['f0_m_plac_wt'] / data['f1_bw']
        ratio_features['plac_bw_ratio'] = 'Placental Weight-to-Birth Weight Ratio'
        print("+ Created Placental Weight-to-Birth Weight ratio")
    
    print(f"\nTotal ratio features created: {len(ratio_features)}")
    return data, ratio_features

def handle_categorical_variables(data, selected_vars):
    """Step 2.4: Handle categorical variables properly"""
    print("\n" + "="*50)
    print("STEP 2.4: HANDLE CATEGORICAL VARIABLES")
    print("="*50)
    
    categorical_encodings = {}
    
    # Identify categorical variables
    categorical_vars = []
    for var in selected_vars:
        if var in data.columns:
            if data[var].dtype == 'object' or data[var].nunique() < 10:
                categorical_vars.append(var)
    
    print(f"Identified categorical variables: {categorical_vars}")
    
    # Handle each categorical variable
    for var in categorical_vars:
        if var in data.columns:
            unique_values = data[var].nunique()
            print(f"\nProcessing {var}: {unique_values} unique values")
            
            if unique_values <= 5:  # Small number of categories - use one-hot encoding
                # Create dummy variables
                dummies = pd.get_dummies(data[var], prefix=var, dummy_na=True)
                data = pd.concat([data, dummies], axis=1)
                
                # Store encoding info
                categorical_encodings[var] = {
                    'method': 'one_hot',
                    'columns': dummies.columns.tolist()
                }
                print(f"  + One-hot encoded into {len(dummies.columns)} columns")
                
            else:  # Many categories - use label encoding
                le = LabelEncoder()
                data[f'{var}_encoded'] = le.fit_transform(data[var].astype(str))
                
                # Store encoding info
                categorical_encodings[var] = {
                    'method': 'label',
                    'column': f'{var}_encoded',
                    'classes': le.classes_.tolist()
                }
                print(f"  + Label encoded into {var}_encoded")
    
    # Special handling for specific variables
    # Child sex (binary)
    if 'f1_sex' in data.columns:
        if data['f1_sex'].dtype == 'object':
            data['f1_sex_male'] = (data['f1_sex'] == 'Male').astype(int)
            data['f1_sex_female'] = (data['f1_sex'] == 'Female').astype(int)
            categorical_encodings['f1_sex'] = {
                'method': 'binary',
                'columns': ['f1_sex_male', 'f1_sex_female']
            }
            print("+ Created binary encoding for child sex")
    
    # Delivery mode
    if 'f0_m_del_mode' in data.columns:
        if data['f0_m_del_mode'].dtype == 'object':
            data['f0_m_del_mode_encoded'] = LabelEncoder().fit_transform(data['f0_m_del_mode'].astype(str))
            categorical_encodings['f0_m_del_mode'] = {
                'method': 'label',
                'column': 'f0_m_del_mode_encoded'
            }
            print("+ Created label encoding for delivery mode")
    
    print(f"\nTotal categorical variables processed: {len(categorical_encodings)}")
    return data, categorical_encodings

def create_polynomial_features(data, selected_vars):
    """Create polynomial features for key variables"""
    print("\n" + "="*50)
    print("CREATE POLYNOMIAL FEATURES")
    print("="*50)
    
    polynomial_features = {}
    
    # Key variables for polynomial features
    key_vars = ['f0_m_age', 'f0_m_bmi_prepreg', 'f0_m_ht', 'f0_m_wt_prepreg', 'f0_m_GA_Del']
    
    for var in key_vars:
        if var in data.columns:
            # Square term
            data[f'{var}_squared'] = data[var] ** 2
            polynomial_features[f'{var}_squared'] = f'{var} squared'
            
            # Square root term (if all values are positive)
            if (data[var] >= 0).all():
                data[f'{var}_sqrt'] = np.sqrt(data[var])
                polynomial_features[f'{var}_sqrt'] = f'{var} square root'
            
            print(f"+ Created polynomial features for {var}")
    
    print(f"\nTotal polynomial features created: {len(polynomial_features)}")
    return data, polynomial_features

def create_feature_engineering_visualizations(data, interaction_terms, composite_scores, ratio_features):
    """Create visualizations for feature engineering results"""
    print("\nCreating feature engineering visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Feature engineering summary
    feature_types = ['Original', 'Interactions', 'Composite Scores', 'Ratios', 'Polynomial']
    feature_counts = [
        len([col for col in data.columns if not any(x in col for x in ['_interaction', '_index', '_ratio', '_squared', '_sqrt', '_encoded', '_normalized'])]),
        len(interaction_terms),
        len(composite_scores),
        len(ratio_features),
        len([col for col in data.columns if any(x in col for x in ['_squared', '_sqrt'])])
    ]
    
    axes[0, 0].bar(feature_types, feature_counts, color=['blue', 'green', 'orange', 'red', 'purple'])
    axes[0, 0].set_title('Feature Engineering Summary')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(feature_counts):
        axes[0, 0].text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # 2. Interaction terms correlation with target
    if interaction_terms and 'f1_bw' in data.columns:
        interaction_corrs = []
        interaction_names = []
        
        for term in interaction_terms.keys():
            if term in data.columns:
                corr = data[term].corr(data['f1_bw'])
                if not pd.isna(corr):
                    interaction_corrs.append(abs(corr))
                    interaction_names.append(term.replace('_interaction', ''))
        
        if interaction_corrs:
            axes[0, 1].barh(range(len(interaction_names)), interaction_corrs)
            axes[0, 1].set_yticks(range(len(interaction_names)))
            axes[0, 1].set_yticklabels(interaction_names, fontsize=8)
            axes[0, 1].set_xlabel('Absolute Correlation with Birthweight')
            axes[0, 1].set_title('Interaction Terms Correlation')
    
    # 3. Composite scores distribution
    if composite_scores and 'f1_bw' in data.columns:
        composite_corrs = []
        composite_names = []
        
        for score in composite_scores.keys():
            if score in data.columns:
                corr = data[score].corr(data['f1_bw'])
                if not pd.isna(corr):
                    composite_corrs.append(abs(corr))
                    composite_names.append(score.replace('_index', '').replace('_score', ''))
        
        if composite_corrs:
            axes[0, 2].barh(range(len(composite_names)), composite_corrs)
            axes[0, 2].set_yticks(range(len(composite_names)))
            axes[0, 2].set_yticklabels(composite_names, fontsize=8)
            axes[0, 2].set_xlabel('Absolute Correlation with Birthweight')
            axes[0, 2].set_title('Composite Scores Correlation')
    
    # 4. Ratio features correlation
    if ratio_features and 'f1_bw' in data.columns:
        ratio_corrs = []
        ratio_names = []
        
        for ratio in ratio_features.keys():
            if ratio in data.columns:
                corr = data[ratio].corr(data['f1_bw'])
                if not pd.isna(corr):
                    ratio_corrs.append(abs(corr))
                    ratio_names.append(ratio.replace('_ratio', ''))
        
        if ratio_corrs:
            axes[1, 0].barh(range(len(ratio_names)), ratio_corrs)
            axes[1, 0].set_yticks(range(len(ratio_names)))
            axes[1, 0].set_yticklabels(ratio_names, fontsize=8)
            axes[1, 0].set_xlabel('Absolute Correlation with Birthweight')
            axes[1, 0].set_title('Ratio Features Correlation')
    
    # 5. Feature importance by type
    all_features = list(interaction_terms.keys()) + list(composite_scores.keys()) + list(ratio_features.keys())
    if all_features and 'f1_bw' in data.columns:
        feature_corrs = []
        for feature in all_features:
            if feature in data.columns:
                corr = data[feature].corr(data['f1_bw'])
                if not pd.isna(corr):
                    feature_corrs.append(abs(corr))
        
        if feature_corrs:
            axes[1, 1].hist(feature_corrs, bins=10, alpha=0.7, edgecolor='black')
            axes[1, 1].set_xlabel('Absolute Correlation with Birthweight')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Engineered Features Correlation Distribution')
    
    # 6. Total features count
    total_original = len([col for col in data.columns if not any(x in col for x in ['_interaction', '_index', '_ratio', '_squared', '_sqrt', '_encoded', '_normalized'])])
    total_engineered = len(data.columns) - total_original
    
    axes[1, 2].pie([total_original, total_engineered], 
                   labels=['Original Features', 'Engineered Features'], 
                   autopct='%1.1f%%',
                   colors=['lightblue', 'lightgreen'])
    axes[1, 2].set_title(f'Feature Count\nTotal: {len(data.columns)} features')
    
    plt.tight_layout()
    plt.savefig('PLOTS/MLE_Improved/phase2_feature_engineering_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved to PLOTS/MLE_Improved/phase2_feature_engineering_analysis.png")

def save_phase2_results(data, interaction_terms, composite_scores, ratio_features, categorical_encodings, polynomial_features):
    """Save Phase 2 results for next phases"""
    print("\nSaving Phase 2 results...")
    
    # Create results directory if it doesn't exist
    import os
    os.makedirs('Data/processed/MLE_Improved', exist_ok=True)
    
    # Save the engineered dataset
    data.to_csv('Data/processed/MLE_Improved/phase2_engineered_dataset.csv', index=False)
    print(f"+ Saved engineered dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Save feature engineering summary
    feature_summary = {
        'phase': 'Phase 2: Feature Engineering',
        'original_features': len([col for col in data.columns if not any(x in col for x in ['_interaction', '_index', '_ratio', '_squared', '_sqrt', '_encoded', '_normalized'])]),
        'total_features': len(data.columns),
        'engineered_features': len(data.columns) - len([col for col in data.columns if not any(x in col for x in ['_interaction', '_index', '_ratio', '_squared', '_sqrt', '_encoded', '_normalized'])]),
        'interaction_terms': len(interaction_terms),
        'composite_scores': len(composite_scores),
        'ratio_features': len(ratio_features),
        'categorical_encodings': len(categorical_encodings),
        'polynomial_features': len(polynomial_features),
        'expected_impact': '10-20% RMSE reduction'
    }
    
    import json
    with open('Data/processed/MLE_Improved/phase2_summary.json', 'w') as f:
        json.dump(feature_summary, f, indent=2)
    
    # Save detailed feature information
    feature_details = {
        'interaction_terms': interaction_terms,
        'composite_scores': composite_scores,
        'ratio_features': ratio_features,
        'categorical_encodings': categorical_encodings,
        'polynomial_features': polynomial_features
    }
    
    with open('Data/processed/MLE_Improved/phase2_feature_details.json', 'w') as f:
        json.dump(feature_details, f, indent=2)
    
    print("Phase 2 results saved successfully!")

def main():
    """Main function to run Phase 2"""
    # Load Phase 1 results
    data, selected_vars, categories_df = load_phase1_results()
    
    # Step 2.1: Create interaction terms
    data, interaction_terms = create_interaction_terms(data, selected_vars)
    
    # Step 2.2: Create composite scores
    data, composite_scores = create_composite_scores(data, selected_vars)
    
    # Step 2.3: Create ratio features
    data, ratio_features = create_ratio_features(data, selected_vars)
    
    # Step 2.4: Handle categorical variables
    data, categorical_encodings = handle_categorical_variables(data, selected_vars)
    
    # Create polynomial features
    data, polynomial_features = create_polynomial_features(data, selected_vars)
    
    # Create visualizations
    create_feature_engineering_visualizations(data, interaction_terms, composite_scores, ratio_features)
    
    # Save results
    save_phase2_results(data, interaction_terms, composite_scores, ratio_features, categorical_encodings, polynomial_features)
    
    print("\n" + "="*60)
    print("PHASE 2 COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"+ Original features: {len([col for col in data.columns if not any(x in col for x in ['_interaction', '_index', '_ratio', '_squared', '_sqrt', '_encoded', '_normalized'])])}")
    print(f"+ Total features: {len(data.columns)}")
    print(f"+ Engineered features: {len(data.columns) - len([col for col in data.columns if not any(x in col for x in ['_interaction', '_index', '_ratio', '_squared', '_sqrt', '_encoded', '_normalized'])])}")
    print(f"+ Interaction terms: {len(interaction_terms)}")
    print(f"+ Composite scores: {len(composite_scores)}")
    print(f"+ Ratio features: {len(ratio_features)}")
    print(f"+ Ready for Phase 3: Advanced ML Models")
    print("="*60)

if __name__ == "__main__":
    main()
