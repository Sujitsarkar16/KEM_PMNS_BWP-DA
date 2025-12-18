"""
Feature Engineering for Advanced Stacking Models
Creates powerful interaction features based on biological theory

Power Features:
1. Genetic_Volume = f0_m_ht * f0_f_head_cir_ini
   (Maternal height × Paternal head circumference)
   
2. Placental_Efficiency_Proxy = f0_m_plac_wt / f0_m_wt_prepreg
   (Placental weight / Pre-pregnancy maternal weight)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Paths
BASE_DIR = Path(r"e:\KEM\Project")
DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Advanced_Stacking_Models" / "Data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the cleaned engineered dataset"""
    data_path = DATA_DIR / "processed" / "cleaned_dataset_with_engineered_features.csv"
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    return df

def create_power_features(df):
    """Create the two power interaction features"""
    df_enhanced = df.copy()
    
    # 1. Genetic Envelope Interaction
    # Theory: Birth weight constrained by maternal height (uterine capacity) 
    # but driven by paternal genetics (skeletal potential)
    if 'f0_m_ht' in df.columns and 'f0_f_head_cir_ini' in df.columns:
        df_enhanced['Genetic_Volume'] = df['f0_m_ht'] * df['f0_f_head_cir_ini']
        print("✓ Created Genetic_Volume = f0_m_ht × f0_f_head_cir_ini")
    else:
        print("⚠ Warning: Could not create Genetic_Volume (missing features)")
    
    # 2. Placental Efficiency Proxy
    # Theory: Placental weight normalized by pre-pregnancy maternal weight 
    # gives biological efficiency score
    if 'f0_m_plac_wt' in df.columns and 'f0_m_wt_prepreg' in df.columns:
        # Avoid division by zero
        df_enhanced['Placental_Efficiency_Proxy'] = np.where(
            df['f0_m_wt_prepreg'] > 0,
            df['f0_m_plac_wt'] / df['f0_m_wt_prepreg'],
            np.nan
        )
        print("✓ Created Placental_Efficiency_Proxy = f0_m_plac_wt / f0_m_wt_prepreg")
    else:
        print("⚠ Warning: Could not create Placental_Efficiency_Proxy (missing features)")
    
    return df_enhanced

def generate_feature_statistics(df, feature_name):
    """Generate statistics for a feature"""
    if feature_name not in df.columns:
        return None
    
    feature_data = df[feature_name].dropna()
    if len(feature_data) == 0:
        return None
    
    return {
        'count': len(feature_data),
        'mean': float(feature_data.mean()),
        'std': float(feature_data.std()),
        'min': float(feature_data.min()),
        'q25': float(feature_data.quantile(0.25)),
        'median': float(feature_data.median()),
        'q75': float(feature_data.quantile(0.75)),
        'max': float(feature_data.max()),
        'missing': int(df[feature_name].isna().sum()),
        'missing_pct': float(df[feature_name].isna().sum() / len(df) * 100)
    }

def main():
    print("=" * 80)
    print("FEATURE ENGINEERING FOR ADVANCED STACKING MODELS")
    print("=" * 80)
    print()
    
    # Load data
    df = load_data()
    print()
    
    # Create power features
    print("Creating Power Interaction Features...")
    print("-" * 80)
    df_enhanced = create_power_features(df)
    print()
    
    # Generate statistics
    print("Generating Feature Statistics...")
    print("-" * 80)
    
    stats = {}
    
    # Statistics for base features
    base_features = {
        'f0_m_ht': 'Maternal Height',
        'f0_f_head_cir_ini': 'Paternal Head Circumference',
        'f0_m_plac_wt': 'Maternal Placental Weight',
        'f0_m_wt_prepreg': 'Maternal Pre-pregnancy Weight'
    }
    
    for feat, desc in base_features.items():
        feat_stats = generate_feature_statistics(df_enhanced, feat)
        if feat_stats:
            stats[feat] = {'description': desc, 'stats': feat_stats}
            print(f"✓ {desc} ({feat}): Mean={feat_stats['mean']:.2f}, Missing={feat_stats['missing']}")
    
    # Statistics for power features
    power_features = {
        'Genetic_Volume': 'Genetic Envelope (Maternal Ht × Paternal Head Circ)',
        'Placental_Efficiency_Proxy': 'Placental Efficiency (Placental Wt / Pre-preg Wt)'
    }
    
    for feat, desc in power_features.items():
        feat_stats = generate_feature_statistics(df_enhanced, feat)
        if feat_stats:
            stats[feat] = {'description': desc, 'stats': feat_stats}
            print(f"✓ {desc}: Mean={feat_stats['mean']:.4f}, Missing={feat_stats['missing']}")
    
    print()
    
    # Save enhanced dataset
    output_path = OUTPUT_DIR / "power_features_dataset.csv"
    df_enhanced.to_csv(output_path, index=False)
    print(f"✓ Saved enhanced dataset to: {output_path}")
    print(f"  Shape: {df_enhanced.shape}")
    print()
    
    # Save statistics
    stats_path = OUTPUT_DIR / "power_features_statistics.json"
    stats_output = {
        'timestamp': datetime.now().isoformat(),
        'original_shape': list(df.shape),
        'enhanced_shape': list(df_enhanced.shape),
        'new_features': list(power_features.keys()),
        'feature_statistics': stats
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"✓ Saved feature statistics to: {stats_path}")
    print()
    
    # Create a simplified dataset with power features + top features
    # This will be used for modeling
    print("Creating Modeling Dataset...")
    print("-" * 80)
    
    # Essential features for modeling
    essential_cols = [
        'f1_bw',  # Target variable
        'Genetic_Volume',
        'Placental_Efficiency_Proxy',
        # Add key predictors
        'f0_m_ht',
        'f0_m_wt_prepreg',
        'f0_m_bmi_prepreg',
        'f0_m_age',
        'f0_m_GA_Del',
        'f0_m_plac_wt',
        'f0_f_head_cir_ini',
        'f0_f_ht_ini',
        'f0_f_bmi_ini',
        'f0_m_hb_v1',
        'f0_m_hb_v2',
        'f0_m_fer_v1',
        'f0_m_fer_v2',
        'f0_m_b12_v1',
        'f0_m_b12_v2'
    ]
    
    # Filter to columns that exist
    available_cols = [col for col in essential_cols if col in df_enhanced.columns]
    df_modeling = df_enhanced[available_cols].copy()
    
    # Drop rows with missing target
    if 'f1_bw' in df_modeling.columns:
        df_modeling = df_modeling.dropna(subset=['f1_bw'])
    
    modeling_path = OUTPUT_DIR / "modeling_dataset_with_power_features.csv"
    df_modeling.to_csv(modeling_path, index=False)
    print(f"✓ Saved modeling dataset to: {modeling_path}")
    print(f"  Shape: {df_modeling.shape}")
    print(f"  Features: {list(df_modeling.columns)}")
    print()
    
    # Summary
    print("=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print("\n📊 Summary:")
    print(f"  • Original features: {df.shape[1]}")
    print(f"  • Enhanced features: {df_enhanced.shape[1]}")
    print(f"  • New power features: {len(power_features)}")
    print(f"  • Modeling dataset samples: {len(df_modeling)}")
    print()
    print("🎯 Next Steps:")
    print("  1. Run: python catboost_baseline.py")
    print("  2. Run: python catboost_optimized.py")
    print("  3. Run: python stacking_ensemble.py")
    print()

if __name__ == "__main__":
    main()
