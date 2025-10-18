import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("SAVING CLEANED DATASET WITH ENGINEERED FEATURES")
print("="*60)

def load_and_clean_data():
    """Load the original dataset and add engineered features"""
    print("\n📊 Loading and cleaning dataset...")
    
    # Load the original dataset
    df = pd.read_excel('Data/raw/IMPUTED_DATA_WITH REDUCED_columns_21_09_2025.xlsx')
    print(f"✅ Original dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Add LBW_flag variable
    if 'f1_bw' in df.columns:
        df['LBW_flag'] = np.where(df['f1_bw'] < 2500, 1, 0)
        print("✅ LBW_flag variable added")
        
        # Display LBW statistics
        lbw_count = df['LBW_flag'].sum()
        lbw_percent = (lbw_count / len(df)) * 100
        print(f"📊 LBW cases: {lbw_count} ({lbw_percent:.1f}%)")
    else:
        print("⚠️  f1_bw variable not found - LBW_flag not created")
    
    return df

def save_cleaned_dataset(df):
    """Save the cleaned dataset with engineered features"""
    print("\n📊 Saving cleaned dataset...")
    
    # Create output directory
    output_dir = Path('Data/processed')
    output_dir.mkdir(exist_ok=True)
    
    # Save as Excel
    excel_file = output_dir / 'cleaned_dataset_with_engineered_features.xlsx'
    df.to_excel(excel_file, index=False)
    print(f"✅ Excel file saved: {excel_file}")
    
    # Save as CSV
    csv_file = output_dir / 'cleaned_dataset_with_engineered_features.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV file saved: {csv_file}")
    
    # Create a summary of the cleaned dataset
    summary = {
        'Total_Rows': len(df),
        'Total_Columns': len(df.columns),
        'Numerical_Columns': len(df.select_dtypes(include=[np.number]).columns),
        'Categorical_Columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'Missing_Values': df.isnull().sum().sum(),
        'Duplicate_Rows': df.duplicated().sum(),
        'LBW_Cases': df['LBW_flag'].sum() if 'LBW_flag' in df.columns else 0,
        'LBW_Percentage': (df['LBW_flag'].sum() / len(df)) * 100 if 'LBW_flag' in df.columns else 0
    }
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / 'cleaned_dataset_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Summary saved: {summary_file}")
    
    return excel_file, csv_file, summary_file

def main():
    """Main execution function"""
    # Load and clean data
    df = load_and_clean_data()
    
    # Save cleaned dataset
    excel_file, csv_file, summary_file = save_cleaned_dataset(df)
    
    print("\n" + "="*60)
    print("CLEANED DATASET SAVED SUCCESSFULLY")
    print("="*60)
    print("✅ Files created:")
    print(f"  - Excel: {excel_file}")
    print(f"  - CSV: {csv_file}")
    print(f"  - Summary: {summary_file}")
    print("="*60)

if __name__ == "__main__":
    main()
