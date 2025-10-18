import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*60)
print("GENERATING REMAINING QC PLOTS")
print("="*60)

def load_data():
    """Load the dataset"""
    try:
        df = pd.read_excel('Data/raw/IMPUTED_DATA_WITH REDUCED_columns_21_09_2025.xlsx')
        print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def create_correlation_heatmap(df):
    """Create correlation heatmap for a subset of numerical variables"""
    print("\n📊 Creating correlation heatmap...")
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove constant columns
    numerical_cols = [col for col in numerical_cols if df[col].nunique() > 1]
    
    # Select a subset of columns for correlation (first 50 to avoid memory issues)
    cols_for_corr = numerical_cols[:50]
    
    print(f"📊 Using {len(cols_for_corr)} columns for correlation analysis")
    
    # Calculate correlation matrix
    correlation_matrix = df[cols_for_corr].corr()
    
    # Create the heatmap
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
               square=True, linewidths=0.1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap (First 50 Numerical Variables)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save the plot
    plots_dir = Path('PLOTS')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Correlation heatmap saved")

def create_boxplots(df):
    """Create box plots for a subset of numerical variables"""
    print("\n📊 Creating box plots...")
    
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove constant columns
    numerical_cols = [col for col in numerical_cols if df[col].nunique() > 1]
    
    # Select first 12 columns for box plots
    cols_for_box = numerical_cols[:12]
    
    print(f"📊 Creating box plots for {len(cols_for_box)} columns")
    
    # Create subplots
    n_cols = 3
    n_rows = (len(cols_for_box) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(cols_for_box):
        if i < len(axes):
            df.boxplot(column=col, ax=axes[i])
            axes[i].set_title(f'{col}', fontweight='bold', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45, labelsize=8)
    
    # Hide empty subplots
    for i in range(len(cols_for_box), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Box Plots for Numerical Variables (First 12)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plots_dir = Path('PLOTS')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Box plots saved")

def create_summary_plot(df):
    """Create a summary plot showing data quality metrics"""
    print("\n📊 Creating summary plot...")
    
    # Calculate basic statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
    
    # Create summary data
    summary_data = {
        'Metric': ['Total Rows', 'Total Columns', 'Numerical Columns', 'Constant Columns', 
                  'Columns with Missing Values', 'Duplicate Rows'],
        'Count': [df.shape[0], df.shape[1], len(numerical_cols), len(constant_cols), 
                 len(missing_cols), df.duplicated().sum()]
    }
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(summary_data['Metric'], summary_data['Count'], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    plt.title('Data Quality Summary', fontsize=16, fontweight='bold')
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, summary_data['Count']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plots_dir = Path('PLOTS')
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(plots_dir / 'data_quality_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Data quality summary plot saved")

def main():
    """Main execution function"""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Create plots
    create_correlation_heatmap(df)
    create_boxplots(df)
    create_summary_plot(df)
    
    print("\n" + "="*60)
    print("ALL REMAINING PLOTS GENERATED SUCCESSFULLY")
    print("="*60)
    print("✅ Files created in PLOTS/ directory:")
    print("  - correlation_heatmap.png")
    print("  - boxplots.png") 
    print("  - data_quality_summary.png")

if __name__ == "__main__":
    main()
