# KEM PMNS NBWP Data Analysis

## 🎯 Project Purpose

This project analyzes maternal and child health data to identify risk factors for low birthweight (LBW) outcomes. The study transforms a raw dataset of 855 variables into a research-ready dataset with actionable insights for public health intervention.

## 📊 Key Findings

- **High LBW Rate**: 36.7% of births are low birthweight (2.4x WHO threshold)
- **Gender Disparity**: Female babies at 41.0% vs male babies at 33.4% LBW rate
- **Key Risk Factors**: Electricity access, maternal gravidity, living female children, maternal parity, and two-wheeler ownership

## 🔬 Methods

### Data Processing Pipeline
1. **Data Audit**: 855 variables, 791 observations, 99.6% dictionary match rate
2. **Feature Categorization**: 7 domain categories (maternal clinical, anthropometry, socio-demographic, etc.)
3. **Outcome Engineering**: Birthweight analysis with LBW definition (<2,500g)
4. **Exploratory Analysis**: Correlation analysis, statistical testing, pattern recognition

## 📁 Project Structure

- `Data/` - Raw, processed, and final datasets
  - `raw/` - Original unprocessed data files
  - `processed/` - Cleaned and engineered datasets
  - `external/` - Reference files and data dictionaries
- `Scripts/` - Python analysis scripts and Jupyter notebooks
  - `Feature_Engineering/` - Data processing and feature creation scripts
  - `mle_procedural_notebook_with_evaluation.ipynb` - Main MLE analysis notebook
- `NoteBooks/` - Jupyter notebooks for exploratory analysis
- `PLOTS/` - Generated visualizations and analysis plots
  - `Feature_Engineering/` - EDA and feature analysis plots
  - `dermatoglyphic_qc_simple/` - Quality control visualizations
- `Reports/` - Analysis reports and summaries
- `Summary/` - Detailed documentation and methodology
  - `MLE/` - Maximum Likelihood Estimation documentation
  - `Step_1[FE]/` - Feature Engineering step documentation

## 🚀 Getting Started

1. Clone the repository
2. Install required Python packages
3. Run the analysis scripts in the `Scripts/` directory
4. View results in `PLOTS/` and `Reports/` directories

## 📈 Results

The analysis identified 5 significant risk factors for low birthweight with statistical significance (p<0.05). The findings provide clear guidance for targeted public health interventions and policy development.

## 📄 License

MIT License - Copyright (c) 2024 Sujit Sarkar

## 📞 Contact

For questions about this research project, please contact the research team.