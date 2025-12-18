# BART Model Updates - PMNS Dataset 20 Variables

**Date:** 2025-12-06  
**Updated By:** Sujit Sarkar

## Overview

Both BART model files have been updated to use the **20 PMNS Dataset variables** instead of the previously used top 25 variables from feature importance ranking.

## Files Updated

1. `PMNS_Variables/BART/BART.py` - Baseline BART model
2. `PMNS_Variables/BART/BART_Optimized.py` - BART with hyperparameter optimization

## PMNS Dataset 20 Variables

The following 20 variables are now used in both models:

| # | Variable Name | Description |
|---|---------------|-------------|
| 1 | f0_m_parity_v1 | Maternal parity |
| 2 | f0_m_wt_prepreg | Maternal weight pre-pregnancy |
| 3 | f0_m_fundal_ht_v2 | Maternal fundal height |
| 4 | f0_m_abd_cir_v2 | Maternal abdominal circumference |
| 5 | f0_m_wt_v2 | Maternal weight |
| 6 | f0_m_r4_v2 | Maternal R4 measurement |
| 7 | f0_m_lunch_cal_v1 | Maternal lunch calories |
| 8 | f0_m_p_sc_v1 | Maternal P skinfold |
| 9 | f0_m_o_sc_v1 | Maternal O skinfold |
| 10 | f0_m_pulse_r1_v2 | Maternal pulse reading 1 |
| 11 | f0_m_pulse_r2_v2 | Maternal pulse reading 2 |
| 12 | f0_m_glu_f_v2 | Maternal glucose fasting |
| 13 | f0_m_rcf_v2 | Maternal RCF |
| 14 | f0_m_g_sc_v1 | Maternal G skinfold |
| 15 | f0_m_plac_wt | Maternal placental weight |
| 16 | f0_m_GA_Del | Gestational age at delivery |
| 17 | f0_f_head_cir_ini | Fetal head circumference initial |
| 18 | f0_f_plt_ini | Fetal platelet count initial |
| 19 | f1_sex | Child sex |
| 20 | f0_m_age_eld_child | Maternal age of eldest child |

## Key Changes

### 1. **BART.py** (Baseline Model)

**Changed:**
- Removed `ranking_path` parameter from `__init__()` method
- Removed dependency on external feature ranking CSV file
- Directly defined 20 PMNS variables in `load_and_prepare_features()` method
- Updated data path to `Data/PMNS_Data.csv`
- Updated output directories: `PLOTS/BART_PMNS20` and `Data/processed/BART_PMNS20`
- Updated all file naming to use `pmns20` identifier
- Updated model type to `BART_PMNS20`

**Print Statement Updates:**
- "TOP 25 VARIABLES" → "PMNS 20 VARIABLES"
- "Using Top 25 variables" → "Using 20 PMNS Dataset variables"

### 2. **BART_Optimized.py** (Hyperparameter Optimization)

**Changed:**
- Removed `ranking_path` parameter from `__init__()` method
- Removed dependency on external feature ranking CSV file
- Directly defined 20 PMNS variables in `load_and_prepare_features()` method
- Updated data path to `Data/PMNS_Data.csv`
- Updated output directories: `PLOTS/BART_PMNS20_Optimized` and `Data/processed/BART_PMNS20_Optimized`
- Updated all file naming to use `pmns20` identifier
- Updated model type to `BART_Optimized_PMNS20`

**Print Statement Updates:**
- "TOP 25 VARIABLES" → "PMNS 20 VARIABLES"
- "Using Top 25 variables" → "Using 20 PMNS Dataset variables"
- "Use Top 25: True" → "Use PMNS 20: True"

## Benefits of This Update

1. **Dataset Specificity:** Models are now explicitly tied to the PMNS dataset
2. **Reduced Dependencies:** No longer require external feature ranking files
3. **Clarity:** Clear indication that these models use exactly 20 predefined variables
4. **Consistency:** Both baseline and optimized versions use identical feature sets
5. **Traceability:** Output files clearly indicate PMNS20 variant

## Output Changes

### File Naming Convention

**Before:**
- `bart_top25_results_*.json`
- `bart_top25_metrics_*.csv`
- `bart_optimized_results_*.json`

**After:**
- `bart_pmns20_results_*.json`
- `bart_pmns20_metrics_*.csv`
- `bart_optimized_pmns20_results_*.json`

### Directory Structure

**Before:**
```
PLOTS/BART_Top25/
PLOTS/BART_Top25_Optimized/
Data/processed/BART_Top25/
Data/processed/BART_Top25_Optimized/
```

**After:**
```
PLOTS/BART_PMNS20/
PLOTS/BART_PMNS20_Optimized/
Data/processed/BART_PMNS20/
Data/processed/BART_PMNS20_Optimized/
```

## Usage

### Running Baseline BART Model
```python
python PMNS_Variables/BART/BART.py
```

### Running Optimized BART Model
```python
python PMNS_Variables/BART/BART_Optimized.py
```

Both scripts will now:
1. Load data from `Data/PMNS_Data.csv`
2. Use exactly 20 PMNS variables
3. Generate results in PMNS20-specific directories
4. Save outputs with PMNS20 naming convention

## Notes

- Target variable remains `f1_bw` (birth weight)
- All other model configurations remain unchanged
- Hyperparameter optimization settings remain the same (RandomizedSearchCV with 5-fold CV, 20 iterations)
- Default BART parameters: n_trees=200, n_burn=200, n_samples=1000, alpha=0.95, beta=2.0
