# SHAP Feature Importance Analysis Report
## Top 30 Engineered Features

**Generated:** 2025-12-06 12:32:46  
**Dataset:** Paper/data/clean_top15_features_20251206.csv  
**Total Samples:** 791  
**SHAP Sample Size:** 500  
**Features Analyzed:** 15

---

## 1. Executive Summary

This report presents a comprehensive SHAP (SHapley Additive exPlanations) analysis of the top 30 engineered features
for birth weight prediction. SHAP values provide model-agnostic interpretability by quantifying each feature's
contribution to individual predictions.

### Models Analyzed
- **XGBoost**
- **RandomForest**

---

## 2. Feature Importance Rankings

### XGBoost

| Rank | Feature | Importance (%) | Mean SHAP | Std SHAP | Cumulative (%) |
|------|---------|----------------|-----------|----------|----------------|
| 1 | `f0_m_plac_wt` | 31.67% | +1.56 | 149.04 | 31.67% |
| 2 | `f0_m_GA_Del` | 26.13% | -6.48 | 138.21 | 57.80% |
| 3 | `gestational_health_index` | 14.56% | -0.25 | 63.47 | 72.36% |
| 4 | `f0_m_abd_cir_v2` | 6.54% | -0.77 | 29.86 | 78.90% |
| 5 | `f0_m_rcf_v2` | 5.00% | +1.11 | 22.10 | 83.90% |
| 6 | `bmi_age_interaction` | 2.41% | -0.17 | 11.55 | 86.31% |
| 7 | `wt_ht_interaction` | 2.30% | -0.84 | 11.03 | 88.60% |
| 8 | `f0_m_wt_prepreg_squared` | 2.04% | +0.13 | 8.75 | 90.64% |
| 9 | `nutritional_status` | 1.66% | -0.77 | 8.53 | 92.30% |
| 10 | `f0_m_bi_v1` | 1.62% | +0.02 | 9.24 | 93.91% |
| 11 | `f0_m_fundal_ht_v2` | 1.62% | -1.18 | 7.17 | 95.53% |
| 12 | `f0_m_ht` | 1.58% | +0.25 | 8.16 | 97.11% |
| 13 | `f0_m_int_sin_ma` | 1.11% | +0.49 | 7.34 | 98.22% |
| 14 | `f0_m_age` | 1.05% | -0.18 | 4.58 | 99.27% |
| 15 | `bmi_age_ratio` | 0.73% | -0.18 | 3.76 | 100.00% |

### RandomForest

| Rank | Feature | Importance (%) | Mean SHAP | Std SHAP | Cumulative (%) |
|------|---------|----------------|-----------|----------|----------------|
| 1 | `f0_m_plac_wt` | 30.03% | +0.39 | 155.99 | 30.03% |
| 2 | `f0_m_GA_Del` | 23.07% | -5.25 | 138.39 | 53.10% |
| 3 | `gestational_health_index` | 16.10% | -2.60 | 79.64 | 69.20% |
| 4 | `f0_m_abd_cir_v2` | 6.09% | -0.65 | 32.23 | 75.29% |
| 5 | `f0_m_rcf_v2` | 4.84% | +1.72 | 25.87 | 80.13% |
| 6 | `f0_m_fundal_ht_v2` | 4.04% | +0.36 | 20.67 | 84.16% |
| 7 | `bmi_age_interaction` | 2.81% | -0.08 | 15.54 | 86.97% |
| 8 | `wt_ht_interaction` | 2.51% | -1.54 | 14.10 | 89.48% |
| 9 | `f0_m_bi_v1` | 2.20% | +0.35 | 12.32 | 91.68% |
| 10 | `nutritional_status` | 2.02% | -1.17 | 11.41 | 93.70% |
| 11 | `f0_m_wt_prepreg_squared` | 1.81% | +0.65 | 9.48 | 95.51% |
| 12 | `f0_m_ht` | 1.70% | -0.30 | 11.01 | 97.21% |
| 13 | `bmi_age_ratio` | 1.23% | +0.02 | 6.94 | 98.43% |
| 14 | `f0_m_int_sin_ma` | 0.85% | +0.38 | 5.64 | 99.28% |
| 15 | `f0_m_age` | 0.72% | -0.31 | 4.16 | 100.00% |


---

## 3. Key Insights

### Top Features Across Models

**Common Top 5 Features:**
- `f0_m_plac_wt`
- `f0_m_GA_Del`
- `f0_m_rcf_v2`
- `f0_m_abd_cir_v2`
- `gestational_health_index`


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

**XGBoost:**
- Summary plot: `plots/xgboost_shap_summary.png`
- Bar plot: `plots/xgboost_shap_bar.png`
- Waterfall plots: `plots/xgboost_waterfall_sample*.png`
- Dependence plots: `plots/xgboost_dependence_*.png`
- Force plots: `plots/xgboost_force_sample*.png`

**RandomForest:**
- Summary plot: `plots/randomforest_shap_summary.png`
- Bar plot: `plots/randomforest_shap_bar.png`
- Waterfall plots: `plots/randomforest_waterfall_sample*.png`
- Dependence plots: `plots/randomforest_dependence_*.png`
- Force plots: `plots/randomforest_force_sample*.png`

### Data Files

- `results/xgboost_shap_importance.csv`: Detailed importance metrics
- `results/randomforest_shap_importance.csv`: Detailed importance metrics
- `results/feature_ranking_comparison.csv`: Cross-model ranking comparison

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
