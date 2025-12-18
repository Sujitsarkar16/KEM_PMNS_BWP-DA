# 🔬 Advanced Stacking Models: SOTA Approach for Birth Weight Prediction

## Overview
This folder implements state-of-the-art (SOTA) machine learning strategies to maximize RMSE performance and efficiency on the PMNS intergenerational cohort dataset (n=793).

## Research-Driven Strategy

### 1. **CatBoost Integration**
- Superior to XGBoost and BART for biological/clinical datasets
- Uses "Ordered Boosting" to reduce overfitting on small cohorts
- 10x faster than BART, 2x faster than XGBoost
- Better handles non-linear interactions (genetics × nutrition)

### 2. **Stacking Regressor Architecture**
Instead of heavy hyperparameter tuning for marginal gains, we use ensemble stacking:

**Level 0 (Base Models):**
- XGBoost: Captures sharp splits and tree-based patterns
- BART: Captures smooth uncertainty and Bayesian prior knowledge
- LinearRegression: Captures baseline linear trends

**Level 1 (Meta-Learner):**
- RidgeRegression: Learns optimal weights to combine base predictions
- **Effect**: Cancels out model-specific errors, typically improves RMSE by 3-5%

### 3. **Power Interaction Features**

#### A. **Genetic Envelope Interaction**
```python
Genetic_Volume = f0_m_ht * f0_f_head_cir_ini
```
**Theory**: Birth weight is constrained by maternal height (uterine capacity) but driven by paternal genetics (skeletal potential). This interaction captures that biological tension.

#### B. **Placental Efficiency Proxy**
```python
Placental_Efficiency_Proxy = f0_m_plac_wt / f0_m_wt_prepreg
```
**Theory**: Normalizing placental weight by pre-pregnancy maternal weight provides a "Biological Efficiency Score" of the reproductive system.

## Files Structure

```
Advanced_Stacking_Models/
├── README.md (this file)
├── feature_engineering.py - Creates power interaction features
├── catboost_baseline.py - CatBoost baseline model
├── catboost_optimized.py - CatBoost with hyperparameter tuning
├── stacking_ensemble.py - Stacking regressor implementation
├── comparison_report.py - Generates comprehensive comparison
├── Data/ - Generated datasets with engineered features
└── Results/ - Model outputs, metrics, and comparisons
```

## Expected Performance Gains

1. **CatBoost** alone: 2-4% RMSE improvement over XGBoost
2. **Power Features**: 3-5% RMSE improvement
3. **Stacking Ensemble**: Additional 3-5% RMSE improvement
4. **Combined**: Potential 8-14% total RMSE improvement

## Usage

1. **Feature Engineering**:
   ```bash
   python feature_engineering.py
   ```

2. **Run CatBoost Models**:
   ```bash
   python catboost_baseline.py
   python catboost_optimized.py
   ```

3. **Run Stacking Ensemble**:
   ```bash
   python stacking_ensemble.py
   ```

4. **Generate Comparison Report**:
   ```bash
   python comparison_report.py
   ```

## Scientific Rationale

### Why This Approach Works for PMNS Data:

1. **Intergenerational Effects**: The dataset includes paternal (`f0_f_*`) and maternal historical variables. The genetic envelope interaction explicitly models these cross-generational effects.

2. **Small Sample Size**: With n=793, traditional deep learning fails. Stacking leverages multiple weak learners effectively.

3. **Mixed Data Types**: CatBoost's "Ordered Boosting" handles categorical variables (sociodemographic) with continuous variables (clinical biomarkers) better than other methods.

4. **Non-linear Interactions**: Nutrition × Genetics × Environment interactions are highly non-linear. Stacking captures this better than single models.

## Kaggle Grandmaster Technique

This "Stacking > Tuning" philosophy is standard in competitive ML:
- Instead of spending days tuning BART's beta/alpha for 0.01 improvement
- We combine complementary models that make different types of errors
- The meta-learner finds optimal weights automatically

## References

- CatBoost: Prokhorenkova et al. (2018) "CatBoost: unbiased boosting with categorical features"
- Stacking: Wolpert (1992) "Stacked generalization"
- Interaction Features: Hastie et al. (2009) "Elements of Statistical Learning"
