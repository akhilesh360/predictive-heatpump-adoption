# Policy Brief: Predicting Heat Pump Adoption

**Objective:** Optimize limited incentive budgets by identifying households most likely to adopt heat pumps.

## Key Findings
- **Model Performance:** XGBoost AUC ~0.83 (Logit ~0.80). Stable across folds.
- **Drivers of Adoption (SHAP):**
  - Higher electricity rates, baseline gas usage, DAC status, and prior rebates are strongest predictors.
- **Targeting Impact:**
  - Top 2–3 deciles show ~2x adoption vs average.
  - Targeting them yields ~10% higher enrollment without additional budget.

## Actionable Recommendations
1. Prioritize top propensity deciles in marketing campaigns.
2. Allocate outreach staff to DAC census tracts with high predicted adoption probability.
3. Monitor fairness: DAC households are not disadvantaged by the model.

## Visuals for Stakeholders
- **ROC Curve Comparison:** Logistic vs RandomForest vs XGBoost.
- **Lift by Decile:** Adoption rate across probability bands.
- **SHAP Feature Importance:** Transparent explanation of drivers.

## Next Steps
- Integrate scored households into Tableau dashboards for program managers.
- Explore budget-constrained targeting optimization (cost-per-acquisition modeling).
