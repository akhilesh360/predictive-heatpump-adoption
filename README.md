# Predictive Heat Pump Adoption

## Project Overview

This repository provides a comprehensive, production-style pipeline for predicting household adoption of electric heat pumps using machine learning. The project leverages a synthetic dataset with realistic household, housing, and utility features, and demonstrates best practices in data science, model evaluation, and interpretability. The workflow is designed to support policy makers, consultants, and researchers in understanding adoption drivers and targeting outreach efforts.

## Key Features
- **End-to-end pipeline:** Data preprocessing, model training, evaluation, and reporting
- **Multiple models:** Logistic Regression (baseline), Random Forest, and XGBoost
- **Interpretability:** Feature importance and SHAP analysis
- **Fairness analysis:** Performance across demographic and regional groups
- **Consulting-ready structure:** Modular code, clear documentation, reproducible environment



```
predictive-heatpump-adoption/
├── src/                # Source code (models, preprocessing, evaluation, utils)
├── notebooks/          # Jupyter notebooks (EDA, model demo, comparison)
├── data/               # Synthetic and scored datasets
├── outputs/            # Model artifacts, metrics, plots (excluded from git)
├── config/             # Configuration files
├── docs/               # Policy brief and documentation
├── Dockerfile          # Containerization setup
├── Makefile            # Automation commands
├── requirements.txt    # Python dependencies
├── README.md           # Project overview and instructions
├── run_compare.py      # Script for model comparison
├── run_pipeline.py     # Main pipeline script
```

## Step-by-Step Usage

### 1. Clone the Repository
```bash
# Predictive Heat Pump Adoption

### 2. Set Up the Environment
#### Option A: Python Virtual Environment
```bash
python -m venv .venv

# Predictive Heat Pump Adoption

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Docker Ready](https://img.shields.io/badge/Docker-ready-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
```bash
make train        # Run baseline logistic regression pipeline
make compare      # Compare Logistic Regression, Random Forest, and XGBoost
- **Fairness analysis:** Performance across demographic and regional groups
- **Consulting-ready structure:** Modular code, clear documentation, reproducible environment

```
├── Makefile            # Automation commands
├── requirements.txt    # Python dependencies

## About the Dataset
The project uses a synthetic dataset designed to mimic real-world household, housing, and utility characteristics relevant to heat pump adoption. Key features include:
- **Household income:** Captures economic status and purchasing power.
- **Region:** Reflects geographic differences in climate, policy, and infrastructure.
- **DAC status:** Indicates whether a household is in a Disadvantaged Community, supporting fairness analysis.
- **Housing attributes:** Such as home type, age, and square footage, which influence adoption likelihood.
- **Utility data:** Includes energy usage and costs, providing context for potential savings.
- **Adoption flag:** The target variable indicating whether a household has adopted a heat pump.

The synthetic nature of the data ensures privacy and reproducibility, while still allowing for meaningful analysis and demonstration of modeling techniques. The dataset is explored in detail in the first notebook, where distributions and relationships are visualized to inform feature engineering and model selection.

## Technologies Used
- **Python 3.10+:** Main programming language
- **scikit-learn:** Machine learning models and pipelines
- **XGBoost:** Advanced gradient boosting model
- **SHAP:** Model explainability
- **pandas, numpy:** Data manipulation
- **matplotlib:** Visualization
- **Docker:** Containerization for reproducible environments
- **Jupyter Notebooks:** Interactive analysis and reporting

## Example Outputs
- **ROC Curve Comparison:** Visualizes the trade-off between true positive and false positive rates for all models.
- **Feature Importance:** Bar chart showing which features most influence adoption predictions.
- **Fairness Report:** JSON file summarizing model performance across demographic and regional groups.
- **SHAP Summary Plot:** Beeswarm plot showing the impact of each feature on model output.

## Step-by-Step Usage

1. Clone the Repository
```bash
git clone <repo-url>
2. Set Up the Environment
Option A: Python Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Option B: Docker
```bash
docker build -t heatpump .
docker run --rm heatpump python run_pipeline.py
```

3. Explore the Data
Open notebooks/01_exploration.ipynb in Jupyter.

4. Train and Evaluate Models
```bash
python run_pipeline.py   # Train baseline logistic regression
python run_compare.py    # Compare Logistic, RF, XGB
```

5. Fairness and Explainability
Outputs include:
- outputs/fairness_report.json
- outputs/shap_summary_xgb.png
- outputs/feature_importance.png

6. Inspect Results and Artifacts
All results are in outputs/. Notebooks provide step-by-step analysis.

## How to Contribute
Fork the repository and create a new branch.
Make your changes, following the modular structure.
Submit a pull request with a clear description of your changes.
👉 See CONTRIBUTING.md for detailed guidelines.

## License
This project is licensed under the MIT License.
  - All models were trained and tested on the same data split to ensure fairness.
