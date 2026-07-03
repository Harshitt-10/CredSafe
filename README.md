# CredSafe

An end-to-end Machine Learning pipeline for **credit risk assessment and loan default prediction** using the **Home Credit Default Risk** dataset. The project combines SQL-based feature engineering, machine learning, explainable AI, and an interactive dashboard to predict customer default risk.

---

## Features

- SQL-based feature engineering using SQLite
- Automated preprocessing pipeline with Scikit-learn
- Customer-level feature aggregation from multiple relational tables
- Explainable predictions using SHAP *(Coming Soon)*
- Interactive Streamlit dashboard *(Coming Soon)*

---

## Tech Stack

### Languages
- Python
- SQL

### Data Engineering
- SQLite
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost *(Coming Soon)*
- SHAP *(Coming Soon)*

### Visualization
- Matplotlib
- Plotly *(Coming Soon)*
- Streamlit *(Coming Soon)*

---

## Project Structure

```text
CredSafe/

├── app/
├── data/
│   ├── raw/
│   ├── processed/
│   └── home_credit.db
│
├── models/
│   ├── preprocessor.joblib
│   └── feature_names.joblib
│
├── notebooks/
├── reports/
├── sql/
│   ├── create_views.sql
│   └── final_query.sql
│
├── src/
│   ├── load_data.py
│   ├── check_database.py
│   ├── create_views.py
│   ├── build_feature_table.py
│   └── preprocess.py
│
├── requirements.txt
└── README.md
```

---

## Machine Learning Pipeline

```
CSV Files
      │
      ▼
SQLite Database
      │
      ▼
SQL Feature Engineering
      │
      ▼
Feature Table
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Model Training
      │
      ▼
Evaluation
      │
      ▼
Explainability (SHAP)
      │
      ▼
Streamlit Dashboard
```

---

## Current Progress

- [x] Project setup
- [x] Data ingestion pipeline
- [x] SQLite database creation
- [x] SQL feature engineering
- [x] Feature table generation
- [x] Automated preprocessing pipeline
- [ ] Baseline model training
- [ ] Model evaluation
- [ ] Hyperparameter tuning
- [ ] Explainability with SHAP
- [ ] Streamlit dashboard
- [ ] Deployment

---

## Dataset

**Home Credit Default Risk**

The dataset contains historical loan application records along with customer demographics, previous credit history, installment payments, credit card balances, and bureau information.

Target Variable:

- **0** → Loan Repaid
- **1** → Loan Default

---

## Upcoming Work

- Train Logistic Regression, Random Forest, and XGBoost models
- Handle class imbalance using SMOTE and `scale_pos_weight`
- Hyperparameter tuning with RandomizedSearchCV
- SHAP-based model explainability
- Interactive Streamlit dashboard
- Batch prediction support
- Docker deployment