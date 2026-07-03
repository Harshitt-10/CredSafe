# 💳 CredSafe

An end-to-end Credit Risk Modeling and Explainable AI pipeline built on the **Home Credit Default Risk** dataset. CredSafe predicts the probability of loan default using machine learning, SQL-based feature engineering, and SHAP explainability, and presents predictions through an interactive Streamlit dashboard.

---

## Features

- Multi-source credit data ingestion using SQLite
- SQL-based customer-level feature engineering
- Production-style preprocessing pipeline using Scikit-learn
- Baseline model comparison
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Balanced XGBoost
- Hyperparameter tuning with RandomizedSearchCV
- Threshold optimization for imbalanced classification
- SHAP explainability for global and local model interpretation
- Interactive Streamlit dashboard

---

## Project Structure

```text
CredSafe/
│
├── app.py
│
├── src/
│   ├── load_data.py
│   ├── create_views.py
│   ├── build_feature_table.py
│   ├── preprocess.py
│   ├── train.py
│   ├── tune_xgboost.py
│   ├── threshold_tuning.py
│   ├── explain.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── home_credit.db
│
├── models/
│
├── reports/
│
├── requirements.txt
└── README.md
```

---

# Dataset Used

**Home Credit Default Risk**

- 307,511 loan applications
- Multiple relational datasets
- Customer demographics
- Bureau records
- Previous applications
- Installment history
- Credit card balances
- POS cash balances

Dataset: https://www.kaggle.com/competitions/home-credit-default-risk

---

# Pipeline

```text
Raw CSV Files
      │
      ▼
SQLite Database
      │
      ▼
SQL Views
      │
      ▼
Customer-Level Feature Table
      │
      ▼
Preprocessing Pipeline
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Threshold Optimization
      │
      ▼
SHAP Explainability
      │
      ▼
Streamlit Dashboard
```

---

# SQL Feature Engineering

Customer-level features were created by aggregating information from multiple relational tables.

Examples include:

- Number of previous applications
- Average bureau credit
- Average credit card balance
- Payment-to-installment ratio
- Number of POS cash loans
- Installment payment statistics

These engineered features were merged with the primary application dataset for model training.

---

# Machine Learning Pipeline

## Data Preprocessing

- Median imputation for numerical features
- Most-frequent imputation for categorical features
- One-Hot Encoding
- Stratified Train/Test Split
- Reusable preprocessing pipeline saved using Joblib

---

## Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost
- Balanced XGBoost (`scale_pos_weight`)
- Tuned XGBoost

---

## Hyperparameter Tuning

RandomizedSearchCV was used to optimize XGBoost using ROC-AUC as the objective metric.

Optimized parameters include:

- Number of estimators
- Maximum tree depth
- Learning rate
- Minimum child weight
- Subsample ratio
- Column sampling ratio
- Gamma
- L1/L2 Regularization

---

## Threshold Optimization

Instead of using the default probability threshold (0.50), the decision threshold was optimized using the validation set.

**Optimal Threshold**

```
0.65
```

This improved the balance between precision and recall for the imbalanced credit-risk classification task.

---

# Model Performance

| Model               |         Accuracy |                              Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | ---------------: | -------------------------------------: | -----: | -------: | ------: |
| Logistic Regression |           0.9193 |                                 0.0000 | 0.0000 |   0.0000 |  0.6317 |
| Random Forest       |           0.9193 |                                 1.0000 | 0.0008 |   0.0016 |  0.7315 |
| XGBoost             |           0.9200 |                                 0.5827 | 0.0326 |   0.0618 |  0.7689 |
| XGBoost (Balanced)  |           0.7412 |                                 0.1849 | 0.6473 |   0.2877 |  0.7670 |
| **XGBoost (Tuned)** | **0.77 ROC-AUC** | *(Optimized using RandomizedSearchCV)* |

---

# Explainability

SHAP was used to provide both global and local explanations.

Generated reports include:

- SHAP Summary Plot
- SHAP Feature Importance
- Individual Waterfall Explanation

Top influential features include:

- EXT_SOURCE_3
- EXT_SOURCE_2
- AMT_GOODS_PRICE
- AMT_CREDIT
- Payment-to-Installment Ratio
- DAYS_EMPLOYED
- DAYS_BIRTH

---

# Streamlit Dashboard

The interactive dashboard allows users to:

- Select customers by Customer ID
- Predict default probability
- Classify customer risk using the optimized threshold
- View customer summary
- Visualize SHAP explanations
- Inspect top contributing features
- Compare model performance

---

# Tech Stack

### Languages

- Python
- SQL

### Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Joblib
- SQLite

---

# Installation

Clone the repository

```bash
git clone https://github.com/Harshitt-10/CredSafe.git
cd CredSafe
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the dashboard

```bash
streamlit run app.py
```

---

# Future Improvements

- Docker support
- FastAPI deployment
- Probability calibration
- Interactive SHAP dependence plots
- Automated feature selection
- Model monitoring and drift detection