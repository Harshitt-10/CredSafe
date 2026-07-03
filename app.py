from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

st.set_page_config(
    page_title="CredSafe",
    page_icon="💳",
    layout="wide",
)

st.title("💳 CredSafe")
st.subheader("Credit Risk Modeling Dashboard")


@st.cache_resource
def load_model():
    return joblib.load(MODELS_DIR / "xgboost_tuned.joblib")


@st.cache_data
def load_data():
    return pd.read_csv(PROCESSED_DIR / "feature_table.csv")


model = load_model()
df = load_data()
explainer = shap.TreeExplainer(model)

customer_ids = df["SK_ID_CURR"].tolist()
selected_customer = st.sidebar.selectbox(
    "Select Customer",
    customer_ids,
)
customer = df[df["SK_ID_CURR"] == selected_customer]


@st.cache_resource
def load_preprocessor():
    return joblib.load(MODELS_DIR / "preprocessor.joblib")


@st.cache_resource
def load_feature_names():
    return joblib.load(MODELS_DIR / "feature_names.joblib")


preprocessor = load_preprocessor()
feature_names = load_feature_names()

X = customer.drop(
    columns=["SK_ID_CURR", "TARGET"]
)
X_processed = preprocessor.transform(X)

probability = model.predict_proba(X_processed)[0][1]

threshold = 0.65

tab1, tab2, tab3 = st.tabs(
    [
        "Risk Prediction",
        "SHAP Explanation",
        "Model Performance",
    ]
)

with tab1:
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Customer Summary")
        summary = pd.DataFrame({
            "Feature": [
                "Age (days)",
                "Annual Income",
                "Credit Amount",
                "Goods Price",
                "Employment Days",
                "Education",
                "Family Status",
                "Children"
            ],
            "Value": [
                customer["DAYS_BIRTH"].iloc[0],
                customer["AMT_INCOME_TOTAL"].iloc[0],
                customer["AMT_CREDIT"].iloc[0],
                customer["AMT_GOODS_PRICE"].iloc[0],
                customer["DAYS_EMPLOYED"].iloc[0],
                customer["NAME_EDUCATION_TYPE"].iloc[0],
                customer["NAME_FAMILY_STATUS"].iloc[0],
                customer["CNT_CHILDREN"].iloc[0],
            ]
        })
        st.dataframe(summary, use_container_width=True)
        with st.expander("View Complete Customer Record"):
            st.dataframe(customer, use_container_width=True)

    with right:
        st.metric(
            "Default Probability",
            f"{probability:.2%}",
        )
        st.metric(
            "Decision Threshold",
            threshold,
        )
        if probability >= threshold:
            st.error("🔴 High Risk")
        elif probability >= 0.40:
            st.warning("🟡 Medium Risk")
        else:
            st.success("🟢 Low Risk")


with tab2:
    st.subheader("SHAP Explanation")
    shap_values = explainer(X_processed)
    st.write(
        "The waterfall plot explains why the model assigned this risk score."
    )
    shap.plots.waterfall(
        shap_values[0],
        max_display=15,
        show=False,
    )

    st.pyplot(plt.gcf())
    plt.clf()
    plt.close(plt.gcf())
    st.divider()
    st.subheader("Top Feature Contributions")
    contributions = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP Value": shap_values.values[0],
            "Feature Value": X_processed[0],
        }
    )
    contributions["Absolute SHAP"] = (
        contributions["SHAP Value"].abs()
    )
    contributions = contributions.sort_values(
        "Absolute SHAP",
        ascending=False,
    )
    st.dataframe(
        contributions[
            ["Feature", "Feature Value", "SHAP Value"]
        ].head(15),
        use_container_width=True,
    )


with tab3:
    st.subheader("Model Performance")
    comparison = pd.read_csv(
        PROJECT_ROOT
        / "reports"
        / "model_comparison.csv"
    )
    st.dataframe(
        comparison,
        use_container_width=True,
    )
    st.metric(
        "Selected Model",
        "XGBoost (Tuned)",
    )
    st.metric(
        "Threshold",
        "0.65",
    )
    st.metric(
        "ROC-AUC",
        "0.7712",
    )
