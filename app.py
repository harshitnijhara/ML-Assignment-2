import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ML Model Comparison App",
    layout="wide"
)

st.title("Model Comparison of Breast Cancer Detection for Machine Learning Assignment")
st.write("Here we Compare multiple ML classifiers using common performance metrics")


uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())


    if "diagnosis" not in df.columns:
        st.error("Dataset must contain a 'diagnosis' column")
        st.stop()

    df = df.drop(columns=["Unnamed: 32", "id"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    test_size = st.slider(
        "Test Size (%)",
        min_value=10,
        max_value=40,
        value=20
    ) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    st.subheader("Select Machine Learning Models")

    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=800),
        "Decision Tree": DecisionTreeClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(),
        "XGBoost (Ensemble)": XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss"
        )
    }

    selected_models = st.multiselect(
        "Choose ML models to evaluate",
        list(model_options.keys()),
        default=list(model_options.keys())
    )


    if st.button("Run Model Comparison"):

        results = []

        for name in selected_models:
            model = model_options[name]

            if name == "Logistic Regression":
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            results.append([
                name,
                accuracy_score(y_test, y_pred),
                roc_auc_score(y_test, y_prob),
                precision_score(y_test, y_pred),
                recall_score(y_test, y_pred),
                f1_score(y_test, y_pred),
                matthews_corrcoef(y_test, y_pred)
            ])

        comparison_table = pd.DataFrame(
            results,
            columns=["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        )

        st.subheader("Model Comparison Results")
        st.dataframe(comparison_table.round(4))

        st.subheader("AUC Comparison")
        fig, ax = plt.subplots()
        ax.bar(
            comparison_table["ML Model Name"],
            comparison_table["AUC"]
        )
        ax.set_ylabel("AUC Score")
        ax.set_xticklabels(
            comparison_table["ML Model Name"],
            rotation=45,
            ha="right"
        )
        st.pyplot(fig)

else:
    st.info("Upload a CSV file to get started")