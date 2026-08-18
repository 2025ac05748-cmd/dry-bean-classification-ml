import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.preprocessing import label_binarize


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🌱",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("🌱 Dry Bean Classification")
st.write(
    "Compare machine learning models for classifying "
    "seven different varieties of dry beans."
)


# ==========================================
# MODEL FILES
# ==========================================

model_files = {
    "Logistic Regression":
        "model/logistic_regression.pkl",

    "Decision Tree":
        "model/decision_tree.pkl",

    "KNN":
        "model/knn.pkl",

    "Naive Bayes":
        "model/naive_bayes.pkl",

    "Random Forest":
        "model/random_forest.pkl"
}


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Model Settings")

selected_model = st.sidebar.selectbox(
    "Select a model",
    list(model_files.keys())
)


# ==========================================
# FILE UPLOAD
# ==========================================

st.subheader("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the test data",
    type=["csv"]
)


# ==========================================
# EVALUATION
# ==========================================

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success(
        f"Dataset uploaded successfully: "
        f"{data.shape[0]} rows × {data.shape[1]} columns"
    )

    st.subheader("Uploaded Data")

    st.dataframe(
        data.head(10),
        use_container_width=True
    )

    if "Class" not in data.columns:

        st.error(
            "The uploaded CSV must contain a 'Class' column."
        )

    else:

        # Separate features and target
        X = data.drop("Class", axis=1)
        y = data["Class"]

        # Load selected model
        model_path = model_files[selected_model]

        if not os.path.exists(model_path):

            st.error(
                f"Model file not found: {model_path}"
            )

        else:

            with open(model_path, "rb") as file:
                model = pickle.load(file)

            # Predictions
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)

            classes = model.classes_

            # Binarize labels for multiclass AUC
            y_binary = label_binarize(
                y,
                classes=classes
            )

            # Metrics
            accuracy = accuracy_score(
                y,
                y_pred
            )

            auc = roc_auc_score(
                y_binary,
                y_proba,
                multi_class="ovr",
                average="weighted"
            )

            precision = precision_score(
                y,
                y_pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y,
                y_pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y,
                y_pred,
                average="weighted",
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y,
                y_pred
            )


            # ======================================
            # DISPLAY METRICS
            # ======================================

            st.subheader(
                f"2. Evaluation Results — {selected_model}"
            )

            col1, col2, col3 = st.columns(3)
            col4, col5, col6 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            col2.metric(
                "AUC",
                f"{auc:.4f}"
            )

            col3.metric(
                "Precision",
                f"{precision:.4f}"
            )

            col4.metric(
                "Recall",
                f"{recall:.4f}"
            )

            col5.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            col6.metric(
                "MCC",
                f"{mcc:.4f}"
            )


            # ======================================
            # CONFUSION MATRIX
            # ======================================

            st.subheader("3. Confusion Matrix")

            cm = confusion_matrix(
                y,
                y_pred,
                labels=classes
            )

            cm_df = pd.DataFrame(
                cm,
                index=classes,
                columns=classes
            )

            st.dataframe(
                cm_df,
                use_container_width=True
            )


            # ======================================
            # CLASSIFICATION REPORT
            # ======================================

            st.subheader(
                "4. Classification Report"
            )

            report = classification_report(
                y,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(
                report
            ).transpose()

            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )


else:

    st.info(
        "Please upload test_data.csv to evaluate a model."
    )


# ==========================================
# DATASET INFORMATION
# ==========================================

st.divider()

st.subheader("About the Dataset")

st.write(
    """
    The Dry Bean Dataset contains measurements of dry bean grains.
    The classification task contains seven bean varieties:
    Seker, Barbunya, Bombay, Cali, Dermason, Horoz and Sira.
    """
)