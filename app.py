"""
app.py
-------
Healthcare Support System using Data Mining Algorithm - Streamlit Dashboard.

Run with:
    streamlit run app.py

Make sure you have already run `python train_model.py` at least once so
that the trained model artifacts exist inside the models/ folder.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Support System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart_disease.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LOGO_PATH = os.path.join(IMAGES_DIR, "logo.png")


# ---------------------------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------------------------
def load_custom_css():
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.6rem;
            font-weight: 800;
            color: #1B4965;
            text-align: center;
            padding-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #5C6B73;
            text-align: center;
            padding-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F1F6F9;
            padding: 1.2rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
        .risk-low {
            background-color: #D8F3DC;
            color: #1B4332;
            padding: 1.2rem;
            border-radius: 12px;
            font-size: 1.3rem;
            font-weight: 700;
            text-align: center;
        }
        .risk-medium {
            background-color: #FFF3CD;
            color: #7A5C00;
            padding: 1.2rem;
            border-radius: 12px;
            font-size: 1.3rem;
            font-weight: 700;
            text-align: center;
        }
        .risk-high {
            background-color: #F8D7DA;
            color: #721C24;
            padding: 1.2rem;
            border-radius: 12px;
            font-size: 1.3rem;
            font-weight: 700;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    df["chol"] = df["chol"].fillna(df["chol"].median())
    df["trestbps"] = df["trestbps"].fillna(df["trestbps"].median())
    df["thalach"] = df["thalach"].fillna(df["thalach"].median())
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    selected_features = joblib.load(os.path.join(MODELS_DIR, "selected_features.pkl"))
    return model, scaler, selected_features


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
def sidebar_navigation():
    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown("## 🏥 Healthcare Support System")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["🏠 Home", "🩺 Prediction", "📊 Data Visualization", "ℹ️ About"],
        )
        st.markdown("---")
        st.caption("Final Year Mini Project")
        st.caption("AI & Data Science Engineering")
    return page


# ---------------------------------------------------------------------------
# Home Page
# ---------------------------------------------------------------------------
def home_page():
    st.markdown('<div class="main-header">Healthcare Support System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Using Data Mining Algorithms for Early Heart Disease Risk Prediction</div>',
        unsafe_allow_html=True,
    )

    if os.path.exists(LOGO_PATH):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(LOGO_PATH, use_container_width=True)

    st.markdown("### Welcome 👋")
    st.write(
        """
        This dashboard demonstrates a complete **data mining pipeline** applied to
        healthcare data. It uses a **Random Forest Classifier**, benchmarked
        against a **Decision Tree** and **Naive Bayes**, to estimate a patient's
        risk of heart disease from clinical measurements.
        """
    )

    df = load_dataset()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{len(df)}</h3>Patient Records</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{df["target"].mean()*100:.1f}%</h3>Disease Prevalence</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>{df["age"].mean():.0f}</h3>Average Age</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>Random Forest</h3>Core Algorithm</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.info("Use the sidebar to explore data visualizations or make a live prediction for a patient.")
    st.success("System status: Model loaded and ready for predictions.")


# ---------------------------------------------------------------------------
# Prediction Page
# ---------------------------------------------------------------------------
def get_risk_level(probability: float):
    """Map predicted probability of disease to a Low / Medium / High risk band."""
    if probability < 0.35:
        return "Low", "risk-low", "🟢"
    elif probability < 0.65:
        return "Medium", "risk-medium", "🟡"
    else:
        return "High", "risk-high", "🔴"


def prediction_page():
    st.markdown('<div class="main-header">Patient Disease Risk Prediction</div>', unsafe_allow_html=True)
    st.write("Fill in the patient's clinical details below and click **Predict** to assess heart disease risk.")

    try:
        model, scaler, selected_features = load_model_artifacts()
    except FileNotFoundError:
        st.error(
            "Trained model files were not found in the `models/` folder. "
            "Please run `python train_model.py` first, then restart this app."
        )
        return

    with st.form("patient_form"):
        st.markdown("#### Patient Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox(
                "Chest Pain Type",
                [0, 1, 2, 3],
                format_func=lambda x: {
                    0: "Typical Angina", 1: "Atypical Angina",
                    2: "Non-anginal Pain", 3: "Asymptomatic",
                }[x],
            )
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 120)

        with col2:
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 500, 200)
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)

        with col3:
            ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
            thal = st.selectbox(
                "Thalassemia",
                [0, 1, 2, 3],
                format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect", 3: "Other"}[x],
            )
            bmi = st.number_input("BMI", 10.0, 60.0, 24.0, step=0.1)

        submitted = st.form_submit_button("🔍 Predict Disease Risk", use_container_width=True)

    if submitted:
        sex_val = 1 if sex == "Male" else 0

        patient_data = {
            "age": age, "sex": sex_val, "cp": cp, "trestbps": trestbps,
            "chol": chol, "thalach": thalach, "exang": exang,
            "oldpeak": oldpeak, "ca": ca, "thal": thal, "bmi": bmi,
        }

        # Build the feature vector in the exact order the model was trained on
        input_df = pd.DataFrame([patient_data])[selected_features]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        risk_label, risk_css, risk_icon = get_risk_level(probability)

        st.markdown("---")
        st.markdown("### Prediction Result")

        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            if prediction == 1:
                st.warning(f"⚠️ The model predicts a **presence of heart disease risk factors**.")
            else:
                st.success("✅ The model predicts **no significant heart disease risk**.")
            st.metric("Predicted Probability of Disease", f"{probability*100:.1f}%")

        with res_col2:
            st.markdown(
                f'<div class="{risk_css}">{risk_icon} Risk Level: {risk_label}</div>',
                unsafe_allow_html=True,
            )

        st.caption(
            "Disclaimer: This tool is an academic mini-project demonstration only "
            "and is NOT a substitute for professional medical diagnosis."
        )


# ---------------------------------------------------------------------------
# Data Visualization Page
# ---------------------------------------------------------------------------
def visualization_page():
    st.markdown('<div class="main-header">Data Visualization</div>', unsafe_allow_html=True)
    st.write("Explore key patterns in the patient dataset used to train the models.")

    df = load_dataset()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Age Distribution", "Disease Count", "Gender Distribution", "Blood Pressure", "BMI Distribution"]
    )

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["age"], bins=20, color="#2E86AB", edgecolor="white")
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Age Distribution of Patients")
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["target"].value_counts().sort_index()
        ax.bar(["No Disease", "Disease"], counts.values, color=["#52B788", "#E76F51"])
        ax.set_ylabel("Number of Patients")
        ax.set_title("Disease Case Count")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(6, 4))
        gender_counts = df["gender"].value_counts()
        ax.pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%",
               colors=["#4CC9F0", "#F72585"], startangle=90)
        ax.set_title("Gender Distribution")
        st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["trestbps"], bins=20, color="#A23B72", edgecolor="white")
        ax.set_xlabel("Resting Blood Pressure (mm Hg)")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Blood Pressure Distribution")
        st.pyplot(fig)

    with tab5:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["bmi"], bins=20, color="#F18F01", edgecolor="white")
        ax.set_xlabel("BMI")
        ax.set_ylabel("Number of Patients")
        ax.set_title("BMI Distribution")
        st.pyplot(fig)


# ---------------------------------------------------------------------------
# About Page
# ---------------------------------------------------------------------------
def about_page():
    st.markdown('<div class="main-header">About This Project</div>', unsafe_allow_html=True)

    st.markdown(
        """
        ### 🎯 Objective
        To build a healthcare decision-support system that applies **data mining
        algorithms** to clinical data in order to assist in early identification
        of heart disease risk.

        ### 🧠 Algorithm Used
        - **Random Forest Classifier** (primary/main model) — an ensemble of decision
          trees that reduces overfitting and generally yields strong, stable accuracy.
        - **Decision Tree** and **Naive Bayes** are trained alongside it purely for
          performance comparison.

        ### 🛠️ Technologies Used
        - **Python** — core programming language
        - **Pandas / NumPy** — data preprocessing and numerical computation
        - **Scikit-learn** — machine learning models and evaluation metrics
        - **Matplotlib / Seaborn** — data visualization
        - **Streamlit** — interactive web dashboard
        - **Joblib** — model persistence

        ### 👨‍🎓 Developed By
        **Bhavesh Bargat**  
        Final-year Engineering Mini Project — AI & Data Science.

        ### ⚠️ Disclaimer
        This system is built for **academic demonstration purposes only** and must
        not be used for real medical diagnosis or treatment decisions.
        """
    )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    load_custom_css()
    page = sidebar_navigation()

    if page == "🏠 Home":
        home_page()
    elif page == "🩺 Prediction":
        prediction_page()
    elif page == "📊 Data Visualization":
        visualization_page()
    elif page == "ℹ️ About":
        about_page()


if __name__ == "__main__":
    main()
