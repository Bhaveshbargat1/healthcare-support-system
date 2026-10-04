"""
generate_dataset.py
--------------------
Generates a realistic synthetic Heart Disease dataset, modeled on the
column structure and statistical ranges of the well-known UCI Heart
Disease dataset.

WHY A GENERATED DATASET?
This script is included so the project runs end-to-end with zero setup.
For a real academic submission, it is strongly recommended to replace
dataset/heart_disease.csv with the actual UCI Heart Disease dataset
(freely available on Kaggle: search "Heart Disease UCI"). Because this
generator uses the exact same column names, train_model.py and app.py
will work unchanged with the real dataset too.

The script deliberately injects some missing values and duplicate rows
so that the preprocessing pipeline in train_model.py has genuine,
demonstrable work to do.
"""

import numpy as np
import pandas as pd
import os

# Reproducibility
np.random.seed(42)

N_SAMPLES = 1000


def generate_heart_disease_data(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Generate a synthetic but medically-plausible heart disease dataset."""

    age = np.random.randint(28, 80, n_samples)
    sex = np.random.randint(0, 2, n_samples)  # 0 = Female, 1 = Male
    cp = np.random.randint(0, 4, n_samples)  # chest pain type (0-3)
    trestbps = np.random.randint(90, 200, n_samples)  # resting blood pressure
    chol = np.random.randint(120, 400, n_samples)  # serum cholesterol
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # fasting blood sugar > 120mg/dl
    restecg = np.random.randint(0, 3, n_samples)  # resting ECG results
    thalach = np.random.randint(70, 202, n_samples)  # max heart rate achieved
    exang = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])  # exercise induced angina
    oldpeak = np.round(np.random.uniform(0, 6.2, n_samples), 1)  # ST depression
    slope = np.random.randint(0, 3, n_samples)  # slope of peak exercise ST segment
    ca = np.random.randint(0, 5, n_samples)  # number of major vessels colored by fluoroscopy
    thal = np.random.randint(0, 4, n_samples)  # thalassemia
    bmi = np.round(np.random.uniform(16, 45, n_samples), 1)  # body mass index

    # Build a realistic disease "risk score" so the target is genuinely
    # learnable from the features (not random), which is what lets our
    # models achieve meaningful, explainable accuracy.
    risk_score = (
        (age > 54) * 1.4
        + (sex == 1) * 0.8
        + (cp >= 2) * 1.1
        + (trestbps > 140) * 1.0
        + (chol > 240) * 1.0
        + fbs * 0.5
        + (thalach < 120) * 1.2
        + exang * 1.3
        + (oldpeak > 2) * 1.3
        + (ca >= 1) * 1.4
        + (thal >= 2) * 1.0
        + (bmi > 30) * 0.9
        + np.random.normal(0, 1.5, n_samples)  # natural randomness
    )

    threshold = np.percentile(risk_score, 55)  # ~45% positive cases, realistic class balance
    target = (risk_score > threshold).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "bmi": bmi,
        "target": target,
    })

    # --- Inject realistic "messiness" for preprocessing demonstration ---

    # 1) Missing values in a few columns (simulates incomplete patient records)
    for col in ["chol", "trestbps", "thalach", "bmi"]:
        missing_idx = np.random.choice(df.index, size=int(0.03 * n_samples), replace=False)
        df.loc[missing_idx, col] = np.nan

    # 2) Duplicate rows (simulates duplicate hospital entries)
    duplicate_rows = df.sample(n=20, random_state=1)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    # 3) Categorical text column for sex, to demonstrate encoding later
    df["gender"] = df["sex"].map({0: "Female", 1: "Male"})

    # Shuffle rows
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    return df


if __name__ == "__main__":
    data = generate_heart_disease_data()
    out_path = os.path.join(os.path.dirname(__file__), "heart_disease.csv")
    data.to_csv(out_path, index=False)
    print(f"Dataset generated: {out_path}")
    print(f"Shape: {data.shape}")
    print(f"Missing values:\n{data.isnull().sum()}")
    print(f"Duplicate rows: {data.duplicated().sum()}")
