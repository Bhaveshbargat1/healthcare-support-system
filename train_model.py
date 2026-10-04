"""
train_model.py
----------------
Healthcare Support System using Data Mining Algorithms.

This script performs the complete data mining pipeline:
    1. Load the heart disease dataset
    2. Preprocess it (missing values, duplicates, encoding, feature selection)
    3. Split into train/test sets
    4. Train three classifiers: Random Forest (main model), Decision Tree,
       and Naive Bayes
    5. Evaluate and compare them (accuracy, confusion matrix, classification report)
    6. Save the best model + scaler + feature list to disk using Joblib

Run this file first, before app.py, so trained model artifacts exist:
    python train_model.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.feature_selection import SelectKBest, f_classif

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart_disease.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1: Load Data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    print("Loading dataset...")
    df = pd.read_csv(path)
    print(f"Initial shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# STEP 2: Data Preprocessing
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame):
    """Handle missing values, duplicates, encoding, and feature selection."""

    print("\n--- Data Preprocessing ---")

    # 2a. Handle missing values: fill numeric columns with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_before = df.isnull().sum().sum()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    print(f"Missing values handled: {missing_before} -> {df.isnull().sum().sum()}")

    # 2b. Remove duplicate records
    dup_before = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {dup_before}")

    # 2c. Encode categorical columns (the 'gender' text column -> numeric)
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 0, "Male": 1})
    print("Categorical columns encoded (gender -> 0/1)")

    # 2d. Define features (X) and target (y)
    # Drop 'gender' since 'sex' already encodes the same information numerically
    feature_cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "bmi",
    ]
    X = df[feature_cols]
    y = df["target"]

    # 2e. Feature selection - keep the top K most statistically significant features
    k = 10
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Feature selection: kept top {k} features -> {selected_features}")

    X_final = X[selected_features]

    return X_final, y, selected_features


# ---------------------------------------------------------------------------
# STEP 3: Train / Test Split + Scaling
# ---------------------------------------------------------------------------
def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# STEP 4: Train & Evaluate Multiple Models
# ---------------------------------------------------------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(report)

        results[name] = {"accuracy": acc, "confusion_matrix": cm, "report": report}
        trained_models[name] = model

        # Save confusion matrix plot for each model
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Disease", "Disease"],
                    yticklabels=["No Disease", "Disease"])
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        fname = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(IMAGES_DIR, f"confusion_matrix_{fname}.png"), dpi=120)
        plt.close()

    return results, trained_models


# ---------------------------------------------------------------------------
# STEP 5: Compare Models
# ---------------------------------------------------------------------------
def compare_models(results: dict):
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, accuracies, color=["#2E86AB", "#A23B72", "#F18F01"])
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{acc:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"), dpi=120)
    plt.close()

    print("\n--- Model Comparison Summary ---")
    for n in names:
        print(f"{n}: {results[n]['accuracy']*100:.2f}%")

    best_model_name = max(results, key=lambda n: results[n]["accuracy"])
    print(f"\nBest performing model: {best_model_name}")
    return best_model_name


# ---------------------------------------------------------------------------
# STEP 6: Save Artifacts with Joblib
# ---------------------------------------------------------------------------
def save_artifacts(trained_models, best_model_name, scaler, selected_features):
    # Save the Random Forest specifically as the "main" production model,
    # per the project requirement, alongside the scaler and feature list.
    joblib.dump(trained_models["Random Forest"], os.path.join(MODELS_DIR, "random_forest_model.pkl"))
    joblib.dump(trained_models["Decision Tree"], os.path.join(MODELS_DIR, "decision_tree_model.pkl"))
    joblib.dump(trained_models["Naive Bayes"], os.path.join(MODELS_DIR, "naive_bayes_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(selected_features, os.path.join(MODELS_DIR, "selected_features.pkl"))
    joblib.dump(best_model_name, os.path.join(MODELS_DIR, "best_model_name.pkl"))

    print(f"\nAll model artifacts saved to: {MODELS_DIR}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    df = load_data(DATASET_PATH)
    X, y, selected_features = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    results, trained_models = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_model_name = compare_models(results)
    save_artifacts(trained_models, best_model_name, scaler, selected_features)
    print("\nTraining pipeline complete. You can now run the Streamlit app:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()
