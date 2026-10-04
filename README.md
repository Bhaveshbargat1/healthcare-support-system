# 🏥 Healthcare Support System using Data Mining Algorithm

A final-year AI & Data Science engineering mini project that applies data
mining and machine learning techniques to clinical data in order to assist
in the early identification of heart disease risk, presented through an
interactive Streamlit dashboard.

---

## 📌 Project Description

Cardiovascular disease is one of the leading causes of death worldwide, and
early risk detection can meaningfully improve patient outcomes. This project
builds a **Healthcare Support System** that takes a patient's clinical
measurements (age, blood pressure, cholesterol, ECG results, etc.) and uses
a trained **data mining model** to estimate their risk of heart disease,
categorizing it into **Low, Medium, or High** risk.

The system demonstrates a complete, real-world-style data mining workflow:
data cleaning, feature selection, model training, multi-algorithm
comparison, evaluation, and deployment through a web dashboard.

---

## 🎯 Objectives

- Apply data preprocessing techniques to clean a real-world-style healthcare dataset.
- Train and compare multiple data mining algorithms for disease classification.
- Identify the best-performing algorithm using standard evaluation metrics.
- Build an interactive, user-friendly dashboard for real-time prediction.
- Visualize key health indicators through charts for exploratory analysis.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas / NumPy | Data loading, cleaning, and numerical computation |
| Scikit-learn | Machine learning models, feature selection, evaluation metrics |
| Matplotlib / Seaborn | Data visualization and evaluation charts |
| Streamlit | Interactive web dashboard |
| Joblib | Saving and loading trained models |

---

## 🧠 Algorithm Explanation

### Random Forest Classifier (Main Algorithm)
Random Forest builds a large number of decision trees on random subsets of
the data and features, then combines their predictions by majority vote.
This "ensemble" approach makes it more accurate and far less prone to
overfitting than a single decision tree, which is why it is used as the
core prediction engine of this system.

### Decision Tree (Comparison Model)
A single tree-based classifier that splits data based on feature thresholds.
It is simple and interpretable but more prone to overfitting than Random
Forest.

### Naive Bayes (Comparison Model)
A probabilistic classifier based on Bayes' theorem, assuming feature
independence. It is fast and works well as a baseline for comparison.

All three models are trained on the same data and compared using accuracy,
confusion matrices, and classification reports, so you can see empirically
why the chosen algorithm performs the way it does.

---

## ✨ Features

- 🏠 Attractive, informative homepage with dataset summary metrics
- 🖼️ Custom healthcare logo and professional styling
- 📌 Sidebar navigation (Home / Prediction / Data Visualization / About)
- 📝 Interactive patient data input form
- 🔍 One-click disease risk prediction
- 🟢🟡🔴 Color-coded Low / Medium / High risk indicator
- ✅ Success and ⚠️ warning result messages
- 📊 Interactive charts: age distribution, disease count, gender
  distribution, blood pressure distribution, BMI distribution
- 💾 Trained models persisted with Joblib for instant reuse (no retraining needed)

---

## 📁 Project Structure

```
healthcare-support-system/
│
├── dataset/
│   ├── generate_dataset.py     # Generates the synthetic heart disease dataset
│   └── heart_disease.csv       # The dataset used for training and visualization
│
├── models/
│   ├── random_forest_model.pkl # Main trained model
│   ├── decision_tree_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── scaler.pkl              # Fitted StandardScaler
│   ├── selected_features.pkl   # Feature columns used by the model
│   └── best_model_name.pkl
│
├── images/
│   ├── logo.png                          # Dashboard logo
│   ├── model_comparison.png              # Accuracy comparison chart
│   ├── confusion_matrix_random_forest.png
│   ├── confusion_matrix_decision_tree.png
│   └── confusion_matrix_naive_bayes.png
│
├── app.py             # Streamlit dashboard (run this to use the app)
├── train_model.py     # Full training + evaluation pipeline
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## ⚙️ Installation & Setup (VS Code)

### Step 1 — Open the project
Open the `healthcare-support-system` folder in VS Code
(`File > Open Folder...`).

### Step 2 — Create a virtual environment (recommended)
Open the VS Code integrated terminal (`` Ctrl+` ``) and run:

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **macOS / Linux:** `source venv/bin/activate`

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Generate the dataset (already included, but you can regenerate it)
```bash
python dataset/generate_dataset.py
```

> 💡 **Tip:** For a stronger academic submission, replace
> `dataset/heart_disease.csv` with the real UCI Heart Disease dataset
> (search "Heart Disease UCI" on Kaggle). Since the column names match,
> no other code needs to change.

### Step 5 — Train the models
```bash
python train_model.py
```
This will print accuracy, confusion matrices, and classification reports
for all three algorithms, and save the trained models into `models/`.

### Step 6 — Run the dashboard
```bash
streamlit run app.py
```
Your default browser will open automatically at `http://localhost:8501`.

---

## 📊 Model Performance (on the included synthetic dataset)

| Model | Accuracy |
|---|---|
| Random Forest | ~74–78% |
| Decision Tree | ~70–74% |
| Naive Bayes | ~76–79% |

*(Exact figures will vary slightly by run/dataset. See `images/model_comparison.png`
and the console output of `train_model.py` for your actual results.)*

---

## 📸 Screenshots

> Add your dashboard screenshots here after running the app locally, e.g.:
>
> - `images/screenshot_home.png` — Homepage
> - `images/screenshot_prediction.png` — Prediction form and result
> - `images/screenshot_visualization.png` — Data visualization tabs

---

## 🚀 Future Scope

- Integrate a real, larger clinical dataset (e.g., full UCI/Kaggle Heart Disease dataset).
- Extend to multiple disease modules (Diabetes, Kidney Disease, Liver Disease).
- Add deep learning models (e.g., ANN) for comparison.
- Deploy the dashboard on Streamlit Community Cloud / Render / Heroku.
- Add user authentication and patient history tracking with a database.
- Integrate SHAP/LIME for explainable AI, so predictions show *why* a
  patient was flagged at a given risk level.
- Add multi-language support for wider accessibility.

---

## ⚠️ Disclaimer

This project is developed strictly for **academic and educational purposes**
as part of an engineering mini project. It is **not** a certified medical
diagnostic tool and must not be used for real clinical decision-making.

---

## 👨‍🎓 Author

**Bhavesh Bargat**  
Final Year Engineering Student — AI & Data Science  
Mini Project: *Healthcare Support System using Data Mining Algorithm*
