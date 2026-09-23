# 🩺 Intelligent Healthcare Disease Risk Prediction

An explainable machine-learning based healthcare risk prediction system built using Python, Scikit-learn, FastAPI, Streamlit and SHAP.

> **Important:** This project is intended for educational and research purposes. It is a machine-learning risk estimation system and is **not a medical diagnosis tool**.

---

## 📌 Project Overview

This project predicts diabetes-related disease risk from structured patient health information.

The system provides:

* Individual risk prediction
* Probability of predicted risk
* LOW / MEDIUM / HIGH application-level risk category
* Batch prediction using CSV files
* SHAP-based explainability
* Population analytics
* Model evaluation
* ROC and Precision-Recall curves
* Model comparison
* Interactive Streamlit dashboard
* REST API using FastAPI

---

## 🏗️ System Architecture

```text
                    Patient Input
                         │
                         ▼
                 ┌───────────────┐
                 │   Streamlit   │
                 │   Frontend    │
                 └───────┬───────┘
                         │
                         ▼
                 ┌───────────────┐
                 │    FastAPI    │
                 │      API      │
                 └───────┬───────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Feature Engineering  │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Preprocessing + SMOTE│
              └──────────┬───────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │   SVM Model   │
                 └───────┬───────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Prediction              Probability
              │                     │
              └──────────┬──────────┘
                         ▼
                    Risk Level
                         │
                         ▼
                 SHAP Explanation
```

---

## 📂 Project Structure

```text
health-risk-prediction/
│
├── backend/
│   ├── app.py
│   ├── predictor.py
│   └── schemas.py
│
├── data/
│   └── diabetes.csv
│
├── frontend/
│   └── streamlit_app.py
│
├── models/
│   ├── disease_pipeline.pkl
│   └── model_metadata.json
│
├── notebooks/
│
├── outputs/
│   ├── eda_summary.json
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   ├── pr_curve.csv
│   ├── roc_curve.csv
│   ├── shap_background.csv
│   └── test_predictions.csv
│
├── training/
│   ├── download_dataset.py
│   ├── preprocessing.py
│   └── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/DarkNetNinja/health-risk-prediction.git
```

Then:

```bash
cd health-risk-prediction
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install dependencies

```powershell
pip install -r requirements.txt
```

---

# 🧠 Model Training

The repository already contains the trained model artifacts.

If you want to retrain the project from the dataset:

```powershell
python training/download_dataset.py
```

Then:

```powershell
python training/train.py
```

This generates the model and evaluation artifacts.

---

# ⚡ Run the FastAPI Backend

From the project root:

```powershell
uvicorn backend.app:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

FastAPI documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🔌 API Endpoints

### Health Check

```text
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model": "SVM"
}
```

### Model Information

```text
GET /model-info
```

Returns:

* selected model
* features
* evaluation metrics
* risk bands

### Individual Prediction

```text
POST /predict
```

Example input:

```json
{
  "pregnancies": 2,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 25,
  "insulin": 100,
  "bmi": 28.5,
  "diabetes_pedigree_function": 0.35,
  "age": 30
}
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.42908175019048794,
  "risk_level": "MEDIUM"
}
```

### Batch Prediction

```text
POST /batch-predict
```

The Streamlit application can upload a CSV and send multiple patient records to this endpoint.

### Explainable Prediction

```text
POST /explain
```

Uses SHAP PermutationExplainer to provide feature-level explanations for the model prediction.

---

# 🖥️ Run the Streamlit Frontend

Keep FastAPI running.

Open a **second terminal**.

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

Run:

```powershell
streamlit run frontend/streamlit_app.py
```

The application will open at:

```text
http://localhost:8501
```

---

# 📊 Streamlit Features

## Project Dashboard

Displays:

* Evaluation records
* Positive cases
* Average predicted risk
* High-risk predictions
* Selected model
* Model comparison

## Individual Risk Prediction

Enter patient information and receive:

* Prediction
* Probability
* Risk category

## Explainable Prediction

Uses SHAP to show factors that increase or decrease the predicted probability.

## Batch Prediction

Upload a CSV containing:

```text
patient_id
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age
```

Example:

```csv
patient_id,Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
P001,2,120,70,25,100,28.5,0.35,30
P002,5,150,80,30,150,32.1,0.50,45
P003,1,90,60,20,80,24.5,0.20,25
```

The application generates predictions and allows downloading the results as CSV.

## Population Analytics

Displays:

* Predicted risk distribution
* Actual disease distribution

## Model Evaluation

Displays:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Sensitivity
* Specificity
* ROC curve
* Precision-Recall curve
* Feature importance

---

# 🤖 Machine Learning Pipeline

The project evaluates multiple classification algorithms:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machine
* XGBoost

The training pipeline includes:

```text
Data Cleaning
     ↓
Missing Value Handling
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Feature Scaling
     ↓
Train/Test Split
     ↓
SMOTE
     ↓
Model Training
     ↓
Hyperparameter Tuning
     ↓
Cross Validation
     ↓
Model Comparison
     ↓
Final Model
```

The final model in the current trained artifacts is **SVM**, selected based on the highest ROC-AUC among the evaluated models.

---

# 🧪 Evaluation

The current trained model achieves approximately:

| Metric      |    SVM |
| ----------- | -----: |
| Accuracy    | 74.68% |
| Precision   | 62.30% |
| Recall      | 70.37% |
| F1          | 66.09% |
| ROC-AUC     | 82.85% |
| Sensitivity | 70.37% |
| Specificity | 77.00% |

These values are based on the project's held-out test set.

---

# 🧩 Feature Engineering

Additional features are generated from the original patient inputs:

* BMI Category
* Age Group
* Risk Factor Count

The application automatically generates these features before prediction.

---

# ⚠️ Risk Categories

The application uses probability-based application categories:

```text
0% – <33%     LOW
33% – <66%    MEDIUM
66% – 100%    HIGH
```

These are **application-level probability bands and are not clinical diagnostic thresholds**.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* XGBoost
* SHAP
* FastAPI
* Uvicorn
* Pydantic
* Streamlit
* Plotly
* Joblib

---

# 👥 Team Setup

Each team member should:

1. Clone the repository
2. Create a virtual environment
3. Install requirements
4. Start FastAPI
5. Start Streamlit

### Terminal 1

```powershell
cd health-risk-prediction
.venv\Scripts\Activate.ps1
uvicorn backend.app:app --reload
```

### Terminal 2

```powershell
cd health-risk-prediction
.venv\Scripts\Activate.ps1
streamlit run frontend/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

---

# 🩺 Disclaimer

This project is developed for educational, academic and research purposes.

It should not be used as a substitute for professional medical advice, diagnosis, or treatment.

Model predictions represent machine-learning estimates based on the training dataset and should not be interpreted as clinical conclusions.
