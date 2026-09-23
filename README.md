# Intelligent Healthcare Disease Risk Prediction & Clinical Decision Support System

Educational final-year project based directly on the supplied project specification.

## Scope
This system predicts diabetes risk from the Pima Indians Diabetes dataset, exposes a FastAPI prediction service, and provides a Streamlit decision-support dashboard.

**Important:** This is an educational decision-support system, not a medical diagnosis tool.

## PDF requirement mapping

| PDF requirement | Implementation |
|---|---|
| Supervised classification | Binary diabetes-risk classification |
| Data cleaning | Duplicate checks + invalid-zero-to-missing handling |
| Missing values | SimpleImputer |
| Outlier analysis | IQR summary/report |
| Feature engineering | BMI Category, Age Group, Risk Factor Count |
| Encoding/scaling | OneHotEncoder + StandardScaler |
| Class imbalance | SMOTE on training data |
| Models | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, optional XGBoost |
| Tuning | GridSearchCV |
| Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity |
| Model extraction | `models/disease_pipeline.pkl` |
| Backend | FastAPI `/health`, `/model-info`, `/predict`, `/batch-predict` |
| Frontend | Streamlit dashboard, individual prediction, analytics, evaluation |
| Explainability | Feature importance + SHAP-ready architecture |
| Batch prediction | API endpoint and CSV-ready structure |
| Documentation | README, architecture and methodology |

## Dataset

Use the Pima Indians Diabetes CSV with:

`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

The dataset is intentionally not committed to Git by default. Download it with:

```bash
python training/download_dataset.py
```

The script uses the public dataset copy referenced in the project documentation.

## Setup

### Windows
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python training/download_dataset.py
python training/train.py
```

### Start backend
```powershell
uvicorn backend.app:app --reload
```

Swagger: `http://127.0.0.1:8000/docs`

### Start frontend
In another terminal:
```powershell
streamlit run frontend/streamlit_app.py
```

## Team workflow

```bash
git clone <YOUR_REPOSITORY_URL>
cd health-risk-prediction
git checkout -b feature/backend
git add .
git commit -m "Add backend prediction service"
git push -u origin feature/backend
```

Use separate branches for preprocessing, ML, backend, frontend, and documentation, then merge through Pull Requests.

## Presentation flow

1. Problem statement
2. Proposed solution
3. Dataset and features
4. ML pipeline
5. Feature engineering
6. Model comparison
7. Evaluation metrics and false-positive/false-negative trade-off
8. System architecture
9. Live prediction
10. Explainability
11. Population analytics / batch prediction
12. Limitations and future scope

## Limitations
- The chosen public dataset is small and population-specific.
- Some physiological missing values are represented by zero and need preprocessing.
- The dataset does not contain every example feature listed in the PDF, such as smoking, diet, family history and gender.
- Risk bands in the application are probability bands for demonstration, not clinical diagnostic thresholds.
