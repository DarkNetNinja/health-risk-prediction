from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "diabetes.csv"
OUTPUTS = BASE / "outputs"

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

def load_data(path=DATA):
    df = pd.read_csv(path)
    required = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df

def clean_data(df):
    out = df.copy()
    duplicate_count = int(out.duplicated().sum())
    out = out.drop_duplicates().reset_index(drop=True)
    for col in ZERO_AS_MISSING:
        out[col] = out[col].replace(0, np.nan)
    return out, duplicate_count

def add_engineered_features(df):
    out = df.copy()
    out["BMI_Category"] = pd.cut(
        out["BMI"], [-np.inf,18.5,25,30,np.inf],
        labels=["Underweight","Normal","Overweight","Obese"]
    ).astype("object")
    out["Age_Group"] = pd.cut(
        out["Age"], [-np.inf,29,44,np.inf],
        labels=["Young","Middle_Age","Senior"]
    ).astype("object")
    # Application-level engineered feature, not a clinical diagnostic rule.
    out["Risk_Factor_Count"] = (
        (out["BMI"] >= 30).astype(int) +
        (out["Glucose"] >= 126).astype(int) +
        (out["BloodPressure"] >= 80).astype(int) +
        (out["Age"] >= 45).astype(int) +
        (out["Pregnancies"] >= 5).astype(int)
    )
    return out

def save_eda_summary(df, duplicate_count):
    OUTPUTS.mkdir(exist_ok=True)
    summary = {
        "records_after_duplicate_removal": int(len(df)),
        "duplicate_rows_removed": duplicate_count,
        "class_distribution": df["Outcome"].value_counts().sort_index().to_dict(),
        "missing_values_after_zero_conversion": df.isna().sum().to_dict(),
    }
    (OUTPUTS/"eda_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
