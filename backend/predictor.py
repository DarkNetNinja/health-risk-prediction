from pathlib import Path
import json
import joblib
import pandas as pd
import shap

from training.preprocessing import add_engineered_features


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "disease_pipeline.pkl"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"


class DiseasePredictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.background = self._load_background()

    def _load_background(self):
        path = BASE_DIR / "outputs" / "shap_background.csv"

        if path.exists():
            return pd.read_csv(path)

        return None

    def prepare_input(self, patient: dict) -> pd.DataFrame:

        df = pd.DataFrame([{
            "Pregnancies": patient["pregnancies"],
            "Glucose": patient["glucose"],
            "BloodPressure": patient["blood_pressure"],
            "SkinThickness": patient["skin_thickness"],
            "Insulin": patient["insulin"],
            "BMI": patient["bmi"],
            "DiabetesPedigreeFunction": patient["diabetes_pedigree_function"],
            "Age": patient["age"],
        }])

        df = add_engineered_features(df)

        return df

    def predict(self, patient: dict):
        print("PIPELINE STEPS:", self.model.named_steps)

        X = self.prepare_input(patient)

        prediction = int(self.model.predict(X)[0])

        probability = float(
            self.model.predict_proba(X)[0][1]
        )

        risk_level = self.get_risk_level(probability)

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
        }

    def explain(self, patient: dict):

        X = self.prepare_input(patient)

        prediction = int(self.model.predict(X)[0])

        probability = float(
            self.model.predict_proba(X)[0][1]
        )

        risk_level = self.get_risk_level(probability)

        # Get preprocessing and final classifier from pipeline
        preprocessor = self.model.named_steps["preprocess"]
        classifier = self.model.named_steps["model"]

        # Convert all features to numeric encoded representation
        X_transformed = preprocessor.transform(X)

        # Get transformed feature names
        feature_names = preprocessor.get_feature_names_out()

        # SHAP on the numeric classifier input
        explainer = shap.Explainer(
            classifier.predict_proba,
            X_transformed,
            algorithm="permutation"
        )

        shap_values = explainer(X_transformed)

        values = shap_values.values[0]

        # Class 1 = positive/risk class
        if len(values.shape) > 1:
            values = values[:, 1]

        factors = []

        for feature, value in zip(feature_names, values):
            factors.append({
                "feature": feature,
                "impact": float(value)
            })

        increasing = sorted(
            [x for x in factors if x["impact"] > 0],
            key=lambda x: x["impact"],
            reverse=True
        )

        reducing = sorted(
            [x for x in factors if x["impact"] < 0],
            key=lambda x: x["impact"]
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "explanation_method": "SHAP PermutationExplainer",
            "factors_increasing_risk": increasing[:5],
            "factors_reducing_risk": reducing[:5],
            "explanation_note": (
                "Positive SHAP values increase the predicted probability "
                "of the positive class, while negative values decrease it."
            ),
        }

    def get_risk_level(self, probability: float) -> str:

        if probability < 0.33:
            return "LOW"

        if probability < 0.66:
            return "MEDIUM"

        return "HIGH"


predictor = DiseasePredictor()