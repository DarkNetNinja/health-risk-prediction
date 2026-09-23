from fastapi import FastAPI, HTTPException
from backend.schemas import PatientInput, PredictionResponse
from backend.predictor import predictor

app = FastAPI(
    title="Health Risk Prediction API",
    description=(
        "Machine-learning based diabetes risk estimation API. "
        "This system is intended for educational and research use "
        "and is not a medical diagnostic system."
    ),
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": predictor.metadata["best_model"],
    }


@app.get("/model-info")
def model_info():
    return {
        "model": predictor.metadata["best_model"],
        "features": predictor.metadata["feature_columns"],
        "target": predictor.metadata["target"],
        "metrics": predictor.metadata["test_metrics"],
        "risk_bands": predictor.metadata["risk_bands"],
        "note": predictor.metadata["note"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    try:
        return predictor.predict(patient.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


@app.post("/batch-predict")
def batch_predict(data: dict):
    try:
        patients = data["patients"]
        results = []

        for patient in patients:
            result = predictor.predict(patient)

            results.append({
                "patient_id": patient["patient_id"],
                "prediction": result["prediction"],
                "probability": result["probability"],
                "risk_level": result["risk_level"],
            })

        return {
            "count": len(results),
            "results": results,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(exc)}",
        )


@app.post("/explain")
def explain(patient: PatientInput):
    try:
        return predictor.explain(patient.model_dump())

    except Exception as exc:
        print(f"EXPLAIN ERROR: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(exc)}",
        )