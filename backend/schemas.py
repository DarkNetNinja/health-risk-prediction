from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    pregnancies: int = Field(..., ge=0, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, description="Plasma glucose concentration")
    blood_pressure: float = Field(..., ge=0, description="Diastolic blood pressure")
    skin_thickness: float = Field(..., ge=0, description="Triceps skin fold thickness")
    insulin: float = Field(..., ge=0, description="Serum insulin")
    bmi: float = Field(..., ge=0, description="Body mass index")
    diabetes_pedigree_function: float = Field(
        ..., ge=0, description="Diabetes pedigree function"
    )
    age: int = Field(..., ge=1, description="Age in years")


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str