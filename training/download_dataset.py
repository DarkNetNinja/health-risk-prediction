from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
OUT = Path(__file__).resolve().parents[1] / "data" / "diabetes.csv"
HEADER = "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome\n"

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = urlopen(URL, timeout=30).read().decode("utf-8")
    OUT.write_text(HEADER + raw, encoding="utf-8")
    print(f"Saved dataset to {OUT}")

if __name__ == "__main__":
    main()
