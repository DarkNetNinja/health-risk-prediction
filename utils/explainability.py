from pathlib import Path
import pandas as pd

def load_feature_importance(base_dir):
    path=Path(base_dir)/"outputs/feature_importance.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["Feature","Importance"])
