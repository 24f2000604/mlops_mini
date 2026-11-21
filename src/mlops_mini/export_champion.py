import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "serving_model"

mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")

model = mlflow.sklearn.load_model("models:/trees@champion")

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

mlflow.sklearn.save_model(model, str(OUT_DIR), serialization_format="cloudpickle")
print(f"Exported @champion to {OUT_DIR}")
