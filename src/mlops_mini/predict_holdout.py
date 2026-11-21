from pathlib import Path

import mlflow
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = REPO_ROOT / "data" / "raw" / "holdout_features.csv"
OUT_DIR = REPO_ROOT / "predictions"
OUT_CSV = OUT_DIR / "holdout_predictions.csv"

FEATURES = [
    "cows",
    "buffaloes",
    "fodder_price_rs",
    "temp_c",
    "vet_visits",
    "feed_kg",
    "breed_mix",
    "village_cluster",
]

mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")


def main():
    holdout = pd.read_csv(HOLDOUT_CSV)
    model = mlflow.pyfunc.load_model("models:/trees@champion")
    preds = model.predict(holdout[FEATURES])

    out = pd.DataFrame(
        {"record_id": holdout["record_id"], "prediction": preds}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
