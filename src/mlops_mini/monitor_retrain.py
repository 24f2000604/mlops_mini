import hashlib
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
PARAMS_PATH = REPO_ROOT / "params.yaml"
TRAIN_CSV = REPO_ROOT / "data" / "raw" / "train.csv"
LATER_CSV = REPO_ROOT / "data" / "raw" / "later.csv"

sys.path.insert(0, str(SRC_DIR))
from utils.clean_data import clean_data

mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")

NUM_COLS = ["cows", "buffaloes", "fodder_price_rs", "temp_c", "vet_visits", "feed_kg"]
CAT_COLS = ["breed_mix", "village_cluster"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "litres_per_day"
GROUP = "village_cluster"
MODEL_NAME = "trees"
BINS = 10
THRESHOLD = 0.12


def psi(expected, actual, bins=BINS):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0] = -np.inf
    edges[-1] = np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.where(e == 0, 1e-6, e)
    a = np.where(a == 0, 1e-6, a)
    return float(np.sum((a - e) * np.log(a / e)))


def build_pipeline(variant_params, seed):
    num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    cat_pipeline = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipeline, NUM_COLS), ("cat", cat_pipeline, CAT_COLS)],
        remainder="drop",
    )
    model = DecisionTreeRegressor(**variant_params, random_state=seed)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    with open(PARAMS_PATH, "r") as f:
        params = yaml.safe_load(f) or {}
    seed = params["models"]["SEED"]

    train_clean = clean_data(pd.read_csv(TRAIN_CSV))
    later = pd.read_csv(LATER_CSV)

    client = MlflowClient()
    champion_v1 = client.get_model_version_by_alias(MODEL_NAME, "champion")
    v1_run = client.get_run(champion_v1.run_id)
    variant = v1_run.data.params.get("model_variant", "tree_2")
    variant_params = params["models"][variant]

    psi_values = {c: psi(train_clean[c].values, later[c].values) for c in NUM_COLS}
    most_drifted = max(psi_values, key=psi_values.get)

    v1 = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    later_pred = v1.predict(later[FEATURES])
    later_eval = later.assign(abs_err=np.abs(later_pred - later[TARGET].values))
    mae_by_group = later_eval.groupby(GROUP)["abs_err"].mean().to_dict()
    worst_group = max(mae_by_group, key=mae_by_group.get)

    later_sorted = later.sort_values("date").reset_index(drop=True)
    cut = int(len(later_sorted) * 0.70)
    later_fit = later_sorted.iloc[:cut]
    later_check = later_sorted.iloc[cut:]

    X_v2 = pd.concat([train_clean[FEATURES], later_fit[FEATURES]], ignore_index=True)
    y_v2 = pd.concat(
        [train_clean[TARGET], later_fit[TARGET]], ignore_index=True
    )

    v2_pipeline = build_pipeline(variant_params, seed)
    v2_pipeline.fit(X_v2, y_v2)

    X_check = later_check[FEATURES]
    y_check = later_check[TARGET].values
    mae_v1 = float(mean_absolute_error(y_check, v1.predict(X_check)))
    mae_v2 = float(mean_absolute_error(y_check, v2_pipeline.predict(X_check)))

    improvement = (mae_v1 - mae_v2) / mae_v1
    decision = "promote" if improvement >= THRESHOLD else "keep"

    data_sha256 = hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest()
    mlflow.set_experiment("exp_1")
    with mlflow.start_run(run_name="v2_retrain"):
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_variant", variant)
        mlflow.log_param("retrain_rows", len(X_v2))
        mlflow.log_metric("mae_v1_on_check", mae_v1)
        mlflow.log_metric("mae", mae_v2)
        mlflow.log_metric("mae_v2_on_check", mae_v2)
        info = mlflow.sklearn.log_model(
            sk_model=v2_pipeline, name="model", serialization_format="cloudpickle"
        )
    v2 = mlflow.register_model(model_uri=info.model_uri, name=MODEL_NAME)

    if decision == "promote":
        client.set_registered_model_alias(MODEL_NAME, "champion", v2.version)
        champion_version = v2.version
    else:
        champion_version = champion_v1.version

    print("PSI (clean train vs later, 10 bins):")
    for c, v in psi_values.items():
        print(f"  {c:18s} {v:.4f}")
    print("most_drifted_feature:", most_drifted)
    print("\nmae_by_group (v1 on later):")
    for g, v in mae_by_group.items():
        print(f"  {g}: {v:.3f}")
    print("worst_served_group:", worst_group)
    print(f"\nmae_v1_on_check: {mae_v1:.4f}")
    print(f"mae_v2_on_check: {mae_v2:.4f}")
    print(f"improvement: {improvement * 100:.2f}% (need >= {THRESHOLD * 100:.0f}%)")
    print("decision:", decision)
    print(f"v1 version: {champion_v1.version} | v2 version: {v2.version}")
    print(f"champion_version after decision: {champion_version}")
    print("experiment_name: exp_1")
    print("best_run_id (v1):", champion_v1.run_id)


if __name__ == "__main__":
    main()
