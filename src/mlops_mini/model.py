import hashlib
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
PARAMS_PATH = REPO_ROOT / "params.yaml"
TRAIN_CSV = REPO_ROOT / "data" / "raw" / "train.csv"

sys.path.insert(0, str(SRC_DIR))
from utils.clean_data import clean_data

mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")

with open(PARAMS_PATH, "r") as f:
    params = yaml.safe_load(f) or {}

train_df = pd.read_csv(TRAIN_CSV)
df = clean_data(train_df)

X = df.drop(columns=["litres_per_day"])
y = df["litres_per_day"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["models"]["SPLIT"],
    random_state=params["models"]["SEED"],
)

num_cols = [
    "cows",
    "buffaloes",
    "fodder_price_rs",
    "temp_c",
    "vet_visits",
    "feed_kg",
]
cat_cols = ["breed_mix", "village_cluster"]

data_sha256 = hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest()

mlflow.set_experiment("exp_1")

MODEL_NAME = "trees"
trees = ["tree", "tree_2", "tree_3"]

best_mae = float("inf")
best_model_uri = None
best_run_id = None
best_variant = ""


def build_pipeline(variant):
    num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    cat_pipeline = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)],
        remainder="drop",
    )
    tree_model = DecisionTreeRegressor(
        **params["models"][variant],
        random_state=params["models"]["SEED"],
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", tree_model)])


for tree in trees:
    pipeline = build_pipeline(tree)
    with mlflow.start_run(run_name=f"v1_{tree}") as run:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))

        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_variant", tree)
        mlflow.log_params(params["models"][tree])
        mlflow.log_metric("mae", mae)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            serialization_format="cloudpickle",
        )

        print(f"[{tree}] run {run.info.run_id} | MAE: {mae:.4f}")

        if mae < best_mae:
            best_mae = mae
            best_model_uri = model_info.model_uri
            best_run_id = run.info.run_id
            best_variant = tree

client = MlflowClient()
registered = mlflow.register_model(model_uri=best_model_uri, name=MODEL_NAME)
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version=registered.version,
)

print(
    f"\nv1 winner: {best_variant} (run {best_run_id}) -> "
    f"trees v{registered.version} @champion | MAE: {best_mae:.4f}"
)
