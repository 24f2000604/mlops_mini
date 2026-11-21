from utils.clean_data import clean_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import yaml
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")

with open(r"../../params.yaml", "r") as f:
    params = yaml.safe_load(f) or {}
train_df = pd.read_csv("../data/raw/train.csv")
df = clean_data(train_df)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params.get("models").get("SPLIT"),
    random_state=params.get("models").get("SEED"),
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

num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
cat_pipeline = Pipeline(steps=[("encoder", OneHotEncoder())])
preprocessor = ColumnTransformer(
    transformers=[("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)],
    remainder="drop",
)

mlflow.set_experiment("exp_1")

MODEL_NAME = "trees"
trees = ["tree", "tree_2", "tree_3"]
for tree in trees:
    with mlflow.start_run(run_name=f"{tree}") as run:
        tree = DecisionTreeRegressor(
            **params.get("models").get(tree),
            random_state=params.get("models").get("SEED"),
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", tree)])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        mlflow.log_params(params.get("models").get("tree"))
        mlflow.log_metric("mae", mae)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

client = MlflowClient()
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version="1",
)
print(f"Registered {MODEL_NAME} v1 as @champion (MAE: {mae:.4f})")
