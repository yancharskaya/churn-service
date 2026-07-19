import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import os
import pandas as pd
import joblib
import logging

logging.basicConfig(
    level=logging.INFO,  # показывать INFO и важнее
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

data_path = os.getenv("DATA_PATH")
if data_path:
    df = pd.read_csv(data_path)
    X, y = df.drop("target", axis=1), df["target"]
else:
    X, y = load_breast_cancer(return_X_y=True)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)


def train_model_mlflow():
    churn = os.getenv("CHURN", "churn")
    C = float(os.getenv("C") or 1.0)
    mlflow.set_tracking_uri(mlflow_uri)  # сервер из compose
    mlflow.set_experiment(churn)  # папка для run'ов
    with mlflow.start_run():  # начало «записи»
        mlflow.log_param("model", "logreg")  # записать параметр
        mlflow.log_param("C", C)
        pipe = Pipeline(
            [("sc", StandardScaler()), ("lr", LogisticRegression(C=C, max_iter=1000))]
        )
        pipe.fit(X_tr, y_tr)
        f1 = f1_score(y_te, pipe.predict(X_te))
        mlflow.log_metric("f1", f1)  # записать метрику
        mlflow.sklearn.log_model(  # записать саму модель
            pipe,
            name="model",  #   (артефакт → в MinIO!)
            registered_model_name="churn-model",
        )  #   и в реестр
        log.info(
            f"Model trained successfully on dataset with {len(X)} rows using C={C}"
        )
        log.info(f"f1:, {f1}")


def train_model_local():
    pipe = Pipeline(
        [("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))]
    )
    pipe.fit(X_tr, y_tr)
    log.info(f"f1:, {f1_score(y_te, pipe.predict(X_te))}")
    joblib.dump(pipe, "model.pkl")


if mlflow_uri:
    train_model_mlflow()
else:
    train_model_local()
