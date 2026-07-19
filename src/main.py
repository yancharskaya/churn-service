from fastapi import FastAPI
from pydantic import BaseModel, Field
import logging
import mlflow.sklearn
import os
import joblib

logging.basicConfig(
    level=logging.INFO,  # показывать INFO и важнее
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI()

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
    model = mlflow.sklearn.load_model("models:/churn-model@champion")
else:
    model = joblib.load("model.pkl")

log.info("model loaded")


class Features(BaseModel):  # breast_cancer имеет 30 признаков;
    values: list[float] = Field(
        ..., min_length=30, max_length=30
    )  # для простоты принимаем списком


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(f: Features):
    pred = model.predict([f.values])
    return {"prediction": int(pred[0])}
