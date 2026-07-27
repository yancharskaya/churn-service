"""FastAPI-сервис предсказания оттока."""

from functools import lru_cache

from fastapi import Depends, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from churn.api.schemas import PredictRequest, PredictResponse
from churn.common.logging import get_logger, setup_logging
from churn.common.model_store import load_model

setup_logging()  # ТОЧКА ВХОДА: настраиваем логи один раз
log = get_logger(__name__)

app = FastAPI(title="churn-service")


@lru_cache(maxsize=1)
def get_model():
    """Модель из реестра MLflow ИЛИ из локального файла — решает store."""
    return load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    """Готовность: модель действительно загружена (для readinessProbe)."""
    get_model()
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, model=Depends(get_model)) -> PredictResponse:
    pred = model.predict([request.values])[0]
    log.info("prediction=%s", pred)
    return PredictResponse(prediction=int(pred))


Instrumentator().instrument(app).expose(app)
