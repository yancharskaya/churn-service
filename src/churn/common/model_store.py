"""Единая точка загрузки и сохранения модели.

Два режима, выбор по settings.mlflow_tracking_uri:
  задан  -> реестр MLflow (боевой путь, работает в compose и в кластере)
  пуст   -> локальный файл model.pkl (быстрая проверка без инфраструктуры)
"""

from pathlib import Path
from typing import Any

import joblib

from churn.common.config import settings
from churn.common.logging import get_logger

log = get_logger(__name__)


def use_registry() -> bool:
    """True, если работаем через MLflow. Одно место принятия решения."""
    return bool(settings.mlflow_tracking_uri.strip())


def save_model(model: Any, params: dict | None = None, metrics: dict | None = None) -> None:
    """Сохраняет модель в реестр MLflow или в локальный файл."""
    if not use_registry():
        path = Path(settings.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        log.warning(
            "ЛОКАЛЬНЫЙ режим: модель сохранена в %s, история экспериментов НЕ ведётся", path
        )
        return

    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run():
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model", registered_model_name=settings.model_name)
    log.info("модель зарегистрирована в MLflow как %s", settings.model_name)


def load_model() -> Any:
    """Загружает модель из реестра MLflow или из локального файла."""
    if not use_registry():
        path = Path(settings.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Нет файла модели {path}. Сначала обучите: make train "
                f"(или задайте MLFLOW_TRACKING_URI для работы с реестром)"
            )
        log.warning("ЛОКАЛЬНЫЙ режим: модель загружена из %s", path)
        return joblib.load(path)

    import mlflow.sklearn

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    uri = f"models:/{settings.model_name}@{settings.model_alias}"
    log.info("загрузка модели из реестра: %s", uri)
    return mlflow.sklearn.load_model(uri)
