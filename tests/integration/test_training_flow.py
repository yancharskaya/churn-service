"""Интеграционные: требуют поднятого docker compose (MLflow + MinIO).

Запуск: make test-integration
Обычный `make test` их НЕ трогает — они помечены маркером.
"""

import mlflow
import pytest

from churn.common.config import settings

pytestmark = pytest.mark.integration  # маркер на ВЕСЬ файл


@pytest.fixture(scope="module")
def mlflow_client():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return mlflow.MlflowClient()


def test_mlflow_is_reachable(mlflow_client):
    """Проверяем, что стек вообще поднят, — иначе остальное бессмысленно."""
    mlflow_client.search_experiments()


def test_training_creates_run(mlflow_client):
    """Полный цикл: обучение -> run в MLflow -> модель в реестре."""
    from churn.training.train import main

    main(c=0.5)

    exp = mlflow_client.get_experiment_by_name(settings.mlflow_experiment_name)
    runs = mlflow_client.search_runs(
        [exp.experiment_id], max_results=1, order_by=["start_time DESC"]
    )
    assert runs, "ни одного run не появилось"
    assert "f1" in runs[0].data.metrics
    assert runs[0].data.params["C"] == "0.5"
