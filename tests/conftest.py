"""Общие фикстуры. pytest находит этот файл автоматически."""

import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from churn.training.train import build_pipeline


@pytest.fixture(scope="session")
def tiny_model():
    """Настоящая модель, но обученная на 100 строках — доли секунды.

    scope='session' => обучается ОДИН раз на весь прогон тестов.
    """
    X, y = load_breast_cancer(return_X_y=True)
    return build_pipeline().fit(X[:100], y[:100])


@pytest.fixture
def client(tiny_model):
    """TestClient, где загрузка модели из MLflow подменена фикстурой."""
    from churn.api.main import app, get_model

    app.dependency_overrides[get_model] = lambda: tiny_model
    yield TestClient(app)
    app.dependency_overrides.clear()  # убрать подмену после теста


@pytest.fixture
def sample_values() -> list[float]:
    """Валидный вход: 30 признаков, как у breast_cancer."""
    X, _ = load_breast_cancer(return_X_y=True)
    return X[0].tolist()
