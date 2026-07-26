"""Быстрые тесты API: без сети, без MLflow, без обученной модели с диска."""

import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_ok(client, sample_values):
    r = client.post("/predict", json={"values": sample_values})
    assert r.status_code == 200
    assert r.json()["prediction"] in (0, 1)


@pytest.mark.parametrize(
    "bad_body",
    [
        {},  # пустое тело
        {"values": []},  # пустой список
        {"values": [1.0] * 5},  # не 30 признаков
        {"values": "тридцать чисел"},  # не тот тип
        {"wrong_key": [1.0] * 30},  # не то поле
    ],
)
def test_predict_rejects_bad_input(client, bad_body):
    """Pydantic обязан отбить любой кривой вход кодом 422."""
    assert client.post("/predict", json=bad_body).status_code == 422
