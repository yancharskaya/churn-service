import pytest

from churn.common import model_store


def test_local_mode_when_uri_empty(monkeypatch):
    monkeypatch.setattr(model_store.settings, "mlflow_tracking_uri", "")
    assert model_store.use_registry() is False


def test_registry_mode_when_uri_set(monkeypatch):
    monkeypatch.setattr(model_store.settings, "mlflow_tracking_uri", "http://mlflow:5000")
    assert model_store.use_registry() is True


def test_local_roundtrip(monkeypatch, tmp_path, tiny_model):
    """Сохранили локально -> загрузили -> модель работает."""
    monkeypatch.setattr(model_store.settings, "mlflow_tracking_uri", "")
    monkeypatch.setattr(model_store.settings, "model_path", str(tmp_path / "model.pkl"))

    model_store.save_model(tiny_model)
    loaded = model_store.load_model()
    assert hasattr(loaded, "predict")


def test_local_load_without_file_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(model_store.settings, "mlflow_tracking_uri", "")
    monkeypatch.setattr(model_store.settings, "model_path", str(tmp_path / "нет-такого.pkl"))
    with pytest.raises(FileNotFoundError, match="make train"):
        model_store.load_model()
