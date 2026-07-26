from sklearn.datasets import load_breast_cancer

from churn.training.train import build_pipeline


def test_pipeline_has_scaler_and_model():
    pipe = build_pipeline()
    assert list(pipe.named_steps) == ["scaler", "model"]


def test_pipeline_learns_something():
    """Дымовой тест: модель обучается и даёт осмысленное качество."""
    X, y = load_breast_cancer(return_X_y=True)
    pipe = build_pipeline().fit(X[:200], y[:200])
    assert pipe.score(X[200:], y[200:]) > 0.8
