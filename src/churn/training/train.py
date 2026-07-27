"""Обучение модели. Куда сохранить — решает model_store."""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn.common.config import settings
from churn.common.logging import get_logger, setup_logging
from churn.common.model_store import save_model

log = get_logger(__name__)


def build_pipeline(c: float = 1.0) -> Pipeline:
    """Отдельная функция — чтобы её можно было проверить unit-тестом."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=c, max_iter=1000)),
        ]
    )


def main(c: float = 1.0) -> None:
    setup_logging()

    if settings.data_path:
        df = pd.read_csv(settings.data_path)
        X, y = df.drop("target", axis=1), df["target"]
    else:
        X, y = load_breast_cancer(return_X_y=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    pipe = build_pipeline(c)
    pipe.fit(X_tr, y_tr)

    f1 = f1_score(y_te, pipe.predict(X_te))
    log.info("f1=%.4f", f1)

    # store сам решит: реестр MLflow или локальный model.pkl
    save_model(
        pipe,
        params={"model": "logreg", "C": c, "rows": len(df)},
        metrics={"f1": f1},
    )


if __name__ == "__main__":  # запуск: python -m churn.training.train
    main()
