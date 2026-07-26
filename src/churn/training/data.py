import argparse

import pandas as pd
from sklearn.datasets import load_breast_cancer

from churn.common.config import settings
from churn.common.logging import get_logger, setup_logging

log = get_logger(__name__)


def main():
    setup_logging()
    # Инициализируем парсер
    parser = argparse.ArgumentParser(description="Скрипт генерации датасета")
    # Добавляем аргумент --count
    # type=int автоматически преобразует строку "100" в число 100
    # default задает значение по умолчанию,
    # если аргумент не передан (например, просто make data-generate)
    parser.add_argument("--count", type=int, default=569, help="Количество строк в датасете")

    # Парсим аргументы
    args = parser.parse_args()

    X, y = load_breast_cancer(return_X_y=True)
    # Теперь доступ к count можно получить через args.count
    count = args.count or 569
    count = max(1, min(count, 569))

    df = pd.DataFrame(X).assign(target=y)[:count]

    df.to_csv(settings.data_path, index=False)
    log.info(f"Data with {len(df)} rows saved successfully to {settings.data_path}")


if __name__ == "__main__":  # запуск: python -m churn.training.data
    main()
