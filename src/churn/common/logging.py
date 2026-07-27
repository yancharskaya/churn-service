"""Единая настройка логирования для всех точек входа."""

import logging
import sys

from churn.common.config import settings


def setup_logging() -> None:
    """Вызывается ОДИН раз при старте процесса (API, обучение, drift)."""
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,  # в контейнере логи читают именно из stdout
        force=True,  # перебить настройки библиотек (uvicorn и др.)
    )
    # библиотеки слишком болтливы на DEBUG — приглушаем
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Логгер модуля: get_logger(__name__)."""
    return logging.getLogger(name)
