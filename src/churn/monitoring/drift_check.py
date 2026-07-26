"""Проверка дрейфа данных: сравнивает эталон с текущей выборкой."""

from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from churn.common.config import settings
from churn.common.logging import get_logger, setup_logging

log = get_logger(__name__)


def make_current(reference: pd.DataFrame) -> pd.DataFrame:
    """Эмуляция «свежих данных из прода» со сдвигом двух признаков.

    На реальном проекте здесь была бы выгрузка за последний период.
    """
    current = reference.copy()
    if "mean radius" in current:
        current["mean radius"] *= 1.3
    if "mean texture" in current:
        rng = np.random.default_rng(42)
        current["mean texture"] += rng.normal(0, 5, len(current))
    return current


def main() -> None:
    setup_logging()
    reference = pd.read_csv(settings.data_path).drop(columns="target")
    current = make_current(reference)

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)

    out_dir = Path(settings.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "drift_report.html"
    result.save_html(str(out_file))
    log.info("drift report saved: %s", out_file)


if __name__ == "__main__":
    main()
