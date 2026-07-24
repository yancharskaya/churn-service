import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import logging

logging.basicConfig(
    level=logging.INFO,  # показывать INFO и важнее
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

# --- 1. ЭТАЛОН: данные, на которых училась модель ---
# в проекте это ваш train.csv (тот, что под DVC):
reference = pd.read_csv("data/train.csv").drop(columns=["target"])

# --- 2. ТЕКУЩИЕ данные: что «приходит из прода» ---
# на реальном проекте — свежая выгрузка; для учёбы эмулируем,
# взяв копию эталона и СЛЕГКА испортив два признака:
current = reference.copy()
current["0"] *= 1.3  # сдвиг масштаба
current["1"] += np.random.normal(0, 5, len(current))

# --- 3. сравнить и сохранить отчёт ---
report = Report([DataDriftPreset()])
result = report.run(reference_data=reference, current_data=current)
result.save_html("reports/drift_report.html")
log.info("отчёт готов: reports/drift_report.html — открой в браузере")
