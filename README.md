# churn-service

Учебный MLOps-проект: sklearn-модель за FastAPI — с Docker,
CI, реестром моделей (MLflow) и версионированием данных (DVC).

## Стек
Python 3.11 · FastAPI · scikit-learn · MLflow · DVC · MinIO ·
Docker/Compose · GitHub Actions · Kubernetes · Airflow · Evidently

## Быстрый старт (локальная разработка)
```
git clone https://github.com/<логин>/churn-service && cd churn-service
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python train.py                # обучить модель (model.pkl)
uvicorn src.main:app --reload  # сервис: http://localhost:8000/docs
```

## Полный стек (Docker Compose)
```
docker compose up -d
```
API: localhost:8000 · MLflow UI: localhost:5000 · MinIO: localhost:9001
(после первого запуска создать bucket'ы mlflow и data — см. консоль MinIO)

## Тесты и стиль
```
pytest -v
ruff check src/
```

## Прогресс
- [x] Multi-stage Docker-образ + /health
- [x] Compose: api + MLflow + MinIO
- [ ] MLflow: трекинг и реестр (модель из реестра)
- [ ] DVC: данные версионируются в MinIO
- [ ] CI: lint → test → build (GitHub Actions)
- [ ] Kubernetes: Deployment + Service
- [ ] Airflow: DAG переобучения
- [ ] Evidently: drift-отчёт
