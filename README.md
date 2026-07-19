# churn-service

Учебный MLOps-проект: sklearn-модель за FastAPI — с Docker,
CI, реестром моделей (MLflow) и версионированием данных (DVC).

## Стек
Python 3.11 · FastAPI · scikit-learn · MLflow · DVC · MinIO ·
Docker/Compose · GitHub Actions · Kubernetes · Airflow · Evidently

## Быстрый старт (локальная разработка)
```
docker compose up -d    # запустить MLflow (http://localhost:5000) и MinIO (http://localhost:9001). После первого запуска создать bucket'ы mlflow и data — см. консоль MinIO

make install            # создать .venv и установить зависимости из requirements-dev.txt
make train              # обучить модель (использует переменные из train/.env)
make start-api          # запустить API: http://localhost:8000/docs (использует переменные из .env)
```

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
