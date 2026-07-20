[![ci](https://github.com/yancharskaya/churn-service/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/yancharskaya/churn-service/actions/workflows/ci.yml)

# churn-service

Учебный MLOps-проект: sklearn-модель за FastAPI — с Docker,
CI, реестром моделей (MLflow) и версионированием данных (DVC).

## Стек
Python 3.14 · FastAPI · scikit-learn · MLflow · DVC · MinIO ·
Docker/Compose · GitHub Actions · Kubernetes · Airflow · Evidently

## Быстрый старт (локальная разработка)
```
docker compose up -d      # запустить MLflow (http://localhost:5000) и MinIO (http://localhost:9001). После первого запуска создать bucket'ы mlflow и data — см. консоль MinIO

make install              # создать .venv и установить зависимости из requirements-dev.txt
make train                # обучить модель (использует переменные из train/.env)
make start-api            # запустить API: http://localhost:8000/docs (использует переменные из .env)
make data-generate        # создать датасет. При желании задать количество строк 0<=COUNT=<569 : make data-generate COUNT=100 

source .venv/bin/activate # временно поменять переменную окружения PATH, брать все команды из venv
```
## Конфигурация dvc (после make install):
```
dvc remote modify --local storage endpointurl <url>
dvc remote modify --local storage access_key_id <name>
dvc remote modify --local storage secret_access_key <password>
```

## Использование dvc
```
dvc add data/train.csv                # файл → под контроль DVC
git add data/train.csv.dvc data/.gitignore
git commit -m 'data: v1'              # Git запомнил УКАЗАТЕЛЬ
dvc push                              # сами данные → в MinIO

# --- вышла «версия 2» данных (добавились строки) ---
dvc add data/train.csv                # пересчитал отпечаток, обновил .dvc
git commit -am 'data: v2' && dvc push

# --- откатиться к v1 (машина времени) ---
git checkout HEAD~1 -- data/train.csv.dvc   # старый указатель
dvc checkout                                 # DVC подменил сам файл

# --- у коллеги / в CI с чистого клона ---
git clone ... && dvc pull             # git даёт указатели, dvc — данные
```

## Тесты и стиль
```
pytest -v
ruff check src/
```

## CI
Для того, чтобы сделать merge в main, должны быть успешно выполнены jobs: lint, test, build.
Добавлена возможность собирать образ в build, находясь в любой ветке, но заливать его в реестр только из main.

## Прогресс
- [x] Multi-stage Docker-образ + /health
- [x] Compose: api + MLflow + MinIO
- [x] MLflow: трекинг и реестр (модель из реестра)
- [x] DVC: данные версионируются в MinIO
- [x] CI: lint → test → build (GitHub Actions)
- [ ] Kubernetes: Deployment + Service
- [ ] Airflow: DAG переобучения
- [ ] Evidently: drift-отчёт
