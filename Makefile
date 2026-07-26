VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: help install train data-generate start-api test test-integration \
        lint format drift clean

help:               ## список целей
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

$(VENV):            ## создать окружение, если его ещё нет
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)    ## окружение + все зависимости (для разработки)
	$(PIP) install -e ".[api,train,dev]"
	$(VENV)/bin/pre-commit install --hook-type pre-commit --hook-type pre-push

data-generate:      ## создать датасет: make data-generate COUNT=100
	$(PY) -m churn.training.data $(if $(COUNT),--count $(COUNT),)

train:              ## обучить модель и залогировать в MLflow
	$(PY) -m churn.training.train

start-api-dev:          ## запустить сервис локально с автоперезагрузкой
	$(VENV)/bin/uvicorn churn.api.main:app --reload --port 8000

start-api:          ## запустить сервис локально с автоперезагрузкой
	$(VENV)/bin/uvicorn churn.api.main:app --port 8000

drift:              ## проверить дрейф и сохранить отчёт в reports/
	$(PY) -m churn.monitoring.drift_check

test:               ## быстрые unit-тесты (секунды, без внешних сервисов)
	$(VENV)/bin/pytest tests/unit -q

test-integration:   ## интеграционные (нужен docker compose up -d)
	$(VENV)/bin/pytest tests/integration -m integration -v

lint:               ## проверить стиль
	$(VENV)/bin/ruff check src tests

format:             ## отформатировать и починить, что можно
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/ruff format src tests

clean:              ## удалить окружение и кэши
	rm -rf $(VENV) .pytest_cache .ruff_cache reports/*.html
	find . -type d -name __pycache__ -exec rm -rf {} +
