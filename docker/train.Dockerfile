FROM python:3.14-slim
WORKDIR /app

# dvc[s3] тянет git — он нужен внутри образа
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[train]"

# логи не буферизуются -> видны в kubectl logs
ENV PYTHONUNBUFFERED=1

# команду не задаём: её указывает вызывающий (Job, CronJob, Airflow)
#   python -m churn.training.train
#   python -m churn.monitoring.drift_check
