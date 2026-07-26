ARG MLFLOW_VERSION=3.14.0
FROM ghcr.io/mlflow/mlflow:v${MLFLOW_VERSION}
RUN pip install --no-cache-dir boto3==1.35.99
