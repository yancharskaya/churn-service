"""Конфигурация приложения. Читается из переменных окружения и .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",  # локально читаем .env, в кластере его нет
        env_file_encoding="utf-8",
        extra="ignore",  # лишние переменные окружения не ломают старт
    )

    # --- MLflow ---
    # ПУСТАЯ строка = локальный режим (без MLflow), см. model_store.py
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "churn"
    model_name: str = "churn-model"
    model_alias: str = "champion"

    # --- S3/MinIO (нужны и обучению, и сервису для чтения артефактов) ---
    mlflow_s3_endpoint_url: str = "http://localhost:9000"
    aws_access_key_id: str = "admin"
    aws_secret_access_key: str = "admin123"

    # --- данные и отчёты ---
    model_path: str = "model.pkl"  # локальный файл модели
    data_path: str = "data/train.csv"
    reports_dir: str = "reports"

    # --- прочее ---
    log_level: str = "INFO"


settings = Settings()  # один экземпляр на процесс, импортируется везде
