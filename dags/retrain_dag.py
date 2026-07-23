from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from datetime import datetime

PROJECT = '/home/malyus/projects/churn-service'   # корень репозитория
PYTHON  = f'{PROJECT}/.venv/bin/python'           # интерпретатор ПРОЕКТА

@dag(schedule='@weekly', start_date=datetime(2026, 7, 1), catchup=False)
def retrain_churn():

    # шаг 1: свежие данные — вызываем dvc проекта (раздел 7)
    # pull_data = BashOperator(
    #     task_id='pull_data',
    #     bash_command=f'cd {PROJECT} && {PROJECT}/.venv/bin/dvc pull')

    # шаг 2: обучение — тот же train.py, что запускали руками (раздел 6),
    #        он сам логирует run и модель в MLflow
    train = BashOperator(
        task_id='train',
        bash_command=f'cd {PROJECT} && {PYTHON} train/train.py',
        env={'MLFLOW_TRACKING_URI': 'http://localhost:5000',
             'MLFLOW_S3_ENDPOINT_URL': 'http://localhost:9000',
             'AWS_ACCESS_KEY_ID': 'admin',
             'AWS_SECRET_ACCESS_KEY': 'admin123'})

    promote = BashOperator(
        task_id='promote',
        bash_command=f'cd {PROJECT} && {PYTHON} promote/promote.py',
        env={'MLFLOW_TRACKING_URI': 'http://localhost:5000'})
        # тут же: если лучше champion — переставить alias (раздел 6.4)

    # pull_data >> train >> promote()         # порядок: данные -> обучение -> выкатка
    train >> promote
retrain_churn()
