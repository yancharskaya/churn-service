import mlflow, mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import os
import pandas as pd

host = os.getenv("HOST", 'http://localhost:5000')
churn = os.getenv("CHURN", 'churn')

mlflow.set_tracking_uri(host)                      # сервер из compose
mlflow.set_experiment(churn)                       # папка для run'ов

data_path = os.getenv("DATA")                      # DATA="data/train.csv"
if data_path:
    df = pd.read_csv(data_path)
    X, y = df.drop('target', axis=1), df['target']
else:
    X, y = load_breast_cancer(return_X_y=True)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)

C = float(os.getenv("C") or 1.0)                  # меняйте между запусками
with mlflow.start_run():                          # начало «записи»
    mlflow.log_param('model', 'logreg')           # записать параметр
    mlflow.log_param('C', C)
    pipe = Pipeline([('sc', StandardScaler()),
                     ('lr', LogisticRegression(C=C, max_iter=1000))])
    pipe.fit(X_tr, y_tr)
    f1 = f1_score(y_te, pipe.predict(X_te))
    mlflow.log_metric('f1', f1)                   # записать метрику
    mlflow.sklearn.log_model(                     # записать саму модель
        pipe, name='model',                       #   (артефакт → в MinIO!)
        registered_model_name='churn-model')      #   и в реестр
    print(f'Model trained successfully on dataset with {len(X)} rows using C={C}')
    print('f1:', f1)
