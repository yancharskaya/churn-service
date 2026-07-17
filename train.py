# train.py — обучение и сохранение модели (запуск: python train.py)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
pipe = Pipeline([('sc', StandardScaler()),
                 ('lr', LogisticRegression(max_iter=1000))])
pipe.fit(X_tr, y_tr)
print('f1:', f1_score(y_te, pipe.predict(X_te)))
joblib.dump(pipe, 'model.pkl')          # с дня 3: mlflow.sklearn.log_model
