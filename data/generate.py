import pandas as pd
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
pd.DataFrame(X).assign(target=y).to_csv('data/train.csv', index=False)