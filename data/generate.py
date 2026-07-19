import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

count = int(os.getenv("COUNT") or 569)
count = max(1, min(count, 569))

df = pd.DataFrame(X).assign(target=y)[:count]

df.to_csv('data/train.csv', index=False)
print(f'Data with {len(df)} rows saved successfully to data/train.csv')
