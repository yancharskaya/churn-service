# src/main.py — сам сервис (запуск: uvicorn src.main:app --port 8000)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()
model = joblib.load('model.pkl')        # с дня 3: load_model из реестра
log.info('model loaded')

class Features(BaseModel):              # breast_cancer имеет 30 признаков;
    values: list[float]                 #   для простоты принимаем списком

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict')
def predict(f: Features):
    pred = model.predict([f.values])
    return {'prediction': int(pred[0])}
