import pytest
from fastapi.testclient import TestClient   # «виртуальный браузер»:
from src.main import app                    #   дергает API без запуска
                                            #   uvicorn и без сети
client = TestClient(app)

def test_health():
    assert client.get('/health').status_code == 200

def test_predict_valid():
    r = client.post('/predict', json={'values': [1.0] * 30})
    assert r.status_code == 200
    assert r.json()['prediction'] in (0, 1)   # форма и диапазон ответа

def test_predict_invalid_type():             # Pydantic должен отбить
    r = client.post('/predict', json={'values': 'тридцать чисел'})
    assert r.status_code == 422

@pytest.mark.parametrize('bad', [            # один тест — три случая:
    {},                                      #   пустое тело
    {'values': []},                          #   пустой список
    {'wrong_key': [1.0] * 30},               #   не то поле
])
def test_predict_bad_inputs(bad):
    assert client.post('/predict', json=bad).status_code == 422
