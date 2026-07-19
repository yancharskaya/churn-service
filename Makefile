install:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements-dev.txt

start-api:
	.venv/bin/uvicorn src.main:app --reload --env-file .env

clean:
	rm -rf .venv

.PHONY: train
train:
	.venv/bin/dotenv -f train/.env run -- .venv/bin/python train/train.py

data-generate:
	.venv/bin/python data/generate.py
