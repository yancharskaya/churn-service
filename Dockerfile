# --- этап 1: сборка зависимостей ---
FROM python:3.14-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- этап 2: чистый рабочий образ ---
FROM python:3.14-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY model.pkl .
COPY src/ ./src
EXPOSE 8000
HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
