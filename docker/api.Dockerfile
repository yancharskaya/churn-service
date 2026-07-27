FROM python:3.14-slim AS builder
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install ".[api]"

FROM python:3.14-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/

# логи не буферизуются -> видны в kubectl logs
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
