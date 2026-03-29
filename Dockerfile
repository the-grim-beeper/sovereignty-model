FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY model/ model/
COPY app/ app/

RUN pip install --no-cache-dir .

CMD ["sh", "-c", "uvicorn app.dashboard:app --host 0.0.0.0 --port ${PORT:-8000}"]
