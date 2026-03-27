FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY model/ model/
COPY app/ app/

RUN pip install --no-cache-dir .

EXPOSE ${PORT:-8501}

CMD streamlit run app/dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true
