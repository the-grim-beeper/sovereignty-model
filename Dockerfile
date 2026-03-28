FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY model/ model/
COPY app/ app/

RUN pip install --no-cache-dir .

EXPOSE ${PORT:-8501}

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
