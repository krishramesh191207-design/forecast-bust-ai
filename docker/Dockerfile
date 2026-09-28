FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libeccodes0 libeccodes-dev libnetcdf-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY Makefile ./

ENV PYTHONPATH=/app/src FBEWS_ROOT=/app FBEWS_DATA_MODE=sandbox
EXPOSE 8008

# Generate sandbox artifacts during the image build so runtime stays within the free plan memory limit.
RUN python scripts/cli.py all

CMD ["sh", "-c", "uvicorn fbews.api.main:app --host 0.0.0.0 --port ${PORT:-8008}"]
