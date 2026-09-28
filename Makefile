PY ?= python3
export PYTHONPATH := src

.PHONY: install sandbox download probe features labels train evaluate infer run-api run-demo test lint all clean

install:
	$(PY) -m pip install -r requirements.txt

sandbox:            ## generate the clearly-labelled synthetic sandbox dataset
	$(PY) scripts/cli.py sandbox

download:           ## attempt real-data ingestion (needs network + credentials)
	$(PY) scripts/cli.py download

probe:              ## report which real data sources are reachable from here
	$(PY) scripts/cli.py probe

boundaries:
	$(PY) scripts/prepare_boundaries.py

features:
	$(PY) scripts/cli.py features

labels:
	$(PY) scripts/cli.py labels

train:
	$(PY) scripts/cli.py train

evaluate:
	$(PY) scripts/cli.py evaluate

infer:
	$(PY) scripts/cli.py infer

run-api:            ## start the API and dashboard on http://localhost:8008
	$(PY) -m uvicorn fbews.api.main:app --host 0.0.0.0 --port 8008

run-demo: sandbox boundaries features labels train evaluate run-api

test:
	$(PY) -m pytest tests -q

all: sandbox boundaries features labels train evaluate

clean:
	rm -rf data/processed data/features data/labels data/products artifacts/models
