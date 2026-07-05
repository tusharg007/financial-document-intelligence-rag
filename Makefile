.PHONY: install test lint format ingest-demo index evaluate run-api run-app docker-up

install:
	pip install -r requirements.txt

test:
	python -m pytest

lint:
	python -m compileall api.py app.py src scripts tests

format:
	python -m compileall api.py app.py src scripts tests

ingest-demo:
	python scripts/prepare_datasets.py --demo-ok

index:
	python -m src.indexing.build_indexes

evaluate:
	python scripts/evaluate.py --eval-set demo --retriever hybrid_rerank --llm extractive --output reports/evaluation_latest.json

run-api:
	uvicorn api:app --reload

run-app:
	streamlit run app.py

docker-up:
	docker compose up --build
