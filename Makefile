.PHONY: install collect preprocess represent cluster extract-features topic-model sentiment classify validate test all

all: install collect preprocess represent cluster extract-features topic-model sentiment classify validate

install:
	uv sync

collect:
	uv run python -m src.collection.collector

preprocess:
	uv run python -m src.preprocessing.cleaner

represent:
	uv run python -m src.representation.vectorizer

cluster:
	uv run python -m src.clustering.clusterer

extract-features:
	uv run python -m src.feature_engineering.scorers

topic-model:
	uv run python -m src.topic_modeling.topic_modeler

sentiment:
	uv run python -m src.sentiment.analyzer

classify:
	uv run python -m src.classification.classifier

validate:
	uv run python -m src.validation.validator

test:
	uv run pytest tests/
