# Contributing to BeyondBots

Welcome to the BeyondBots project! This document outlines our repository workflow, coding standards, and team roles.

---

## Team Split and Areas of Responsibility

Our project is organized around seven distinct text mining pipeline stages. The team is divided as follows:

### 1. Infrastructure & Pipeline Foundation Team (Infra Team)
- **Responsibility**: Data acquisition, text cleaning, representation, and workflow scheduling.
- **Key Modules**: 
  - `src/collection/` (Reddit API & PRAW client)
  - `src/preprocessing/` (Indic tokenization, clean-vs-retained token pipelines, transliteration flags)
  - `src/representation/` (TF-IDF baselines & HuggingFace MuRIL embedding extraction)
- **Deliverables**: Pipeline serialization format (Parquet files in `data/processed/`), API connections.

### 2. Clustering & Coordinated Group Detection Team (Clustering Team)
- **Responsibility**: Grouping similar posts and identifying temporal or relational spikes.
- **Key Modules**:
  - `src/clustering/` (HDBSCAN for near-duplicates, rolling window Z-scores, Leiden community detection)
- **Deliverables**: Coordinated cluster tables linking posts to specific coordinated events.

### 3. Linguistic Scoring & Features Team (Features Team)
- **Responsibility**: Synthesizing textual representation outputs into numeric feature signals.
- **Key Modules**:
  - `src/feature_engineering/` (Marketing copy, Fan identity, Criticism absence, Specificity scores)
  - `src/sentiment/` (Multilingual XLM-RoBERTa sentiment prediction, cluster variance)
- **Deliverables**: The 4 standardized scorer classes returning floats in the `[0, 1]` range.

### 4. Classification & Interpretability Team (Classification Team)
- **Responsibility**: Final classification of coordinated clusters and explains how decisions were reached.
- **Key Modules**:
  - `src/classification/` (Few-shot Claude motive classifier, logistic regression SHAP explainer)
  - `src/topic_modeling/` (BERTopic cluster topic summaries)
- **Deliverables**: Categorized final Parquet table + SHAP feature importance charts.

### 5. Validation & Ground Truth Annotation Team (Annotation Team)
- **Responsibility**: Standardizing the labeling interface and validating pipeline predictions.
- **Key Modules**:
  - `annotation/` (Schema definition, CLI labeling utility)
  - `src/validation/` (Case-study cross-reference)
- **Deliverables**: Manually labeled validation dataset (`data/labeled/`) + classification report comparing pipeline predictions to ground truth.

---

## Coding Standards & Pull Request Guidelines
- **Modularity**: Every pipeline stage *must* exist in its own named module. Do not write generic "utils.py" or bundle multi-stage steps together.
- **Verification**: Write unit tests for each pipeline component in the `tests/` directory. All tests must pass before opening a PR.
- **Type Hints**: Annotate all functions with type hints.
- **Configuration**: Do not hardcode strings (film titles, thresholds, model names). Register them in `config.yaml` and load them via the configuration utilities.
