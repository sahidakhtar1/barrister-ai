# Barrister-AI Architecture

## Overview

Barrister-AI uses a `src/` package layout (`src/barrister_ai`) separating production application modules from runnable pipeline scripts (`scripts/`), datasets (`data/`), experiments (`notebooks/`), and tests (`tests/`).

## Core Modules (`src/barrister_ai/`)

### 1. Configuration (`config.py`)
Centralizes directory paths (`ROOT_DIR`, `DATA_DIR`, `KNOWLEDGE_BASE`, `ACTS_DATA_DIR`, `PRECEDENTS_DIR`, `REPORTS_DIR`) and environment-driven configuration dataclasses (`DatabaseConfig`, `LLMConfig`, `QdrantConfig`, `Settings`).

### 2. Citation Validation (`citation_validation/`)
Broken into small, testable submodules with dependency injection:
- `parser.py`: Extracts `Act_Name`, `Year`, `Section`, `Sub-section`, and `Proposition` via regex (`parse_citation`, `extract_citation_details`).
- `db/`:
  - `models.py`: SQLAlchemy `ActMetadata` and `Section` models plus lazy engine/session helpers (`init_db`, `create_session_factory`).
  - `repository.py`: `ActRepository` and `get_sections_by_act_and_year` for corpus verification.
  - `schema.sql`: Canonical PostgreSQL `legal` schema DDL.
- `llm_client.py`: `CitationLLMClient` and `validate_citation_locally` calling an OpenAI-compatible inference endpoint grounded strictly on retrieved Act context.
- `validator.py`: `CitationValidator` orchestrating parsing -> DB lookup -> LLM verification.
- `vector_store/qdrant_store.py`: `ActVectorStore` for Qdrant indexing and metadata-filtered semantic search over Act sections.
- `ingestion/`:
  - `postgres_loader.py`: CSV-to-PostgreSQL ingestion (`ingest_csv_to_postgres`).
  - `india_code_scraper.py`: India Code DSpace API scraper and Act PDF downloader.

### 3. Precedent Retrieval (`retrieval/`)
- `tfidf_search.py`: `LegalPrecedentSearchEngine` building TF-IDF vectors over legal case fields and ranking candidates by cosine similarity.

### 4. Query Classification (`classification/`)
- `case_type.py`: `CaseTypeClassifier` encoding legal queries with SentenceTransformers (`all-MiniLM-L6-v2`) and classifying case domains using `LogisticRegression` or incremental `SGDClassifier`.

### 5. REST API (`api/`)
- `app.py`: FastAPI REST API (`create_api_app`, `app`), Uvicorn runner (`serve_api`), and query processor (`process_query`) with `GET /health` and `POST /baristerai/query`.

### 6. Web UI (`ui/`)
- `app.py`: FastAPI Web UI server (`create_app`, `app`, `serve_ui`) and template loader (`load_home_html`) serving `index.html` and `/health`.
- `templates/index.html`: Legal query submission interface calling `${API_BASE_URL}/baristerai/query`.

### 7. CLI Entry Point (`main.py`)
Provides subcommands: `info`, `parse-citation`, `validate-citation`, `search`, `serve-api`, and `serve-ui`.
