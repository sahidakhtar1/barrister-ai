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

### 5. Web UI (`ui/`)
- `app.py`: Static asset loader and HTTP server (`serve_ui`, `create_ui_server`) with `/health` check.
- `templates/index.html`: Legal query submission interface targeting `/api/query`.

### 6. CLI Entry Point (`main.py`)
Provides subcommands: `info`, `parse-citation`, `validate-citation`, `search`, and `serve-ui`.
