# Barrister-AI

Barrister-AI is a modular AI legal engineering platform providing:
1. **Citation Validation** (`barrister_ai.citation_validation`) — regex-based citation parsing, PostgreSQL Act/Section corpus lookup, Qdrant semantic Act search, India Code ingestion, and LLM-grounded verification.
2. **Legal Precedent Retrieval** (`barrister_ai.retrieval`) — TF-IDF and cosine-similarity search across legal cases.
3. **Query Classification** (`barrister_ai.classification`) — SBERT + linear classifier for legal case-type routing.
4. **Web UI** (`barrister_ai.ui`) — lightweight frontend query interface.

## Project Structure

```text
barrister-ai/
├── pyproject.toml
├── requirements.txt
├── src/
│   └── barrister_ai/
│       ├── config.py                     # Centralized paths & env-based settings
│       ├── main.py                       # Unified CLI entry point
│       ├── citation_validation/          # Citation validation engine
│       │   ├── parser.py                 # Regex citation extraction
│       │   ├── validator.py              # Validation orchestrator
│       │   ├── llm_client.py             # Local/OpenAI-compatible LLM client
│       │   ├── db/                       # SQLAlchemy models, repository, SQL DDL
│       │   ├── vector_store/             # Qdrant indexing & search
│       │   └── ingestion/                # PostgreSQL loader & India Code scraper
│       ├── retrieval/                    # TF-IDF precedent search
│       ├── classification/               # Case-type query classifier
│       └── ui/                           # Static HTML UI & server
├── scripts/
│   ├── setup/                            # Project & knowledge base setup scripts
│   └── pipelines/                        # Dataset, search, & hardening pipelines
├── data/
│   ├── knowledge_base/                   # Acts, courts, domains, templates, generated cases
│   ├── precedents/                       # Hardened pairwise precedent datasets
│   ├── exports/                          # Exported CSV datasets
│   ├── acts/                             # Statutory Act CSVs & downloaded PDFs
│   └── qdrant_storage/                   # Local Qdrant persistence
├── notebooks/                            # Jupyter & experimental Colab notebooks
├── tests/                                # Unit test suite
├── docs/                                 # Architecture & project documentation
├── reports/                              # Validation & evaluation reports
├── models/                               # Saved model checkpoints
└── logs/                                 # Runtime logs
```

## Quick Start

```bash
# Run CLI info
PYTHONPATH=src python -m barrister_ai.main info

# Parse a legal citation
PYTHONPATH=src python -m barrister_ai.main parse-citation "Section 138 of the Negotiable Instruments Act, 1881"

# Search legal precedents
PYTHONPATH=src python -m barrister_ai.main search "cheque dishonoured insufficient funds"

# Serve the Web UI
PYTHONPATH=src python -m barrister_ai.main serve-ui --port 8080

# Run unit tests
PYTHONPATH=src python -m unittest discover -s tests -v
```
