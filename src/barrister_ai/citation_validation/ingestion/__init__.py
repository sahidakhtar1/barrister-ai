"""Ingestion pipelines for loading statutory Acts into PostgreSQL and local storage."""

from barrister_ai.citation_validation.ingestion.india_code_scraper import (
    download_act_pdfs,
    extract_pdf_previews,
    fetch_act_sections,
    fetch_acts_metadata,
)
from barrister_ai.citation_validation.ingestion.postgres_loader import (
    ingest_csv_to_postgres,
)

__all__ = [
    "download_act_pdfs",
    "extract_pdf_previews",
    "fetch_act_sections",
    "fetch_acts_metadata",
    "ingest_csv_to_postgres",
]
