"""Unified CLI entry point for Barrister_AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from barrister_ai.citation_validation import extract_citation_details, validate_citation
from barrister_ai.config import get_settings
from barrister_ai.retrieval import LegalPrecedentSearchEngine
from barrister_ai.ui import serve_ui


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="barrister-ai",
        description="Barrister_AI legal precedent search and citation validation CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("info", help="Display project configuration and paths.")

    parse_cmd = subparsers.add_parser("parse-citation", help="Parse a legal citation.")
    parse_cmd.add_argument("citation", help="Citation text to deconstruct.")

    val_cmd = subparsers.add_parser(
        "validate-citation",
        help="Validate a legal citation against PostgreSQL and LLM.",
    )
    val_cmd.add_argument("citation", help="Citation text to validate.")

    search_cmd = subparsers.add_parser(
        "search", help="Search legal precedents using TF-IDF."
    )
    search_cmd.add_argument("query", help="Natural-language legal query.")
    search_cmd.add_argument("--top-k", type=int, default=5, help="Results to return.")
    search_cmd.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional override path to legal_cases.json.",
    )

    ui_cmd = subparsers.add_parser("serve-ui", help="Serve the Barrister_AI Web UI.")
    ui_cmd.add_argument("--host", default="127.0.0.1", help="Bind address.")
    ui_cmd.add_argument("--port", type=int, default=8080, help="Port number.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the Barrister_AI CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()

    if args.command == "info":
        print(f"{settings.project_name} v{settings.version}")
        print(f"Root           : {settings.root_dir}")
        print(f"Knowledge Base : {settings.knowledge_base_dir}")
        print(f"Acts Data      : {settings.acts_data_dir}")
        print(f"Reports        : {settings.reports_dir}")
        return 0

    if args.command == "parse-citation":
        print(json.dumps(extract_citation_details(args.citation), indent=2))
        return 0

    if args.command == "validate-citation":
        print(json.dumps(validate_citation(args.citation), indent=2, default=str))
        return 0

    if args.command == "search":
        dataset_path = args.dataset or (
            settings.knowledge_base_dir / "generated" / "legal_cases.json"
        )
        engine = LegalPrecedentSearchEngine(dataset_path=dataset_path)
        engine.initialize()
        results = engine.search(args.query, top_k=args.top_k)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    if args.command == "serve-ui":
        serve_ui(host=args.host, port=args.port)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
