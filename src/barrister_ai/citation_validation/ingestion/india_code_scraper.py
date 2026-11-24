"""Scraper and PDF downloader for India Code DSpace API."""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any

import pandas as pd
import requests

DEFAULT_COLLECTION_SCOPE = "69a0c1fb-7b22-4481-b16a-1dc59b5d02e6"
DEFAULT_API_BASE = "https://indiacode.gov.in/server/api"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def fetch_acts_metadata(
    year_range: str = "[2020 TO 2026]",
    scope: str = DEFAULT_COLLECTION_SCOPE,
    size: int = 100,
    page: int = 0,
    output_csv: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fetch Acts metadata from India Code DSpace search API."""
    api_url = f"{DEFAULT_API_BASE}/discover/search/objects"
    query_params = {
        "f.dateIssued": f"{year_range},equals",
        "scope": scope,
        "size": size,
        "page": page,
    }

    response = requests.get(
        api_url, params=query_params, headers=DEFAULT_HEADERS, timeout=30
    )
    response.raise_for_status()
    data = response.json()

    objects = (
        data.get("_embedded", {})
        .get("searchResult", {})
        .get("_embedded", {})
        .get("objects", [])
    )

    acts_list: list[dict[str, Any]] = []
    for obj in objects:
        item = obj.get("_embedded", {}).get("indexableObject", {})
        metadata = item.get("metadata", {})
        acts_list.append(
            {
                "act_id": item.get("uuid"),
                "handle": item.get("handle"),
                "title": (
                    metadata.get("dc.title", [{}])[0].get("value", "N/A")
                    if metadata.get("dc.title")
                    else "N/A"
                ),
                "date_issued": (
                    metadata.get("dc.date.issued", [{}])[0].get("value", "N/A")
                    if metadata.get("dc.date.issued")
                    else "N/A"
                ),
                "act_no": (
                    metadata.get("dc.title.alternative", [{}])[0].get("value", "N/A")
                    if metadata.get("dc.title.alternative")
                    else "N/A"
                ),
            }
        )

    if output_csv:
        pd.DataFrame(acts_list).to_csv(output_csv, index=False)

    return acts_list


def fetch_act_sections(
    acts_list: list[dict[str, Any]],
    output_csv: str | Path | None = None,
    delay_seconds: float = 0.2,
) -> list[dict[str, Any]]:
    """Fetch inline section metadata for each Act item from India Code."""
    sections_data: list[dict[str, Any]] = []

    for act in acts_list:
        act_id = act["act_id"]
        act_title = act["title"]
        detail_url = f"{DEFAULT_API_BASE}/core/items/{act_id}"
        det_resp = requests.get(detail_url, headers=DEFAULT_HEADERS, timeout=15)

        if det_resp.status_code == 200:
            metadata = det_resp.json().get("metadata", {})
            section_titles = metadata.get("indiacode.section.title", [])
            section_numbers = metadata.get("indiacode.section.number", [])
            section_details = metadata.get("indiacode.section.description", [])

            if section_titles:
                for i, sec_title_entry in enumerate(section_titles):
                    sections_data.append(
                        {
                            "act_id": act_id,
                            "act_title": act_title,
                            "section_number": (
                                section_numbers[i].get("value", "")
                                if i < len(section_numbers)
                                else ""
                            ),
                            "section_title": sec_title_entry.get("value", ""),
                            "section_description": (
                                section_details[i].get("value", "")
                                if i < len(section_details)
                                else ""
                            ),
                        }
                    )
            else:
                sections_data.append(
                    {
                        "act_id": act_id,
                        "act_title": act_title,
                        "section_number": "N/A",
                        "section_title": "Refer to full PDF attachment",
                        "section_description": "No inline metadata sections found",
                    }
                )

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    if output_csv:
        pd.DataFrame(sections_data).to_csv(output_csv, index=False)

    return sections_data


def download_act_pdfs(
    acts_list: list[dict[str, Any]],
    output_dir: str | Path,
    delay_seconds: float = 0.3,
) -> list[Path]:
    """Download Act PDF bitstreams from India Code into *output_dir*."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for act in acts_list:
        act_id = act["act_id"]
        act_title = act["title"]
        safe_title = re.sub(r'[\\/*?:"<>|]', "_", act_title)
        file_path = target_dir / f"{safe_title}.pdf"

        if file_path.exists():
            downloaded.append(file_path)
            continue

        bundles_url = f"{DEFAULT_API_BASE}/core/items/{act_id}/bundles"
        response = requests.get(bundles_url, headers=DEFAULT_HEADERS, timeout=15)
        if response.status_code != 200:
            continue

        pdf_download_url = None
        for bundle in response.json().get("_embedded", {}).get("bundles", []):
            if bundle.get("name") != "ORIGINAL":
                continue
            bitstreams_link = (
                bundle.get("_links", {}).get("bitstreams", {}).get("href")
            )
            if not bitstreams_link:
                continue
            bits_resp = requests.get(bitstreams_link, headers=DEFAULT_HEADERS, timeout=15)
            if bits_resp.status_code != 200:
                continue
            for bitstream in bits_resp.json().get("_embedded", {}).get("bitstreams", []):
                mimetype = bitstream.get("format", {}).get("mimetype", "").lower()
                name = bitstream.get("name", "").lower()
                if "pdf" in mimetype or name.endswith(".pdf"):
                    pdf_download_url = (
                        bitstream.get("_links", {}).get("content", {}).get("href")
                    )
                    break
            if pdf_download_url:
                break

        if pdf_download_url:
            file_resp = requests.get(
                pdf_download_url, headers=DEFAULT_HEADERS, stream=True, timeout=30
            )
            if file_resp.status_code == 200:
                with file_path.open("wb") as handle:
                    for chunk in file_resp.iter_content(chunk_size=8192):
                        handle.write(chunk)
                downloaded.append(file_path)

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    return downloaded


def extract_pdf_previews(
    pdf_folder: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Extract page counts and first-page previews from downloaded Act PDFs."""
    from pypdf import PdfReader

    folder = Path(pdf_folder)
    pdf_files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".pdf")
    extracted_data: list[dict[str, Any]] = []

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            num_pages = len(reader.pages)
            first_page = reader.pages[0].extract_text() or "" if num_pages > 0 else ""
            preview = first_page[:300].replace("\n", " ").strip() + "..."
            extracted_data.append(
                {
                    "filename": pdf_path.name,
                    "total_pages": num_pages,
                    "first_page_preview": preview,
                }
            )
        except Exception as exc:
            extracted_data.append(
                {
                    "filename": pdf_path.name,
                    "total_pages": 0,
                    "first_page_preview": f"Error reading file: {exc}",
                }
            )

    df_pdf_info = pd.DataFrame(extracted_data)
    if output_csv:
        df_pdf_info.to_csv(output_csv, index=False)
    return df_pdf_info
