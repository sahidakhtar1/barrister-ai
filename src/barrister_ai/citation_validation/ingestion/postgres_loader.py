"""CSV-to-PostgreSQL loader for legislative Acts and sections."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from barrister_ai.citation_validation.db.models import (
    ActMetadata,
    Section,
    create_session_factory,
    init_db,
)
from barrister_ai.config import ACTS_DATA_DIR


def ingest_csv_to_postgres(
    csv_file_path: str | Path,
    session: Session | None = None,
) -> int:
    """Ingest Act and section records from *csv_file_path* into PostgreSQL.

    Returns the number of processed rows.
    """
    path = Path(csv_file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path}")

    df = pd.read_csv(path)
    owns_session = session is None
    if owns_session:
        init_db()
        db_session = create_session_factory()()
    else:
        db_session = session

    try:
        for _, row in df.iterrows():
            act_name = row.get("Act_Name")
            date_issued_str = row.get("Date_Issued")

            date_issued = None
            year = None
            if pd.notna(date_issued_str):
                try:
                    date_issued = datetime.strptime(
                        str(date_issued_str).strip(), "%Y-%m-%d"
                    ).date()
                    year = date_issued.year
                except ValueError:
                    pass

            existing_act = (
                db_session.query(ActMetadata)
                .filter_by(act_name=act_name, date_issued=date_issued)
                .first()
            )

            if existing_act:
                act_id = existing_act.act_id
            else:
                new_act = ActMetadata(
                    act_name=act_name,
                    act_alias=None,
                    year=year,
                    date_issued=date_issued,
                    repeal_date=None,
                    status=None,
                )
                db_session.add(new_act)
                db_session.flush()
                act_id = new_act.act_id

            sec_number = str(int(row.get("Section_Number")))
            sec_title = row.get("Section_Title")
            subsection = row.get("Section_Content")

            existing_section = (
                db_session.query(Section)
                .filter_by(act_id=act_id, section_number=sec_number)
                .first()
            )
            if existing_section:
                existing_section.section_title = sec_title
                existing_section.subsection = subsection
            else:
                db_session.add(
                    Section(
                        act_id=act_id,
                        section_number=sec_number,
                        section_title=str(sec_title) if pd.notna(sec_title) else None,
                        subsection=subsection,
                    )
                )

        db_session.commit()
        return len(df)
    except Exception:
        db_session.rollback()
        raise
    finally:
        if owns_session:
            db_session.close()


if __name__ == "__main__":
    default_csv = ACTS_DATA_DIR / "acts_2020_2026.csv"
    count = ingest_csv_to_postgres(default_csv)
    print(f"Successfully ingested {count} rows into PostgreSQL database.")
