"""Query operations for retrieving Act metadata and sections."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from barrister_ai.citation_validation.db.models import (
    ActMetadata,
    Section,
    create_session_factory,
)


class ActRepository:
    """Repository for querying legal acts and their statutory sections."""

    def __init__(self, session_factory: sessionmaker | None = None) -> None:
        self._session_factory = session_factory

    @property
    def session_factory(self) -> sessionmaker:
        """Lazily create a default session factory if none was injected."""
        if self._session_factory is None:
            self._session_factory = create_session_factory()
        return self._session_factory

    def get_sections_by_act_and_year(
        self,
        act_name: str,
        year: int | str,
        section: str | None = None,
        session: Session | None = None,
    ) -> dict[str, Any]:
        """Look up an Act by name/year and optionally filter by section number."""
        owns_session = session is None
        db_session = session or self.session_factory()

        try:
            act = (
                db_session.query(ActMetadata)
                .filter(
                    ActMetadata.act_name.ilike(f"%{act_name}%"),
                    ActMetadata.year == int(year),
                )
                .first()
            )

            if not act:
                return {
                    "Result": "Fail",
                    "error-massage": f"{act_name} not found in the corpus",
                    "act-found": False,
                    "section-found": False,
                }

            condition = Section.act_id == act.act_id
            if section is not None:
                condition &= Section.section_number == str(section)

            sections = db_session.query(Section).filter(condition).all()

            if not sections:
                return {
                    "Result": "Fail",
                    "error-massage": f"Section {section} in {act_name} not found in the corpus",
                    "act-found": True,
                    "section-found": False,
                }

            act_sections = [
                {
                    "section_title": sec.section_title,
                    "sub-section": sec.subsection,
                    "section_number": sec.section_number,
                }
                for sec in sections
            ]

            return {
                "sections": act_sections,
                "Act_Name": f"{act_name}, {year}",
                "Year": str(year),
                "Result": "Pass",
                "act-found": True,
                "section-found": True,
            }
        finally:
            if owns_session:
                db_session.close()


def get_sections_by_act_and_year(
    act_name: str,
    year: int | str,
    section: str | None = None,
) -> dict[str, Any]:
    """Convenience function querying sections via the default database config."""
    return ActRepository().get_sections_by_act_and_year(
        act_name=act_name,
        year=year,
        section=section,
    )
