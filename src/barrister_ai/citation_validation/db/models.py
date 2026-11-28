"""SQLAlchemy ORM models and database session management."""

from __future__ import annotations

from sqlalchemy import Column, Date, ForeignKey, Integer, String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from barrister_ai.config import DatabaseConfig

Base = declarative_base()


class ActMetadata(Base):
    """Metadata record for a legislative Act."""

    __tablename__ = "act_metadata"

    act_id = Column(Integer, primary_key=True, autoincrement=True)
    act_name = Column(String, nullable=False)
    act_alias = Column(String, nullable=True)
    year = Column(Integer, nullable=True)
    date_issued = Column(Date, nullable=True)
    repeal_date = Column(Date, nullable=True)
    status = Column(String, nullable=True)

    sections = relationship(
        "Section",
        back_populates="act",
        cascade="all, delete-orphan",
    )


class Section(Base):
    """Individual section or subsection belonging to an Act."""

    __tablename__ = "sections"

    section_id = Column(Integer, primary_key=True, autoincrement=True)
    act_id = Column(Integer, ForeignKey("act_metadata.act_id"), nullable=False)
    section_number = Column(String, nullable=True)
    section_title = Column(String, nullable=True)
    subsection = Column(String, nullable=True)

    act = relationship("ActMetadata", back_populates="sections")


def get_engine(config: DatabaseConfig | None = None) -> Engine:
    """Create a SQLAlchemy engine from configuration."""
    db_config = config or DatabaseConfig()
    return create_engine(db_config.url)


def create_session_factory(config: DatabaseConfig | None = None) -> sessionmaker:
    """Create a configured SQLAlchemy sessionmaker."""
    engine = get_engine(config)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(config: DatabaseConfig | None = None) -> Engine:
    """Initialize all database tables for the configured PostgreSQL database."""
    engine = get_engine(config)
    Base.metadata.create_all(bind=engine)
    return engine
