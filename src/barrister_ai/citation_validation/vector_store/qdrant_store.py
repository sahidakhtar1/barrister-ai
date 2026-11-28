"""Qdrant-backed semantic search and indexing for legislative Acts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from barrister_ai.config import QdrantConfig


def generate_unique_id(act_name: str, date_issued: str, section_number: Any) -> int:
    """Generate a deterministic 64-bit integer ID from Act metadata fields."""
    combined_id = f"{act_name}_{date_issued}_{section_number}"
    return int(hashlib.md5(combined_id.encode("utf-8")).hexdigest()[:15], 16)


class ActVectorStore:
    """Wrapper around QdrantClient and SentenceTransformer for Act sections."""

    def __init__(
        self,
        config: QdrantConfig | None = None,
        client: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or QdrantConfig()
        self._client = client
        self._model = model

    @property
    def client(self) -> Any:
        """Lazily initialize the Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.config.url)
        return self._client

    @property
    def model(self) -> Any:
        """Lazily load the SentenceTransformer embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model

    def ensure_collection(self) -> None:
        """Create the target Qdrant collection if it does not already exist."""
        from qdrant_client.models import Distance, VectorParams

        if not self.client.collection_exists(self.config.collection_name):
            vector_size = self.model.get_embedding_dimension()
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def index_csv(self, csv_path: str | Path) -> int:
        """Index Act section records from a CSV file into Qdrant."""
        from qdrant_client.models import PointStruct

        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Acts CSV not found: {path}")

        df_acts = pd.read_csv(path)
        self.ensure_collection()

        indexed = 0
        for _, row in df_acts.iterrows():
            row_id = generate_unique_id(
                str(row["Act_Name"]),
                str(row["Date_Issued"]),
                row["Section_Number"],
            )
            text_for_embedding = (
                f"Act_Name: {row['Act_Name']}. "
                f"Date_Issued: {row['Date_Issued']}. "
                f"Section_Number: {row['Section_Number']}"
            )
            vector = self.model.encode(text_for_embedding).tolist()

            payload = row.to_dict()
            payload["Section_Number"] = str(int(row["Section_Number"]))

            point = PointStruct(id=row_id, vector=vector, payload=payload)
            self.client.upsert(collection_name=self.config.collection_name, points=[point])
            indexed += 1

        return indexed

    def search(
        self,
        query_text: str,
        date_prefix: str | None = None,
        section_number: str | None = None,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Query Qdrant with optional metadata filters on date and section."""
        from qdrant_client.models import FieldCondition, Filter, MatchText

        query_vector = self.model.encode(query_text).tolist()

        conditions = []
        if date_prefix:
            conditions.append(
                FieldCondition(key="Date_Issued", match=MatchText(text=date_prefix))
            )
        if section_number:
            conditions.append(
                FieldCondition(key="Section_Number", match=MatchText(text=section_number))
            )

        query_filter = Filter(must=conditions) if conditions else None
        points = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        ).points

        return [
            {
                "score": float(hit.score),
                "payload": dict(hit.payload or {}),
            }
            for hit in points
        ]
