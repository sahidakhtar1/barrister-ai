"""SBERT + Linear Classifier for legal query case-type classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier


class CaseTypeClassifier:
    """Predicts legal case types (e.g., Consumer, Criminal, Civil) from queries."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        encoder: Any | None = None,
        classifier: Any | None = None,
        probability_threshold: float = 0.6,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self._encoder = encoder
        self.classifier = classifier or LogisticRegression(max_iter=1000)
        self.probability_threshold = probability_threshold

    @property
    def encoder(self) -> Any:
        """Lazily load the SentenceTransformer encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def fit(self, queries: Sequence[str], labels: Sequence[str]) -> "CaseTypeClassifier":
        """Fit the classifier on embedded training queries."""
        embeddings = self.encoder.encode(list(queries))
        self.classifier.fit(embeddings, list(labels))
        return self

    def partial_fit(
        self,
        queries: Sequence[str],
        labels: Sequence[str],
        all_classes: Sequence[str] | None = None,
    ) -> "CaseTypeClassifier":
        """Incrementally update an SGDClassifier with new labeled queries."""
        if not isinstance(self.classifier, SGDClassifier):
            known = list(getattr(self.classifier, "classes_", []))
            self.classifier = SGDClassifier(
                loss="log_loss",
                max_iter=1000,
                tol=1e-3,
                random_state=42,
            )
            combined = sorted(set(known) | set(labels) | set(all_classes or []))
        else:
            existing = list(getattr(self.classifier, "classes_", []))
            combined = sorted(set(existing) | set(labels) | set(all_classes or []))

        embeddings = self.encoder.encode(list(queries))
        self.classifier.partial_fit(embeddings, list(labels), classes=combined)
        return self

    def predict(self, query: str) -> dict[str, Any]:
        """Predict the case type and confidence probabilities for *query*."""
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty.")

        embedding = self.encoder.encode([cleaned])
        predicted = str(self.classifier.predict(embedding)[0])
        probs = self.classifier.predict_proba(embedding)[0]
        class_labels = [str(c) for c in self.classifier.classes_]

        prob_pairs = sorted(
            zip(class_labels, [float(p) for p in probs]),
            key=lambda item: item[1],
            reverse=True,
        )

        above_threshold = [
            (label, prob)
            for label, prob in prob_pairs
            if prob > self.probability_threshold
        ]

        if len(above_threshold) > 1:
            selected = dict(above_threshold[:3])
        elif len(above_threshold) == 1:
            selected = dict(above_threshold)
        else:
            selected = {prob_pairs[0][0]: prob_pairs[0][1]}

        return {
            "query": cleaned,
            "predicted_case_type": predicted,
            "probabilities": selected,
        }

    def save(self, classifier_path: str | Path, encoder_dir: str | Path | None = None) -> None:
        """Persist the trained classifier (and optionally the SBERT encoder)."""
        clf_path = Path(classifier_path)
        clf_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier, clf_path)

        if encoder_dir is not None:
            enc_path = Path(encoder_dir)
            enc_path.mkdir(parents=True, exist_ok=True)
            self.encoder.save(str(enc_path))

    @classmethod
    def load(
        cls,
        classifier_path: str | Path,
        encoder_path_or_name: str = "all-MiniLM-L6-v2",
        probability_threshold: float = 0.6,
    ) -> "CaseTypeClassifier":
        """Load a persisted classifier and encoder."""
        classifier = joblib.load(classifier_path)
        return cls(
            embedding_model_name=encoder_path_or_name,
            classifier=classifier,
            probability_threshold=probability_threshold,
        )
