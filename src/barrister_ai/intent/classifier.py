"""Classical-ML intent classifier for legal queries — inference only.

Three intents are supported:

* ``case_summarisation``  – the user wants a summary / overview of a case.
* ``citation_validation`` – the user wants a legal citation verified.
* ``precedent_search``    – the user wants prior case law / precedents found.

The classifier loads a pre-trained TF-IDF + Logistic Regression model from
``models/intent_classifier.joblib``.  Train / update the model by running::

    python scripts/pipelines/query_intent_pipeline/train_and_save_model.py

It falls back gracefully to ``precedent_search`` when confidence is below the
configured threshold.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent labels
# ---------------------------------------------------------------------------

INTENT_CASE_SUMMARISATION = "case_summarisation"
INTENT_CITATION_VALIDATION = "citation_validation"
INTENT_PRECEDENT_SEARCH = "precedent_search"

ALL_INTENTS = [
    INTENT_CASE_SUMMARISATION,
    INTENT_CITATION_VALIDATION,
    INTENT_PRECEDENT_SEARCH,
]

# ---------------------------------------------------------------------------
# Default model path  (project-root/models/intent_classifier.joblib)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # barrister-ai/
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "intent_classifier.joblib"

# ---------------------------------------------------------------------------
# Text pre-processing helpers (kept in sync with train_and_save_model.py)
# ---------------------------------------------------------------------------

_LEGAL_SYNONYMS: dict[str, str] = {
    r"\bsummarise\b": "summarize",
    r"\bvalidate\b": "verify validate",
    r"\bcheck\b": "verify check",
    r"\bconfirm\b": "verify confirm",
    r"\bprecedent\b": "precedent case law ruling",
    r"\bjudgment\b": "judgment ruling decision",
    r"\bjudgement\b": "judgment ruling decision",
}


def _preprocess(text: str) -> str:
    """Lowercase, normalise whitespace, and expand legal synonyms."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    for pattern, replacement in _LEGAL_SYNONYMS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class IntentClassifier:
    """Loads a pre-trained TF-IDF + Logistic Regression intent classifier.

    Parameters
    ----------
    model_path:
        Path to the ``*.joblib`` file produced by ``train_and_save_model.py``.
        Defaults to ``models/intent_classifier.joblib`` relative to the
        project root.
    probability_threshold:
        Override the threshold stored in the model file.  Pass ``None``
        (default) to use the value embedded at training time.
    fallback_intent:
        Override the fallback intent stored in the model file.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        probability_threshold: float | None = None,
        fallback_intent: str | None = None,
    ) -> None:
        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._load(path)

        # Allow caller to override stored hyper-parameters
        if probability_threshold is not None:
            self.probability_threshold = probability_threshold
        if fallback_intent is not None:
            self.fallback_intent = fallback_intent

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self, path: Path) -> None:
        """Load a persisted model from *path*."""
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{path}'.  "
                "Run train_and_save_model.py to generate it first:\n"
                "  python scripts/pipelines/query_intent_pipeline/train_and_save_model.py"
            )
        data = joblib.load(path)
        self._pipeline = data["pipeline"]
        self.probability_threshold = data.get("probability_threshold", 0.40)
        self.fallback_intent = data.get("fallback_intent", INTENT_PRECEDENT_SEARCH)
        self._fitted = True
        logger.info("IntentClassifier loaded from %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        """Load a persisted classifier from an explicit *path*."""
        return cls(model_path=path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, query: str) -> dict[str, Any]:
        """Classify *query* and return a structured intent result.

        Returns
        -------
        dict with keys:

        * ``query``           – original query string
        * ``intent``          – predicted intent label
        * ``confidence``      – probability of the predicted intent (0-1)
        * ``all_scores``      – dict mapping every intent to its probability
        * ``above_threshold`` – whether confidence met *probability_threshold*
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not self._fitted:
            raise RuntimeError("Classifier is not fitted yet.")

        processed = _preprocess(query)
        proba = self._pipeline.predict_proba([processed])[0]
        classes: list[str] = list(self._pipeline.classes_)

        all_scores = {
            cls: round(float(p), 4) for cls, p in zip(classes, proba)
        }
        best_intent = classes[int(np.argmax(proba))]
        best_conf = float(proba[np.argmax(proba)])

        above = best_conf >= self.probability_threshold
        intent = best_intent if above else self.fallback_intent

        return {
            "query": query,
            "intent": intent,
            "confidence": round(best_conf, 4),
            "all_scores": all_scores,
            "above_threshold": above,
        }


# ---------------------------------------------------------------------------
# Module-level convenience singleton
# ---------------------------------------------------------------------------

_DEFAULT_CLASSIFIER: IntentClassifier | None = None


def _get_default_classifier() -> IntentClassifier:
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = IntentClassifier()
    return _DEFAULT_CLASSIFIER


def classify_intent(query: str) -> dict[str, Any]:
    """Classify *query* using the module-level default classifier.

    This is the primary callable expected by the orchestration layer.

    Parameters
    ----------
    query:
        Raw natural-language legal query from the user.

    Returns
    -------
    dict – see :meth:`IntentClassifier.classify`.
    """
    return _get_default_classifier().classify(query)
