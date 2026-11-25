"""Classical-ML intent classifier for legal queries.

Three intents are supported:

* ``case_summarisation``  – the user wants a summary / overview of a case.
* ``citation_validation`` – the user wants a legal citation verified.
* ``precedent_search``    – the user wants prior case law / precedents found.

The classifier is trained on a small but representative seed corpus using
TF-IDF + Logistic Regression (classical ML, no LLM required at inference
time).  It falls back gracefully to ``precedent_search`` when confidence
is below the configured threshold.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

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
# Seed training corpus
# ---------------------------------------------------------------------------

_SEED_CORPUS: list[tuple[str, str]] = [
    # ---- case_summarisation ------------------------------------------------
    ("summarise the case of Maneka Gandhi vs Union of India", INTENT_CASE_SUMMARISATION),
    ("give me a brief summary of Kesavananda Bharati case", INTENT_CASE_SUMMARISATION),
    ("what happened in the Vishakha vs State of Rajasthan judgment", INTENT_CASE_SUMMARISATION),
    ("can you summarize the judgment in Olga Tellis vs Bombay Municipal Corporation", INTENT_CASE_SUMMARISATION),
    ("provide an overview of the Navtej Singh Johar ruling", INTENT_CASE_SUMMARISATION),
    ("explain the facts and decision in the Shreya Singhal case", INTENT_CASE_SUMMARISATION),
    ("what is the gist of the ADM Jabalpur verdict", INTENT_CASE_SUMMARISATION),
    ("case summary for Indra Sawhney vs Union of India", INTENT_CASE_SUMMARISATION),
    ("briefly describe the Puttaswamy privacy judgment", INTENT_CASE_SUMMARISATION),
    ("what does the MC Mehta environmental case say", INTENT_CASE_SUMMARISATION),
    ("outline the main points of the Shah Bano case", INTENT_CASE_SUMMARISATION),
    ("summarize ruling on reservation for OBC communities", INTENT_CASE_SUMMARISATION),
    ("describe the facts, ratio and decision of Mohd Ahmed Khan vs Shah Bano Begum", INTENT_CASE_SUMMARISATION),
    ("case brief of Arnesh Kumar vs State of Bihar", INTENT_CASE_SUMMARISATION),
    ("what is the summary of the Nirbhaya rape case judgment", INTENT_CASE_SUMMARISATION),

    # ---- citation_validation -----------------------------------------------
    ("validate Section 302 of the Indian Penal Code 1860", INTENT_CITATION_VALIDATION),
    ("is Section 498A of IPC a valid citation", INTENT_CITATION_VALIDATION),
    ("verify Section 420 Indian Penal Code", INTENT_CITATION_VALIDATION),
    ("check whether the citation Section 377 IPC 1860 is correct", INTENT_CITATION_VALIDATION),
    ("validate the citation under Section 138 Negotiable Instruments Act 1881", INTENT_CITATION_VALIDATION),
    ("is Section 124A IPC 1860 still valid", INTENT_CITATION_VALIDATION),
    ("verify Section 125 Code of Criminal Procedure 1973", INTENT_CITATION_VALIDATION),
    ("confirm that the reference to Section 25F Industrial Disputes Act 1947 is correct", INTENT_CITATION_VALIDATION),
    ("validate Section 7 Protection of Children from Sexual Offences Act 2012", INTENT_CITATION_VALIDATION),
    ("check the citation Consumer Protection Act 2019 Section 35", INTENT_CITATION_VALIDATION),
    ("is Section 66A IT Act a valid legal reference", INTENT_CITATION_VALIDATION),
    ("verify Section 304B Indian Penal Code 1860 for dowry death", INTENT_CITATION_VALIDATION),
    ("confirm whether Section 9 Hindu Marriage Act 1955 citation is accurate", INTENT_CITATION_VALIDATION),
    ("validate the statutory reference to Section 2(h) Contract Act 1872", INTENT_CITATION_VALIDATION),
    ("is the citation to Section 73 Indian Contract Act 1872 valid", INTENT_CITATION_VALIDATION),

    # ---- precedent_search --------------------------------------------------
    ("find precedents on right to privacy under Article 21", INTENT_PRECEDENT_SEARCH),
    ("what are the leading cases on anticipatory bail in India", INTENT_PRECEDENT_SEARCH),
    ("search for case law on defamation under Indian law", INTENT_PRECEDENT_SEARCH),
    ("which cases deal with landlord-tenant disputes in India", INTENT_PRECEDENT_SEARCH),
    ("find landmark judgments on freedom of speech and expression", INTENT_PRECEDENT_SEARCH),
    ("precedents for cheque bounce cases under Section 138 NI Act", INTENT_PRECEDENT_SEARCH),
    ("what is the precedent for maintenance of wife under Hindu law", INTENT_PRECEDENT_SEARCH),
    ("cases related to contempt of court in India", INTENT_PRECEDENT_SEARCH),
    ("search prior rulings on motor accident compensation", INTENT_PRECEDENT_SEARCH),
    ("find case law on wrongful termination of employment", INTENT_PRECEDENT_SEARCH),
    ("precedents for environmental pollution liability", INTENT_PRECEDENT_SEARCH),
    ("which judgments govern intellectual property disputes in India", INTENT_PRECEDENT_SEARCH),
    ("look up precedents for cruelty by husband under Section 498A", INTENT_PRECEDENT_SEARCH),
    ("search for cases on consumer grievance against insurance company", INTENT_PRECEDENT_SEARCH),
    ("relevant judgments on cybercrime and data privacy violations", INTENT_PRECEDENT_SEARCH),
]


# ---------------------------------------------------------------------------
# Text pre-processing helpers
# ---------------------------------------------------------------------------

_LEGAL_SYNONYMS: dict[str, str] = {
    r"\bsummarise\b": "summarize",
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
    """TF-IDF + Logistic Regression intent classifier for legal queries.

    Parameters
    ----------
    probability_threshold:
        Minimum confidence for the top prediction to be accepted.
        Below this value the classifier returns the fallback intent.
    fallback_intent:
        Intent returned when no class exceeds *probability_threshold*.
    max_features:
        Maximum TF-IDF vocabulary size.
    ngram_range:
        N-gram range passed to :class:`TfidfVectorizer`.
    C:
        Regularisation parameter for :class:`LogisticRegression`.
    seed_corpus:
        Optional list of ``(query, intent)`` training pairs that *replaces*
        the built-in seed corpus.  Pass extra pairs to extend via
        :meth:`fit_extra`.
    """

    def __init__(
        self,
        probability_threshold: float = 0.40,
        fallback_intent: str = INTENT_PRECEDENT_SEARCH,
        max_features: int = 3000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 5.0,
        seed_corpus: list[tuple[str, str]] | None = None,
    ) -> None:
        self.probability_threshold = probability_threshold
        self.fallback_intent = fallback_intent

        corpus = seed_corpus if seed_corpus is not None else _SEED_CORPUS
        self._queries = [q for q, _ in corpus]
        self._labels = [label for _, label in corpus]

        self._pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        max_features=max_features,
                        ngram_range=ngram_range,
                        sublinear_tf=True,
                        analyzer="word",
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        C=C,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        self._fitted = False
        self._fit_on_seed()

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _fit_on_seed(self) -> None:
        """Fit the pipeline on the internal seed corpus."""
        processed = [_preprocess(q) for q in self._queries]
        self._pipeline.fit(processed, self._labels)
        self._fitted = True
        logger.debug(
            "IntentClassifier fitted on %d seed examples.", len(processed)
        )

    def fit_extra(
        self, queries: list[str], labels: list[str]
    ) -> "IntentClassifier":
        """Extend training data with additional labelled examples and refit.

        Parameters
        ----------
        queries:
            New training query strings.
        labels:
            Corresponding intent labels (must be one of :data:`ALL_INTENTS`).
        """
        if len(queries) != len(labels):
            raise ValueError("queries and labels must have the same length.")
        invalid = [l for l in labels if l not in ALL_INTENTS]
        if invalid:
            raise ValueError(f"Unknown intent label(s): {invalid}")

        self._queries.extend(queries)
        self._labels.extend(labels)
        self._fit_on_seed()
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, query: str) -> dict[str, Any]:
        """Classify *query* and return a structured intent result.

        Returns
        -------
        dict with keys:

        * ``query``       – original query string
        * ``intent``      – predicted intent label
        * ``confidence``  – probability of the predicted intent (0-1)
        * ``all_scores``  – dict mapping every intent to its probability
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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted classifier to *path* using joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self._pipeline,
                "queries": self._queries,
                "labels": self._labels,
                "probability_threshold": self.probability_threshold,
                "fallback_intent": self.fallback_intent,
            },
            path,
        )
        logger.info("IntentClassifier saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        """Load a persisted classifier from *path*."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._pipeline = data["pipeline"]
        obj._queries = data["queries"]
        obj._labels = data["labels"]
        obj.probability_threshold = data["probability_threshold"]
        obj.fallback_intent = data["fallback_intent"]
        obj._fitted = True
        logger.info("IntentClassifier loaded from %s", path)
        return obj


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
