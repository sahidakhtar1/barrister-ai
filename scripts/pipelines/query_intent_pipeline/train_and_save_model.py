"""Train the intent classifier on the CSV training data and save it to the
models folder.

Usage
-----
Run from the project root::

    python scripts/pipelines/query_intent_pipeline/train_and_save_model.py

The script will:
1. Load labelled examples from ``data/query_intent/query_intent_training_data.csv``
2. Pre-process and train a TF-IDF + Logistic Regression pipeline.
3. Evaluate with a stratified 5-fold cross-validation and log accuracy.
4. Persist the fitted model to ``models/intent_classifier.joblib``.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project importable when run directly (python scripts/...)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # barrister-ai/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAINING_CSV = PROJECT_ROOT / "data" / "query_intent" / "query_intent_training_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"

# ---------------------------------------------------------------------------
# Hyper-parameters (mirror those in classifier.py)
# ---------------------------------------------------------------------------

MAX_FEATURES: int = 3000
NGRAM_RANGE: tuple[int, int] = (1, 2)
C: float = 5.0
PROBABILITY_THRESHOLD: float = 0.40
FALLBACK_INTENT: str = "precedent_search"

# ---------------------------------------------------------------------------
# Label mapping  (CSV uses UPPER_CASE; classifier uses lower_snake_case)
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, str] = {
    "CASE_SUMMARIZATION": "case_summarisation",
    "CASE_SUMMARISATION": "case_summarisation",
    "CITATION_VALIDATION": "citation_validation",
    "PRECEDENT_SEARCH": "precedent_search",
}

# ---------------------------------------------------------------------------
# Text pre-processing (kept in sync with classifier.py)
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
# Data loading
# ---------------------------------------------------------------------------


def load_training_data(csv_path: Path) -> tuple[list[str], list[str]]:
    """Load and validate the training CSV.

    Expected columns: ``query``, ``intent``

    Returns
    -------
    queries, labels
        Parallel lists of pre-processed query strings and normalised intent labels.
    """
    logger.info("Loading training data from %s", csv_path)
    df = pd.read_csv(csv_path)

    required_cols = {"query", "intent"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Drop rows with nulls in either column
    before = len(df)
    df = df.dropna(subset=["query", "intent"])
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with missing values.", dropped)

    # Normalise intent labels
    df["intent"] = df["intent"].str.strip().map(LABEL_MAP)
    unknown = df["intent"].isna()
    if unknown.any():
        logger.warning(
            "Dropping %d rows with unrecognised intent labels.", unknown.sum()
        )
        df = df[~unknown]

    queries = [_preprocess(q) for q in df["query"].tolist()]
    labels = df["intent"].tolist()

    label_counts = pd.Series(labels).value_counts().to_dict()
    logger.info(
        "Loaded %d training examples. Label distribution: %s",
        len(queries),
        label_counts,
    )
    return queries, labels


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_pipeline() -> Pipeline:
    """Return an unfitted TF-IDF + Logistic Regression sklearn Pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=MAX_FEATURES,
                    ngram_range=NGRAM_RANGE,
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


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate(queries: list[str], labels: list[str]) -> Pipeline:
    """Train the pipeline with cross-validation logging, then refit on all data."""
    pipeline = build_pipeline()

    logger.info("Running 5-fold stratified cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, queries, labels, cv=cv, scoring="accuracy")
    logger.info(
        "CV accuracy: %.4f ± %.4f  (folds: %s)",
        scores.mean(),
        scores.std(),
        [f"{s:.4f}" for s in scores],
    )

    logger.info("Fitting on the full training set …")
    pipeline.fit(queries, labels)
    logger.info("Training complete.")
    return pipeline


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(pipeline: Pipeline, path: Path) -> None:
    """Persist the fitted pipeline and metadata to *path* using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "probability_threshold": PROBABILITY_THRESHOLD,
            "fallback_intent": FALLBACK_INTENT,
        },
        path,
    )
    logger.info("Model saved to %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not TRAINING_CSV.exists():
        logger.error("Training CSV not found: %s", TRAINING_CSV)
        sys.exit(1)

    queries, labels = load_training_data(TRAINING_CSV)
    pipeline = train_and_evaluate(queries, labels)
    save_model(pipeline, MODEL_PATH)
    logger.info("Done.  Run the app to use the saved model.")


if __name__ == "__main__":
    main()
