"""Intent classification and agent routing for Barrister_AI."""

from barrister_ai.intent.classifier import IntentClassifier, classify_intent
from barrister_ai.intent.router import IntentRouter, route_query

__all__ = [
    "IntentClassifier",
    "classify_intent",
    "IntentRouter",
    "route_query",
]
