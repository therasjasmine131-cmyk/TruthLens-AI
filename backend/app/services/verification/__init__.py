"""Evidence-based verification pipeline for TruthLens AI.

Modules
-------
* ``language``    - English / Tamil / Tanglish detection + Tanglish normalization
* ``claims``      - atomic claim decomposition, claim types, opinion/prediction
* ``entities``    - lightweight entity extraction
* ``temporal``    - date/relative-time extraction + freshness checks
* ``numerical``   - number extraction + claim/evidence number comparison
* ``negation``    - negation-aware polarity handling
* ``knowledge``   - curated offline knowledge base (authoritative references)
* ``sources``     - transparent source-credibility scoring
* ``evidence``    - retrieval from Wikipedia / NewsAPI / Fact Check / Gemini
* ``relevance``   - SUPPORTS / CONTRADICTS / NEUTRAL classification
* ``scoring``     - verdict engine (REAL / FALSE / UNVERIFIED) + confidence
* ``verifier``    - end-to-end orchestration

All modules are offline-first and fail gracefully when API keys are missing.
"""

from .verifier import verify_text

__all__ = ["verify_text"]