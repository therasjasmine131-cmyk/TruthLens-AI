"""Input validation for the analysis endpoints."""

from __future__ import annotations

from ..config import Config
from .errors import ValidationError

MIN_ARTICLE_CHARS = 20


def validate_inputs(headline: str | None, article: str | None) -> None:
    """Raise ValidationError for empty / malformed / oversized inputs."""
    headline = (headline or "").strip()
    article = (article or "").strip()

    max_headline = Config.MAX_HEADLINE_LENGTH
    max_article = Config.MAX_ARTICLE_LENGTH

    if not headline and not article:
        raise ValidationError("Please provide a headline or an article to analyze.")

    if headline and len(headline) > max_headline:
        raise ValidationError(
            f"Headline is too long ({len(headline)} characters). "
            f"Maximum allowed is {max_headline}."
        )

    if article and len(article) > max_article:
        raise ValidationError(
            f"Article is too long ({len(article)} characters). "
            f"Maximum allowed is {max_article}."
        )

    if article and len(article) < MIN_ARTICLE_CHARS and not headline:
        raise ValidationError(
            "Article is too short for a meaningful analysis. "
            f"Please provide at least {MIN_ARTICLE_CHARS} characters."
        )


def validate_csv_row(row: dict) -> str | None:
    """Return an error message for a malformed batch row, else None."""
    headline = str(row.get("headline") or "").strip()
    article = str(row.get("article") or "").strip()
    if not headline and not article:
        return "row has neither headline nor article"
    if len(headline) > Config.MAX_HEADLINE_LENGTH:
        return "headline too long"
    if len(article) > Config.MAX_ARTICLE_LENGTH:
        return "article too long"
    return None
