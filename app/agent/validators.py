"""Validation and source-offset calculation for extracted claims."""

import re

from app.schemas import ClaimExtractionResult, ExtractedClaim


_ABBREVIATION_ENDINGS = re.compile(
    r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|approx|fig|inc|e\.g|i\.e)\."
    r"|(?:[A-Za-z]\.){2,})$",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY = re.compile(r"""[.!?]["'”’)\]}]*\s+(?=["'“‘(\[{A-Z0-9])""")


class ClaimValidationError(ValueError):
    """Raised when an extracted claim cannot be safely mapped to its source."""


def validate_claim_span(
    *,
    answer_id: str,
    ai_answer: str,
    start_char: int,
    end_char: int,
    claim_text: str,
) -> ExtractedClaim:
    """Validate offsets against the original Python Unicode string."""
    if start_char < 0:
        raise ClaimValidationError("Claim start_char must be non-negative")
    if end_char <= start_char:
        raise ClaimValidationError("Claim end_char must be greater than start_char")
    if end_char > len(ai_answer):
        raise ClaimValidationError("Claim span extends beyond the original answer")
    if not claim_text:
        raise ClaimValidationError("Claim text must not be empty")
    if claim_text != claim_text.strip():
        raise ClaimValidationError("Claim text must not include surrounding whitespace")
    if ai_answer[start_char:end_char] != claim_text:
        raise ClaimValidationError(
            "Claim text does not exactly match the original answer at the supplied offsets"
        )

    if start_char > 0 and not ai_answer[start_char - 1].isspace():
        raise ClaimValidationError("Claim span starts inside a sentence token")
    if end_char < len(ai_answer) and not ai_answer[end_char].isspace():
        raise ClaimValidationError("Claim span ends inside a sentence token")

    sentence_end = claim_text.rstrip()
    while sentence_end and sentence_end[-1] in "\"'”’)]}":
        sentence_end = sentence_end[:-1]
    if not sentence_end or sentence_end[-1] not in ".!?。！？…":
        raise ClaimValidationError("Claim span must contain a complete punctuated sentence")
    for boundary in _SENTENCE_BOUNDARY.finditer(claim_text):
        preceding_text = claim_text[: boundary.start() + 1].rstrip()
        if _ABBREVIATION_ENDINGS.search(preceding_text):
            continue
        raise ClaimValidationError("Claim span contains multiple sentences")

    return ExtractedClaim(
        claim_id=f"{answer_id}:{start_char}:{end_char}",
        answer_id=answer_id,
        start_char=start_char,
        end_char=end_char,
        claim_text=claim_text,
    )


def build_extraction_result(
    *,
    answer_id: str,
    ai_answer: str,
    claim_texts: tuple[str, ...],
) -> ClaimExtractionResult:
    """Locate exact claim text in order; repeated source sentences remain distinct."""
    response_counts: dict[str, int] = {}
    for claim_text in claim_texts:
        response_counts[claim_text] = response_counts.get(claim_text, 0) + 1

    for claim_text, response_count in response_counts.items():
        source_count = ai_answer.count(claim_text)
        if source_count != response_count:
            raise ClaimValidationError(
                f"Claim text occurs {source_count} time(s) in the answer but the model "
                f"returned it {response_count} time(s); source spans are ambiguous"
            )

    claims: list[ExtractedClaim] = []
    search_from = 0

    for index, claim_text in enumerate(claim_texts, start=1):
        if not claim_text:
            raise ClaimValidationError(f"Claim {index} has empty claim_text")
        if claim_text != claim_text.strip():
            raise ClaimValidationError(
                f"Claim {index} contains surrounding whitespace not present in its span"
            )
        start_char = ai_answer.find(claim_text, search_from)
        if start_char < 0:
            raise ClaimValidationError(
                f"Claim {index} cannot be located exactly in the original answer "
                "in source order"
            )
        end_char = start_char + len(claim_text)
        claims.append(
            validate_claim_span(
                answer_id=answer_id,
                ai_answer=ai_answer,
                start_char=start_char,
                end_char=end_char,
                claim_text=claim_text,
            )
        )
        search_from = end_char

    return ClaimExtractionResult(answer_id=answer_id, claims=tuple(claims))
