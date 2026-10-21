import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.agent.claim_extractor import (
    ClaimExtractor,
    MalformedModelResponse,
)
from app.agent.openrouter_client import (
    OpenRouterAuthenticationError,
    OpenRouterClient,
    OpenRouterConfigurationError,
    OpenRouterModelUnavailableError,
    OpenRouterProviderError,
    OpenRouterRateLimitError,
    OpenRouterTimeoutError,
)
from app.agent.validators import ClaimValidationError, validate_claim_span
from app.config import get_settings


class FakeCompletionClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return self.response


def _response(*claim_texts: str) -> str:
    return json.dumps({"claims": [{"claim_text": text} for text in claim_texts]})


def test_extracts_a_simple_factual_claim() -> None:
    extractor = ClaimExtractor(FakeCompletionClient(_response("It weighs 2.5 kg.")))

    result = extractor.extract("A01", "It weighs 2.5 kg.")

    assert len(result.claims) == 1
    assert result.claims[0].claim_id == "A01:0:17"
    assert result.claims[0].claim_text == "It weighs 2.5 kg."


def test_extracts_multiple_sentences_separately_and_in_order() -> None:
    answer = "It weighs 2.5 kg. Battery life is 12 hours."
    extractor = ClaimExtractor(
        FakeCompletionClient(_response("It weighs 2.5 kg.", "Battery life is 12 hours."))
    )

    result = extractor.extract("A02", answer)

    assert [claim.claim_text for claim in result.claims] == [
        "It weighs 2.5 kg.",
        "Battery life is 12 hours.",
    ]


def test_returns_no_claims_for_non_factual_answer() -> None:
    extractor = ClaimExtractor(FakeCompletionClient(_response()))

    result = extractor.extract("A03", "This is my favorite speaker.")

    assert result.claims == ()


def test_empty_answer_does_not_call_provider() -> None:
    client = FakeCompletionClient(_response("Not present."))

    result = ClaimExtractor(client).extract("A04", "")

    assert result.claims == ()
    assert client.prompts == []


def test_offsets_use_unicode_code_point_indices() -> None:
    answer = "🎵 Élan is a speaker."
    extractor = ClaimExtractor(FakeCompletionClient(_response("Élan is a speaker.")))

    claim = extractor.extract("A05", answer).claims[0]

    assert claim.start_char == 2
    assert answer[claim.start_char : claim.end_char] == claim.claim_text


def test_preserves_exact_punctuation_and_quotes() -> None:
    text = 'The manual says, "Bluetooth 5.3 is supported."'
    extractor = ClaimExtractor(FakeCompletionClient(_response(text)))

    claim = extractor.extract("A06", text).claims[0]

    assert claim.claim_text == text
    assert claim.end_char == len(text)


def test_preserves_whitespace_inside_claim() -> None:
    text = "Battery life is 12 hours,\nunder test conditions."
    extractor = ClaimExtractor(FakeCompletionClient(_response(text)))

    claim = extractor.extract("A07", text).claims[0]

    assert claim.claim_text == text
    assert claim.claim_text.count("\n") == 1


def test_offsets_skip_leading_whitespace_without_changing_source() -> None:
    answer = " \tIt is 12 hours.  \n"
    text = "It is 12 hours."
    extractor = ClaimExtractor(FakeCompletionClient(_response(text)))

    claim = extractor.extract("A07B", answer).claims[0]

    assert claim.start_char == 2
    assert answer[claim.start_char : claim.end_char] == text
    assert answer.endswith("  \n")


def test_sentence_with_abbreviation_and_parentheses_is_kept_intact() -> None:
    text = "Dr. Rao said the speaker weighs 2.5 kg (without packaging)."
    extractor = ClaimExtractor(FakeCompletionClient(_response(text)))

    claim = extractor.extract("A07C", text).claims[0]

    assert claim.claim_text == text
    assert claim.start_char == 0
    assert claim.end_char == len(text)


def test_rejects_combined_sentences_as_one_claim() -> None:
    answer = "The speaker weighs 2 kg. Battery life is 12 hours."
    extractor = ClaimExtractor(
        FakeCompletionClient(_response(answer))
    )

    with pytest.raises(ClaimValidationError, match="multiple sentences"):
        extractor.extract("A07D", answer)


@pytest.mark.parametrize(
    ("start_char", "end_char", "message"),
    [
        (-1, 10, "non-negative"),
        (0, 50, "beyond"),
    ],
)
def test_rejects_invalid_start_or_end_offsets(
    start_char: int, end_char: int, message: str
) -> None:
    with pytest.raises(ClaimValidationError, match=message):
        validate_claim_span(
            answer_id="A08",
            ai_answer="A factual sentence.",
            start_char=start_char,
            end_char=end_char,
            claim_text="A factual sentence.",
        )


def test_rejects_claim_text_that_does_not_match_source_span() -> None:
    with pytest.raises(ClaimValidationError, match="does not exactly match"):
        validate_claim_span(
            answer_id="A09",
            ai_answer="A factual sentence.",
            start_char=0,
            end_char=19,
            claim_text="A fabricated sentence.",
        )


def test_duplicate_claim_is_rejected_when_source_occurrence_is_missing() -> None:
    extractor = ClaimExtractor(
        FakeCompletionClient(_response("It weighs 2 kg.", "It weighs 2 kg."))
    )

    with pytest.raises(ClaimValidationError, match="source spans are ambiguous"):
        extractor.extract("A10", "It weighs 2 kg.")


def test_repeated_source_sentences_get_distinct_offsets() -> None:
    sentence = "It weighs 2 kg."
    extractor = ClaimExtractor(
        FakeCompletionClient(_response(sentence, sentence))
    )

    result = extractor.extract("A11", f"{sentence} {sentence}")

    assert [claim.start_char for claim in result.claims] == [0, len(sentence) + 1]


def test_rejects_ambiguous_single_claim_when_sentence_repeats_in_source() -> None:
    sentence = "It weighs 2 kg."
    extractor = ClaimExtractor(FakeCompletionClient(_response(sentence)))

    with pytest.raises(ClaimValidationError, match="source spans are ambiguous"):
        extractor.extract("A11B", f"{sentence} {sentence}")


def test_malformed_model_output_raises_clear_error() -> None:
    extractor = ClaimExtractor(FakeCompletionClient("not json"))

    with pytest.raises(MalformedModelResponse, match="claims array"):
        extractor.extract("A12", "A factual sentence.")


def test_missing_openrouter_api_key_raises_configuration_error() -> None:
    settings = replace(
        get_settings(),
        model_name="test-model",
        openrouter_api_key=None,
    )

    with pytest.raises(OpenRouterConfigurationError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(settings)


def _settings_with_key():
    return replace(
        get_settings(),
        model_name="test-model",
        openrouter_api_key="test-secret-never-log",
    )


def _sdk_client_that_raises(error: Exception):
    def create(**_: object) -> object:
        raise error

    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def test_provider_timeout_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.agent.openrouter_client as openrouter_module

    class FakeTimeout(Exception):
        pass

    monkeypatch.setattr(openrouter_module, "APITimeoutError", FakeTimeout)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=_sdk_client_that_raises(FakeTimeout("test-secret-never-log")),
    )

    with pytest.raises(OpenRouterTimeoutError) as error:
        client.complete("prompt")

    assert "test-secret-never-log" not in str(error.value)


def test_provider_network_error_is_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.agent.openrouter_client as openrouter_module

    class FakeConnectionError(Exception):
        pass

    monkeypatch.setattr(openrouter_module, "APIConnectionError", FakeConnectionError)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=_sdk_client_that_raises(
            FakeConnectionError("test-secret-never-log")
        ),
    )

    with pytest.raises(OpenRouterProviderError, match="Could not connect"):
        client.complete("prompt")


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (401, OpenRouterAuthenticationError),
        (404, OpenRouterModelUnavailableError),
        (429, OpenRouterRateLimitError),
        (500, OpenRouterProviderError),
    ],
)
def test_provider_api_errors_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    expected_error: type[Exception],
) -> None:
    import app.agent.openrouter_client as openrouter_module

    class FakeStatusError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__("test-secret-never-log")
            self.status_code = status

    monkeypatch.setattr(openrouter_module, "APIStatusError", FakeStatusError)
    client = OpenRouterClient(
        _settings_with_key(),
        sdk_client=_sdk_client_that_raises(FakeStatusError(status_code)),
    )

    with pytest.raises(expected_error) as error:
        client.complete("prompt")

    assert "test-secret-never-log" not in str(error.value)


def test_optional_live_openrouter_integration() -> None:
    import os

    if os.getenv("RUN_OPENROUTER_INTEGRATION") != "1":
        pytest.skip("Set RUN_OPENROUTER_INTEGRATION=1 to enable the live test")
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is required for the live test")

    from app.agent.claim_extractor import create_default_extractor

    result = create_default_extractor().extract(
        "INTEGRATION",
        "Bluetooth 5.3 is supported.",
    )

    assert result.claims
    assert result.claims[0].claim_text == "Bluetooth 5.3 is supported."
