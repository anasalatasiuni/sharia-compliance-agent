"""Settings behaviour that the first-run experience depends on."""

from __future__ import annotations

import pytest

from sharia_agent.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


@pytest.mark.parametrize(
    "key,expected",
    [
        ("", False),
        ("   ", False),
        # scripts/bootstrap.sh copies .env.example verbatim, so this exact string
        # is what a reviewer has configured before they edit anything.
        ("sk-or-v1-...", False),
        ("sk-or-v1-abc123", True),
    ],
)
def test_placeholder_key_does_not_count_as_configured(key: str, expected: bool) -> None:
    assert _settings(OPENROUTER_API_KEY=key).has_openrouter_key is expected


def test_principals_parse_into_tokens_and_scopes() -> None:
    parsed = _settings(principals="t1:a@mal.ae:assess;t2:b@mal.ae:assess|review").parsed_principals()
    assert parsed["t1"] == {"id": "a@mal.ae", "scopes": {"assess"}}
    assert parsed["t2"]["scopes"] == {"assess", "review"}


def test_malformed_principal_entry_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="malformed principal"):
        _settings(principals="token-without-scopes:a@mal.ae").parsed_principals()
