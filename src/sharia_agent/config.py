"""Central configuration, loaded from the environment once at import time.

Everything tunable lives here so that a deployment is reproducible from its
environment alone — which matters because the audit record pins the settings
that produced each verdict.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Principal(dict):
    """A caller identity resolved from a bearer token."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="SCA_", extra="ignore", case_sensitive=False
    )

    # -- providers ---------------------------------------------------------
    # A single gateway serves reasoning and embeddings. OpenRouter speaks the
    # OpenAI wire format, so swapping it for an in-country host later is a
    # base-URL change rather than a rewrite — which matters because CBUAE
    # residency rules point at a UAE-hosted provider in production.
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Haiku 4.5 rather than a frontier model, because it is the only one this
    # system has actually been evaluated on (§3.0) and the only one deployed.
    # Whether a stronger model earns its cost here is open and argued in §2.2.1;
    # documenting an untested default would be answering that question by
    # assertion.
    model: str = "anthropic/claude-haiku-4.5"
    model_effort: str = "none"
    embed_model: str = "baai/bge-m3"

    # 0 means "ask the provider at ingest time and record it in the manifest".
    embed_dim: int = 0

    # Reranking has no gateway endpoint; see retrieval/rerank.py.
    rerank_backend: str = "llm"                       # llm | local | none
    rerank_model: str = "anthropic/claude-haiku-4.5"  # used when backend == llm
    rerank_local_model: str = "BAAI/bge-reranker-v2-m3"

    # -- vector store ------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    collection: str = "aaoifi_standards"

    # -- retrieval ---------------------------------------------------------
    retrieve_candidates: int = 40
    rerank_top_k: int = 5
    max_retrieval_rounds: int = 3

    # -- guardrails --------------------------------------------------------
    min_rerank_score: float = 0.35
    min_model_confidence: float = 0.70
    min_citations: int = 1

    # -- service -----------------------------------------------------------
    env: str = "dev"
    log_level: str = "INFO"
    log_full_prompts: bool = True
    job_ttl_seconds: int = 3600
    max_concurrent_assessments: int = 8

    # -- auth --------------------------------------------------------------
    principals: str = ""

    @field_validator("model_effort")
    @classmethod
    def _valid_effort(cls, v: str) -> str:
        # "none" omits the parameter. Reasoning-effort is a property of the
        # model, not of every model: Haiku 4.5 predates it and rejects it on the
        # first-party API. The gateway usually drops unsupported parameters, but
        # relying on that would make the model choice silently fragile.
        allowed = {"none", "low", "medium", "high", "xhigh", "max"}
        if v.lower() not in allowed:
            raise ValueError(f"model_effort must be one of {sorted(allowed)}")
        return v.lower()

    @field_validator("rerank_backend")
    @classmethod
    def _valid_backend(cls, v: str) -> str:
        allowed = {"llm", "local", "none"}
        if v.lower() not in allowed:
            raise ValueError(f"rerank_backend must be one of {sorted(allowed)}")
        return v.lower()

    def parsed_principals(self) -> dict[str, dict]:
        """Parse SCA_PRINCIPALS into {token: {"id": ..., "scopes": {...}}}.

        Demo-grade only. Production swaps this for the bank's OIDC introspection;
        the rest of the code only depends on the shape returned here.
        """
        out: dict[str, dict] = {}
        for entry in filter(None, (e.strip() for e in self.principals.split(";"))):
            parts = entry.split(":")
            if len(parts) != 3:
                raise ValueError(f"malformed principal entry: {entry!r}")
            token, pid, scopes = parts
            out[token] = {"id": pid, "scopes": set(scopes.split("|"))}
        return out

    def assert_production_safe(self) -> list[str]:
        """Return a list of settings that are unsafe for a production deployment."""
        problems = []
        if self.env == "prod":
            if self.log_full_prompts:
                problems.append("SCA_LOG_FULL_PROMPTS=true leaks query text into logs")
            if self.principals:
                problems.append("SCA_PRINCIPALS static tokens must not be used in prod")
            if not self.qdrant_api_key:
                problems.append("SCA_QDRANT_API_KEY unset — vector store is unauthenticated")
            if self.rerank_backend == "llm":
                problems.append(
                    "SCA_RERANK_BACKEND=llm sends clause text to a third party on every "
                    "query and does not scale; use 'local' in production"
                )
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
