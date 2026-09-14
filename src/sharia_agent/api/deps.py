"""Authentication and authorisation.

Authorisation runs **before** the pipeline starts, not inside it. The model is
untrusted input: by the time a tool call exists, the decision about whether this
caller may cause retrieval and spend has to have already been made.

The static token map here is demo scaffolding. Production resolves the bearer
token against the bank's OIDC provider and maps group claims onto the same
`scopes` shape, so nothing downstream of this function changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, status

from ..config import Settings, get_settings


@dataclass(frozen=True)
class Principal:
    id: str
    scopes: frozenset[str]

    def require(self, scope: str) -> None:
        if scope not in self.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"principal lacks required scope '{scope}'",
            )


async def current_principal(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> Principal:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    record = settings.parsed_principals().get(token)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return Principal(id=record["id"], scopes=frozenset(record["scopes"]))


async def require_assess(
    principal: Principal = Depends(current_principal),
) -> Principal:
    principal.require("assess")
    return principal
