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

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

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


# Declared as a scheme rather than read off a raw header, so it appears in the
# OpenAPI document and /docs offers an Authorize button. Without that a reviewer
# can read the API but cannot exercise it from the browser.
# auto_error=False keeps the 401-with-WWW-Authenticate below rather than
# FastAPI's default 403, which would be the wrong status for a missing token.
bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="Bearer token",
    description="Any token in SCA_PRINCIPALS. Paste the token alone, without the word Bearer.",
)


async def current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    record = settings.parsed_principals().get(credentials.credentials.strip())
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
