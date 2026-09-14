"""The agent's only tool.

The agent has genuine agency over *what evidence to gather* — it can reformulate
a question into the vocabulary the standards actually use, and narrow to a
specific standard. It has no tool that writes, decides, or escalates. Every
consequential action downstream of retrieval is ordinary Python.

`strict: true` means the arguments are schema-valid when they arrive, so the
executor never has to defend against a malformed tool call.

The shape is the OpenAI function-tool format, which is what the gateway speaks.
"""

from __future__ import annotations

from typing import Any

_SEARCH_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "The search query, phrased in the vocabulary of the standards "
                "(e.g. 'purchase orderer binding promise before ownership')."
            ),
        },
        "standard_no": {
            "type": ["string", "null"],
            "description": (
                "Optional AAOIFI standard number to restrict the search to, "
                "e.g. '8' for Murabahah. Null searches all indexed standards."
            ),
        },
        "reason": {
            "type": "string",
            "description": "Why this search is needed, in one short sentence.",
        },
    },
    "required": ["query", "standard_no", "reason"],
    "additionalProperties": False,
}

SEARCH_STANDARDS: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_standards",
        "strict": True,
        "parameters": _SEARCH_PARAMETERS,
        "description": (
        "Search the indexed AAOIFI Shari'ah Standards for clauses relevant to a "
        "question. Search using the Shari'ah concepts and contract mechanics at "
        "issue rather than the requester's own wording. Returns the most relevant "
            "clauses with their citation ids."
        ),
    },
}

TOOLS = [SEARCH_STANDARDS]
