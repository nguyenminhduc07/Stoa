"""Natural-language query planning for the drawing QA pipeline.

The model only translates a question into a small, validated query plan. It
never receives drawing contents and never supplies construction facts; local
parsers still calculate answers and attach evidence.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional


_TOOL_NAME = "plan_drawing_query"
_INTENTS = {"count", "area", "length", "volume", "lookup", "describe", "compare", "other"}
_TARGETS = {
    "spaces", "rooms", "bedrooms", "kitchens", "bathrooms", "wc_rooms", "saunas",
    "balconies", "doors", "windows", "walls", "slabs", "stairs", "columns", "beams",
    "toilet_bowls", "sinks", "showers_bathtubs", "sanitary_fixtures", "building", "storeys",
    "materials", "dimensions", "text", "drawing", "other", "unknown",
}
_MEASURES = {"none", "net_floor_area", "area", "length", "side_area", "volume"}

_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": sorted(_INTENTS)},
        "target": {"type": "string", "enum": sorted(_TARGETS)},
        "measure": {"type": "string", "enum": sorted(_MEASURES)},
        "scope": {"type": ["string", "null"]},
        "filters": {"type": "array", "items": {"type": "string"}},
        "canonical_question": {"type": "string"},
        "clarification_question": {"type": ["string", "null"]},
    },
    "required": [
        "intent", "target", "measure", "scope", "filters", "canonical_question",
        "clarification_question",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class QueryPlan:
    intent: str
    target: str
    measure: str
    scope: Optional[str]
    filters: tuple[str, ...]
    canonical_question: str
    clarification_question: Optional[str]

    def public(self) -> Dict[str, Any]:
        return asdict(self) | {"filters": list(self.filters)}


def _api_key() -> Optional[str]:
    return os.environ.get("STOA_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")


def is_configured() -> bool:
    return bool(_api_key())


def plan_question(question: str, file_formats: Iterable[str] = ()) -> Optional[QueryPlan]:
    """Translate a user question into a strict query plan; return None if disabled."""
    api_key = _api_key()
    if not api_key:
        return None

    from openai import OpenAI

    formats = sorted({str(item) for item in file_formats if item})
    client = OpenAI(api_key=api_key, timeout=20.0, max_retries=1)
    response = client.responses.create(
        model=os.environ.get("STOA_LLM_MODEL", "gpt-4o-mini"),
        input=[
            {
                "role": "system",
                "content": (
                    "You are a query planner for construction drawing QA. Call the provided function "
                    "with a structured interpretation of the user's request. Do not answer it and do "
                    "not invent drawing facts. Choose the requested operation and entity, preserve "
                    "all level names, room numbers, units, exclusions, and filters exactly, and set "
                    "scope to null when none is stated. Choose the most specific target in the schema. "
                    "Rewrite the question as a concise English parser query while preserving its meaning. "
                    "Put every requested condition or exclusion in filters. For an explicit storey name, "
                    "preserve the exact name in single quotes in canonical_question. If essential scope is ambiguous, set "
                    "clarification_question; otherwise use null. Use target=rooms for a general room "
                    "count and target=spaces for BIM spaces."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "available_file_formats": formats},
                    ensure_ascii=False,
                ),
            },
        ],
        tools=[{
            "type": "function",
            "name": _TOOL_NAME,
            "description": "Normalize a construction drawing question into a safe query plan.",
            "parameters": _SCHEMA,
            "strict": True,
        }],
        tool_choice={"type": "function", "name": _TOOL_NAME},
        max_output_tokens=350,
        store=False,
    )

    tool_call = next(
        (item for item in response.output
         if getattr(item, "type", None) == "function_call"
         and getattr(item, "name", None) == _TOOL_NAME),
        None,
    )
    if tool_call is None:
        raise ValueError("The LLM returned no drawing query plan.")

    arguments = json.loads(tool_call.arguments)
    plan = QueryPlan(
        intent=arguments["intent"],
        target=arguments["target"],
        measure=arguments["measure"],
        scope=arguments["scope"],
        filters=tuple(str(item)[:120] for item in arguments["filters"][:12]),
        canonical_question=str(arguments["canonical_question"]).strip()[:4000],
        clarification_question=(
            str(arguments["clarification_question"]).strip()[:1000]
            if arguments["clarification_question"] else None
        ),
    )
    if plan.intent not in _INTENTS or plan.target not in _TARGETS or plan.measure not in _MEASURES:
        raise ValueError("The LLM returned a query plan with an unknown operation or target.")
    if not plan.canonical_question:
        raise ValueError("The LLM returned an empty canonical question.")
    return plan
