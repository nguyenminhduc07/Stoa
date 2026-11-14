"""Natural-language query planning for the drawing QA pipeline.

The model only translates a question into a small, validated query plan. It
never receives drawing contents and never supplies construction facts; local
parsers still calculate answers and attach evidence.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional


_TOOL_NAME = "plan_drawing_query"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_INTENTS = {"count", "area", "length", "volume", "lookup", "describe", "compare", "other"}
_TARGETS = {
    "spaces", "rooms", "bedrooms", "kitchens", "bathrooms", "wc_rooms", "saunas",
    "balconies", "doors", "windows", "walls", "slabs", "stairs", "columns", "beams",
    "toilet_bowls", "sinks", "showers_bathtubs", "sanitary_fixtures", "building", "storeys",
    "materials", "dimensions", "text", "drawing", "other", "unknown",
}
_MEASURES = {"none", "net_floor_area", "area", "length", "side_area", "volume"}

_SYSTEM_PROMPT = (
    "You are a query planner for construction drawing QA. Return a structured "
    "interpretation of the user's request. Do not answer it and do not invent drawing facts. "
    "Choose the requested operation and entity, preserve all level names, room numbers, units, "
    "exclusions, and filters exactly, and set scope to null when none is stated. Choose the most "
    "specific target in the schema. Rewrite the question as a concise English parser query while "
    "preserving its meaning. Put every requested condition or exclusion in filters. For an "
    "explicit storey name, preserve the exact name in single quotes in canonical_question. If "
    "essential scope is ambiguous, set clarification_question; otherwise use null. Use "
    "target=rooms for a general room count and target=spaces for BIM spaces."
)

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

_GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {"type": "STRING", "enum": sorted(_INTENTS)},
        "target": {"type": "STRING", "enum": sorted(_TARGETS)},
        "measure": {"type": "STRING", "enum": sorted(_MEASURES)},
        "scope": {"type": "STRING", "nullable": True},
        "filters": {"type": "ARRAY", "items": {"type": "STRING"}},
        "canonical_question": {"type": "STRING"},
        "clarification_question": {"type": "STRING", "nullable": True},
    },
    "required": [
        "intent", "target", "measure", "scope", "filters", "canonical_question",
        "clarification_question",
    ],
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


def _gemini_api_key() -> Optional[str]:
    key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if key:
        return key
    stoa_key = os.environ.get("STOA_LLM_API_KEY", "").strip()
    if stoa_key and not stoa_key.startswith("sk-"):
        return stoa_key
    return None


def _openai_api_key() -> Optional[str]:
    stoa_key = os.environ.get("STOA_LLM_API_KEY", "").strip()
    if stoa_key.startswith("sk-"):
        return stoa_key
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    return openai_key or None


def is_configured() -> bool:
    return bool(_gemini_api_key() or _openai_api_key())


def _parse_plan_arguments(arguments: Dict[str, Any]) -> QueryPlan:
    plan = QueryPlan(
        intent=arguments["intent"],
        target=arguments["target"],
        measure=arguments["measure"],
        scope=arguments.get("scope"),
        filters=tuple(str(item)[:120] for item in (arguments.get("filters") or [])[:12]),
        canonical_question=str(arguments["canonical_question"]).strip()[:4000],
        clarification_question=(
            str(arguments["clarification_question"]).strip()[:1000]
            if arguments.get("clarification_question") else None
        ),
    )
    if plan.intent not in _INTENTS or plan.target not in _TARGETS or plan.measure not in _MEASURES:
        raise ValueError("The LLM returned a query plan with an unknown operation or target.")
    if not plan.canonical_question:
        raise ValueError("The LLM returned an empty canonical question.")
    return plan


def _plan_with_gemini(api_key: str, user_payload: str) -> QueryPlan:
    model = (os.environ.get("STOA_LLM_MODEL") or os.environ.get("GEMINI_MODEL") or _DEFAULT_GEMINI_MODEL).strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{urllib.parse.quote(model, safe='')}:generateContent"
    body = json.dumps({
        "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_payload}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 350,
            "responseMimeType": "application/json",
            "responseSchema": _GEMINI_SCHEMA,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini returned no candidates for the drawing query plan.")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(str(part.get("text", "")) for part in parts).strip()
    if not text:
        raise ValueError("Gemini returned an empty drawing query plan.")
    return _parse_plan_arguments(json.loads(text))


def _plan_with_openai(api_key: str, user_payload: str) -> QueryPlan:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=20.0, max_retries=1)
    response = client.responses.create(
        model=os.environ.get("STOA_LLM_MODEL", _DEFAULT_OPENAI_MODEL),
        input=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
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
    return _parse_plan_arguments(json.loads(tool_call.arguments))


def plan_question(question: str, file_formats: Iterable[str] = ()) -> Optional[QueryPlan]:
    """Translate a user question into a strict query plan; return None if disabled."""
    formats = sorted({str(item) for item in file_formats if item})
    user_payload = json.dumps(
        {"question": question, "available_file_formats": formats},
        ensure_ascii=False,
    )

    gemini_key = _gemini_api_key()
    if gemini_key:
        return _plan_with_gemini(gemini_key, user_payload)

    openai_key = _openai_api_key()
    if openai_key:
        return _plan_with_openai(openai_key, user_payload)

    return None
