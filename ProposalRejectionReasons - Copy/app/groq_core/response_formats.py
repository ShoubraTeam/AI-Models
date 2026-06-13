# ---------------------------------------------------------
# Native Groq response formats for structured model output
# ---------------------------------------------------------

from copy import deepcopy
from typing import Any, TypeAlias

from pydantic import BaseModel

from schemas import (
    ExperienceEvidenceSchema,
    JobKeyPointsSchema,
    JobUnderstandingEvalSchema,
    JobToolResponse,
    ProposalToolsResponse,
    ExtractedRequirementsSchema,
    RequirementCoverageSchema,
    LanguageClarityEvalSchema,
    SuperAgentResponse,
)

SchemaType: TypeAlias = type[BaseModel]

BEST_EFFORT_MODE_MODELS = [
    "openai/gpt-oss-safeguard-20b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
]

STRICT_MODE_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

SUPPORTED_SCHEMAS: tuple[SchemaType, ...] = (
    ExperienceEvidenceSchema,
    JobKeyPointsSchema,
    JobUnderstandingEvalSchema,
    JobToolResponse,
    ProposalToolsResponse,
    ExtractedRequirementsSchema,
    RequirementCoverageSchema,
    LanguageClarityEvalSchema,
    SuperAgentResponse,
)



def _make_groq_compatible_schema(json_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Groq strict JSON schema.
    """
    normalized = deepcopy(json_schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())
                node.setdefault("additionalProperties", False)

            for value in node.values():
                visit(value)

        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(normalized)
    return normalized


def json_schema_format(
    strict: bool,
    schema: SchemaType,
    schema_name: str,
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name"  : schema_name,
            "strict": strict,
            "schema": _make_groq_compatible_schema(schema.model_json_schema()),
        },
    }


def get_response_format(
    model_name : str,
    schema     : SchemaType,
    schema_name: str,
) -> dict[str, Any]:
    if model_name in STRICT_MODE_MODELS:
        return json_schema_format(True, schema, schema_name)

    if model_name in BEST_EFFORT_MODE_MODELS:
        return json_schema_format(False, schema, schema_name)

    return {"type": "json_object"}
