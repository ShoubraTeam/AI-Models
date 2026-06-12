# ---------------------------------------------
# Groq-native requirement coverage prompts
# ---------------------------------------------


def _select_prompt(response_format_type: str, json_schema_prompt: str, json_object_prompt: str):
    if response_format_type == "json_schema":
        return "json_schema_prompt", json_schema_prompt
    return "json_object_prompt", json_object_prompt


_REQUIREMENT_EXTRACTOR_JSON_SCHEMA_PROMPT = """
You are an expert requirements engineer.

Analyze the job description and extract a clean list of atomic functional requirements, features, or constraints. Assign each requirement a necessity level: mandatory, recommended, optional, or forbidden.

Rules:
- Extract at most 10 requirements.
- Use sequential IDs: REQ_1, REQ_2, REQ_3, and so on.
- Focus on functional capabilities and explicit constraints.
- Do not extract specific developer tools, frameworks, or programming languages as requirements.
- Return only values that belong to the provided response schema.
"""

_REQUIREMENT_EXTRACTOR_JSON_OBJECT_PROMPT = _REQUIREMENT_EXTRACTOR_JSON_SCHEMA_PROMPT + """

Return one valid JSON object with this shape:
{
  "requirements": [
    {
      "id": "REQ_1",
      "text": "string",
      "necessity_level": "mandatory | recommended | optional | forbidden"
    }
  ]
}
"""


def get_requirement_extractor_prompt(response_format_type: str):
    return _select_prompt(
        response_format_type,
        _REQUIREMENT_EXTRACTOR_JSON_SCHEMA_PROMPT,
        _REQUIREMENT_EXTRACTOR_JSON_OBJECT_PROMPT,
    )


_REQUIREMENT_MATCHER_JSON_SCHEMA_PROMPT = """
You are a strict project-management auditor.

You will receive a list of extracted client requirements and a freelancer proposal. Evaluate whether the proposal satisfies each requirement.

Rules:
- Evaluate every input requirement by its exact original ID.
- Preserve requirement IDs exactly; do not rename, re-index, or invent IDs.
- A requirement is covered when the proposal explicitly or semantically addresses its functional intent.
- For forbidden requirements, mark the ID as missing when the proposal violates the constraint; mark it covered when the proposal respects the constraint.
- Put each requirement ID in exactly one of requirements_covered_ids or missing_requirements_ids.
- Return only values that belong to the provided response schema.
"""

_REQUIREMENT_MATCHER_JSON_OBJECT_PROMPT = _REQUIREMENT_MATCHER_JSON_SCHEMA_PROMPT + """

Return one valid JSON object with this shape:
{
  "details": "string",
  "requirements_covered_ids": ["REQ_1"],
  "missing_requirements_ids": ["REQ_2"],
  "summary": "string"
}
"""


def get_requirement_matcher_prompt(response_format_type: str):
    return _select_prompt(
        response_format_type,
        _REQUIREMENT_MATCHER_JSON_SCHEMA_PROMPT,
        _REQUIREMENT_MATCHER_JSON_OBJECT_PROMPT,
    )


REQUIREMENT_EXTRACTOR_PROMPT = get_requirement_extractor_prompt
REQUIREMENT_MATCHER_PROMPT = get_requirement_matcher_prompt
