# ---------------------------------------------
# Groq-native requirement coverage prompts
# ---------------------------------------------


def _select_prompt(response_format_type: str, json_schema_prompt: str, json_object_prompt: str):
    if response_format_type == "json_schema":
        return "json_schema_prompt", json_schema_prompt
    return "json_object_prompt", json_object_prompt


_REQUIREMENT_EXTRACTOR_JSON_SCHEMA_PROMPT = """
You are a strict requirements engineer. Extract ONLY explicit, functional requirements from the job description.

STRICT RULES:
1. NEVER split one feature into multiple tiny sub-features. Keep them merged.
2. Max requirements is 5. If the text has fewer, extract ONLY that number. Do NOT invent fluff.
3. Absolutely NO tools, frameworks, programming languages, or developer roles allowed.
4. Keep each text under 10 words. No general narrative sentences.

Example:
Input: "Need a backend developer to build a secure REST API using Node.js and JWT."
Output: {"requirements": [{"id": "REQ_1", "text": "Build a secure REST API backend.", "necessity_level": "mandatory"}]}
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
