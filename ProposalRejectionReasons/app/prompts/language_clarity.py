# ---------------------------------------------
# Groq-native language clarity prompt
# ---------------------------------------------


def _select_prompt(response_format_type: str, json_schema_prompt: str, json_object_prompt: str):
    if response_format_type == "json_schema":
        return "json_schema_prompt", json_schema_prompt
    return "json_object_prompt", json_object_prompt


_LANGUAGE_CLARITY_JSON_SCHEMA_PROMPT = """
You are an expert proposal reviewer for freelance platforms.

Evaluate only the language quality of the freelancer proposal. Do not judge technical fit or job relevance.

Decide exactly these points:
1. Whether the proposal is clear and easy to understand.
2. Whether the tone is professional and appropriate for a client.
3. Whether the proposal contains vague, exaggerated, or misleading phrasing.

Rules:
- Be strict: confusing wording, careless grammar, or overly vague claims should be flagged.
- Base the evaluation only on the proposal text.
- Return only values that belong to the provided response schema.
"""

_LANGUAGE_CLARITY_JSON_OBJECT_PROMPT = _LANGUAGE_CLARITY_JSON_SCHEMA_PROMPT + """

Return one valid JSON object with this shape:
{
  "is_clear": true,
  "is_professional": true,
  "has_misleading_phrasing": false,
  "summary": "string",
  "confidence_score": 0.0
}
"""


def get_language_clarity_evaluator_prompt(response_format_type: str):
    return _select_prompt(
        response_format_type,
        _LANGUAGE_CLARITY_JSON_SCHEMA_PROMPT,
        _LANGUAGE_CLARITY_JSON_OBJECT_PROMPT,
    )


LANGUAGE_CLARITY_EVALUATOR_PROMPT = get_language_clarity_evaluator_prompt
