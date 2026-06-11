# ---------------------------------------------
# Groq-native experience evidence prompts
# ---------------------------------------------


def _select_prompt(response_format_type: str, json_schema_prompt: str, json_object_prompt: str):
    if response_format_type == "json_schema":
        return "json_schema_prompt", json_schema_prompt
    return "json_object_prompt", json_object_prompt


_EXPERIENCE_EVIDENCE_JSON_SCHEMA_PROMPT = """
You are an expert technical recruiter and project auditor for freelance proposals.

Analyze the freelancer proposal against the job description and determine whether the freelancer provides concrete evidence of relevant previous experience.

Classification rules:
- Set has_experience_evidence to true only when the freelancer explicitly mentions a specific past project, case study, portfolio item, or hands-on system previously built.
- Set has_experience_evidence to false when the proposal only contains generic claims, years of experience, certifications, or skill lists without a concrete past deliverable.

Extraction rules:
- If has_experience_evidence is false, extracted_projects must be an empty list.
- If true, extract each concrete past project and explain its relevance to the current job.
- Keep project summaries and relevance analysis concise.
- relevance_score must be between 0.0 and 1.0.
- Return only values that belong to the provided response schema.
"""

_EXPERIENCE_EVIDENCE_JSON_OBJECT_PROMPT = _EXPERIENCE_EVIDENCE_JSON_SCHEMA_PROMPT + """

Return one valid JSON object with this shape:
{
  "has_experience_evidence": true,
  "extracted_projects": [
    {
      "project_overview": "string",
      "relevance_analysis": "string",
      "relevance_score": 0.0
    }
  ],
  "summary": "string"
}
"""


def get_experience_evidence_prompt(response_format_type: str):
    return _select_prompt(
        response_format_type,
        _EXPERIENCE_EVIDENCE_JSON_SCHEMA_PROMPT,
        _EXPERIENCE_EVIDENCE_JSON_OBJECT_PROMPT,
    )


EXPERIENCE_EVIDENCE_PROMPT = get_experience_evidence_prompt
