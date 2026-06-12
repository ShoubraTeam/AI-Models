# ---------------------------------------------
# Groq-native tool alignment prompts
# ---------------------------------------------


def _select_prompt(response_format_type: str, json_schema_prompt: str, json_object_prompt: str):
    if response_format_type == "json_schema":
        return "json_schema_prompt", json_schema_prompt
    return "json_object_prompt", json_object_prompt


_JOB_TOOLS_JSON_SCHEMA_PROMPT = """
You are a professional HR analyst for freelance job posts.

Analyze the job description and extract only the tools, frameworks, platforms, or libraries explicitly mentioned by the client. For each tool, classify the necessity level as one of: mandatory, recommended, optional, or forbidden.

Rules:
- Do not invent tools that are not present in the job description.
- Do not infer a necessity level unless the wording supports it.
- If the job contains tags or a skills section, include tags that represent concrete tools or platforms.
- Return only values that belong to the provided response schema.
"""

_JOB_TOOLS_JSON_OBJECT_PROMPT = _JOB_TOOLS_JSON_SCHEMA_PROMPT + """

Return one valid JSON object with this shape:
{
  "tools": [
    {
      "tool_name": "string",
      "necessity_level": "mandatory | recommended | optional | forbidden"
    }
  ]
}
"""


def get_job_tools_extraction_prompt(response_format_type: str):
    return _select_prompt(
        response_format_type,
        _JOB_TOOLS_JSON_SCHEMA_PROMPT,
        _JOB_TOOLS_JSON_OBJECT_PROMPT,
    )


_PROPOSAL_TOOLS_JSON_SCHEMA_PROMPT = """
You are a professional proposal reviewer for freelance jobs.

You will receive a list of tools required or mentioned by the client and a freelancer proposal. Evaluate every tool from the provided job_tools_list against the proposal.

For each input tool, determine:
- whether the proposal mentions the tool or a clear semantic equivalent,
- whether the freelancer mentions it with confidence and relevant context,
- and preserve the original tool name and necessity level from the input list.

Rules:
- Review every tool from the job_tools_list.
- Do not add new tools.
- Do not change the input necessity levels.
- Use null for with_confidence when the tool is not found in the proposal.
- Return only values that belong to the provided response schema.
"""

_PROPOSAL_TOOLS_JSON_OBJECT_PROMPT = _PROPOSAL_TOOLS_JSON_SCHEMA_PROMPT + """

Return one valid JSON object with this shape:
{
  "tool_reviews": [
    {
      "tool_name": "string",
      "necessity_level": "mandatory | recommended | optional | forbidden",
      "found_in_proposal": true,
      "with_confidence": true
    }
  ],
  "summary": "string"
}
"""


def get_proposal_tools_extraction_prompt(response_format_type: str):
    return _select_prompt(
        response_format_type,
        _PROPOSAL_TOOLS_JSON_SCHEMA_PROMPT,
        _PROPOSAL_TOOLS_JSON_OBJECT_PROMPT,
    )


JOB_TOOLS_EXTRACTION_PROMPT = get_job_tools_extraction_prompt
PROPOSAL_TOOLS_EXTRACTION_PROMPT = get_proposal_tools_extraction_prompt
