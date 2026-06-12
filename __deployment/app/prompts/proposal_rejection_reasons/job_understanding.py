# ---------------------------------------------
# Groq-native job understanding prompts
# ---------------------------------------------


def _select_prompt(response_format_type: str, json_schema_prompt: str, json_object_prompt: str):
    if response_format_type == "json_schema":
        return "json_schema_prompt", json_schema_prompt
    return "json_object_prompt", json_object_prompt


_JOB_KEY_POINTS_JSON_SCHEMA_PROMPT = """
You are an expert at analyzing freelance job descriptions.

Extract the most important job key points from the given job description:
1. The core problem or goal the client wants to solve.
2. The concrete deliverables or outcomes the client expects.

Rules:
- Be concise and precise.
- Extract only what is explicitly stated or strongly implied.
- Do not add anything that is not in the job description.
- Return only values that belong to the provided response schema.
"""

_JOB_KEY_POINTS_JSON_OBJECT_PROMPT = _JOB_KEY_POINTS_JSON_SCHEMA_PROMPT + """

Return one valid JSON object with this shape:
{
  "core_problem": "string",
  "required_deliverables": ["string"]
}
"""


def get_job_key_points_extraction_prompt(response_format_type: str):
    return _select_prompt(
        response_format_type,
        _JOB_KEY_POINTS_JSON_SCHEMA_PROMPT,
        _JOB_KEY_POINTS_JSON_OBJECT_PROMPT,
    )


_JOB_UNDERSTANDING_JSON_SCHEMA_PROMPT = """
You are an expert proposal evaluator for freelance platforms.

You will receive extracted job key points, job keywords, and a freelancer proposal. Evaluate whether the proposal demonstrates real understanding of the job.

Decide exactly these points:
1. Whether the freelancer identified the core problem.
2. Whether the freelancer proposed a concrete, relevant solution.
3. Whether the freelancer mentioned practical or actionable steps.
4. Which provided job keywords were matched or semantically implied in the proposal.
5. Which provided job keywords were missing.

Rules:
- Be strict: vague or generic statements do not count as clear understanding.
- Base the evaluation only on the proposal text and the provided job key points.
- Each provided keyword should appear in either matched_keywords or missing_keywords, unless no keywords were provided.
- Return only values that belong to the provided response schema.
"""

_JOB_UNDERSTANDING_JSON_OBJECT_PROMPT = _JOB_UNDERSTANDING_JSON_SCHEMA_PROMPT + """

Return one valid JSON object with this shape:
{
  "problem_identified": true,
  "solution_proposed": true,
  "practical_steps_mentioned": true,
  "matched_keywords": ["string"],
  "missing_keywords": ["string"],
  "summary": "string",
  "confidence_score": 0.0
}
"""


def get_job_understanding_evaluator_prompt(response_format_type: str):
    return _select_prompt(
        response_format_type,
        _JOB_UNDERSTANDING_JSON_SCHEMA_PROMPT,
        _JOB_UNDERSTANDING_JSON_OBJECT_PROMPT,
    )



JOB_KEY_POINTS_EXTRACTION_PROMPT = get_job_key_points_extraction_prompt
JOB_UNDERSTANDING_EVALUATOR_PROMPT = get_job_understanding_evaluator_prompt
