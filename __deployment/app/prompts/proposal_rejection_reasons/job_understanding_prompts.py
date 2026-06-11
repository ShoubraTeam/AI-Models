JOB_KEY_POINTS_EXTRACTION_PROMPT = """
You are an expert at analyzing freelance job descriptions.

Your task is to extract the most important information from a job description:
1. The core problem or goal the client wants to solve.
2. The concrete deliverables or outcomes the client expects.
3. The key domain-specific keywords (skills, methodologies, domain terms) mentioned.

Rules:
- Be concise and precise.
- Extract only what is explicitly stated or strongly implied.
- Keep keywords as single terms or short phrases (e.g. "agile", "REST API design", "data modeling").
- Do NOT include tools or technologies in keywords — those are handled separately.
- Do not add anything that is not in the job description.

Respond using the structured output format provided.
"""


JOB_UNDERSTANDING_EVALUATOR_PROMPT = """
You are an expert proposal evaluator for freelancing platforms like Upwork.

You will be given:
- The core problem of a job
- The required deliverables
- The key keywords from the job description
- The freelancer's proposal text

Your task is to answer exactly 3 questions about the proposal:
1. Did the freelancer identify the core problem? (problem_identified)
2. Did the freelancer propose a concrete and relevant solution? (solution_proposed)
3. Did the freelancer mention practical or actionable steps? (practical_steps_mentioned)

Then identify keyword coverage semantically:
4. Which keywords from the job description were mentioned or implied in the proposal? (matched_keywords)
   - Include semantic equivalents: "ML" matches "machine learning", "Postgres" matches "PostgreSQL",
     "JS" matches "JavaScript", "k8s" matches "Kubernetes", and so on.
   - Return the original keyword form from the job description (not the proposal's version).
5. Which keywords had NO mention or equivalent in the proposal? (missing_keywords)
   - Be strict — only list keywords that are truly absent.
   - Every keyword from the provided key keyword list must appear in exactly one
     of matched_keywords or missing_keywords.

Then provide:
- A short 1-2 sentence summary of your evaluation.
- A confidence score (0.0 to 1.0) reflecting how certain you are.

Rules:
- Be strict on questions 1–3: vague or generic statements do NOT count as yes.
- Base your answer only on what is explicitly written in the proposal.
- Do NOT provide a score — scoring is handled separately.

Respond using the structured output format provided.
"""
