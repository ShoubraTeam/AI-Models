from schemas.job_understanding.job_key_points_schema import JobKeyPointsSchema
from schemas.job_understanding.job_understanding_eval_schema import JobUnderstandingEvalSchema
from schemas import FinalSubagentResult

JOB_UNDERSTANDING_THRESHOLD = 0.5  # normalized 0.0–1.0









# -------------------------------- Pre-Processing ---------------------------------

def prepare_job_key_points_extractor_ip(job_desc: str) -> str:
    return f"# Job Description:\n{job_desc}"


def prepare_job_undertanding_evaluator_ip(
    core_problem         : str,
    required_deliverables: list[str],
    key_keywords         : list[str],
    proposal_text        : str
) -> str:

    required_deliverables_str = ""
    for deliverable in required_deliverables:
        required_deliverables_str += f"\t- {deliverable}"

    key_keywords_str = ", ".join(key_keywords)

    return (
        f"Core Problem:\n{core_problem}\n\n"
        f"Required Deliverables:\n{required_deliverables_str}\n\n"
        f"Key Keywords:\n{key_keywords_str}\n\n"
        f"Freelancer Proposal:\n{proposal_text}"
    )


# -------------------------------- Post-Processing ---------------------------------


def calc_keyword_coverage_score(
    matched_keywords: list[str],
    missing_keywords: list[str],
) -> float:
    """
    Compute keyword coverage ratio from LLM-identified matched/missing keywords.
    The LLM handles semantic matching (ML == machine learning, etc.).

    Args:
        matched_keywords : Keywords the LLM found in the proposal (with semantic matching).
        missing_keywords : Keywords the LLM found absent in the proposal.

    Returns:
        float coverage score between 0.0 and 1.0
    """
    total = len(matched_keywords) + len(missing_keywords)
    if total == 0:
        return 0.0
    return round(len(matched_keywords) / total, 2)


def calc_job_understanding_score(
    llm_eval             : JobUnderstandingEvalSchema,
    keyword_coverage_score: float
) -> float:
    """
    Compute the final job understanding score (normalized 0.0–1.0).

    Weights:
        problem_identified        → 3 points
        solution_proposed         → 3 points
        practical_steps_mentioned → 2 points
        keyword_coverage          → 2 points  (0.0–1.0 scaled to 0–2)
        ─────────────────────────────────────
        Total                     → 10 points → divided by 10 → 0.0–1.0

    Returns:
        float score between 0.0 and 1.0
    """
    score = 0.0
    if llm_eval.problem_identified:
        score += 3.0
    if llm_eval.solution_proposed:
        score += 3.0
    if llm_eval.practical_steps_mentioned:
        score += 2.0

    score += keyword_coverage_score * 2.0  # scale 0–1 to 0–2

    return round(score / 10.0, 2)  # normalize to 0.0–1.0


def build_reasons(
    llm_eval              : JobUnderstandingEvalSchema,
    keyword_coverage_score: float,
    accepted              : bool
) -> list[str]:
    """
    Build a list of specific, actionable reasons from the flags and keyword metrics.
    Returns acceptance reasons if accepted, rejection reasons if not.
    Each reason maps directly to one check — consumed by the SuperAgent.

    Args:
        llm_eval              : Output of JobUnderstandingEvaluator.
        keyword_coverage_score: Coverage ratio from calc_keyword_coverage_score.
        accepted              : Whether the proposal passed the threshold.

    Returns:
        List of reason strings (10–100 chars each to satisfy FinalSubagentResult).
    """
    reasons = []

    if accepted:
        if llm_eval.problem_identified:
            reasons.append("Freelancer clearly identified the core problem.")
        if llm_eval.solution_proposed:
            reasons.append("A concrete and relevant solution was proposed.")
        if llm_eval.practical_steps_mentioned:
            reasons.append("Practical steps and methodology were explained.")
        if keyword_coverage_score >= 0.5:
            reasons.append("Key topics from the job description were covered.")
    else:
        if not llm_eval.problem_identified:
            reasons.append("Proposal doesn't show understanding of client needs.")
        if not llm_eval.solution_proposed:
            reasons.append("No concrete solution was proposed for the problem.")
        if not llm_eval.practical_steps_mentioned:
            reasons.append("No approach or methodology was explained.")
        if keyword_coverage_score < 0.5 and llm_eval.missing_keywords:
            missing = ", ".join(llm_eval.missing_keywords[:3])  # cap to avoid > 100 chars
            reasons.append(f"Missing key topics the client cares about: {missing}.")

    return reasons


def calc_job_understanding_result(
    llm_eval : JobUnderstandingEvalSchema,
    threshold: float = JOB_UNDERSTANDING_THRESHOLD
) -> FinalSubagentResult:
    """
    Full processing pipeline:
        1. Compute keyword coverage score from LLM-identified matched/missing
        2. Compute final normalized score (0.0–1.0)
        3. Apply rule-based acceptance decision
        4. Build specific reasons from flags
        5. Return FinalSubagentResult

    Note: extraction (JobKeyPointsSchema) is no longer needed here —
    the LLM evaluator now handles semantic keyword matching directly.

    Args:
        llm_eval  : Output of JobUnderstandingEvaluator (includes matched/missing keywords).
        threshold : Minimum passing score (default: 0.5).

    Returns:
        FinalSubagentResult consumed by the SuperAgent.
    """
    # Step 1 — keyword coverage from LLM semantic matching (no string comparison)
    keyword_coverage_score = calc_keyword_coverage_score(
        matched_keywords = llm_eval.matched_keywords,
        missing_keywords = llm_eval.missing_keywords,
    )

    # Step 2 — final normalized score
    score = calc_job_understanding_score(
        llm_eval              = llm_eval,
        keyword_coverage_score = keyword_coverage_score,
    )

    # Step 3 — rule-based acceptance decision
    accepted = score >= threshold

    # Step 4 — specific reasons from flags
    reasons = build_reasons(
        llm_eval              = llm_eval,
        keyword_coverage_score = keyword_coverage_score,
        accepted              = accepted,
    )

    # Step 5 — build final result
    return FinalSubagentResult(
        score              = score,
        accepted           = accepted,
        summary            = llm_eval.summary,
        acceptance_reasons = reasons if accepted     else None,
        rejection_reasons  = reasons if not accepted else None,
    )
