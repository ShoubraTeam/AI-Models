from schemas.job_understanding.job_key_points_schema import JobKeyPointsSchema
from schemas.job_understanding.job_understanding_eval_schema import JobUnderstandingEvalSchema
import helpers.config as CFG

JOB_UNDERSTANDING_THRESHOLD = 5.0


def calc_keyword_metrics(
    key_keywords: list[str],
    proposal_text: str
) -> dict:
    """
    Calculate keyword coverage metrics using simple string matching.
    This replaces asking the LLM to do keyword matching — faster, cheaper, deterministic.

    Args:
        key_keywords  : List of keywords extracted from the job description.
        proposal_text : The freelancer's proposal text.

    Returns:
        dict with:
            - matched_keywords      : keywords found in the proposal
            - missing_keywords      : keywords not found in the proposal
            - keyword_coverage_score: float 0.0–1.0
    """
    proposal_lower = proposal_text.lower()

    matched = [kw for kw in key_keywords if kw.lower() in proposal_lower]
    missing = [kw for kw in key_keywords if kw.lower() not in proposal_lower]

    total = len(key_keywords)
    coverage = len(matched) / total if total > 0 else 0.0

    return {
        "matched_keywords"      : matched,
        "missing_keywords"      : missing,
        "keyword_coverage_score": round(coverage, 2),
    }


def calc_job_understanding_score(
    llm_eval: JobUnderstandingEvalSchema,
    keyword_coverage_score: float
) -> float:
    """
    Compute the final job understanding score from:
        - LLM boolean flags (3 questions, weighted)
        - Keyword coverage score (from string matching)

    Weights:
        problem_identified        → 3 points
        solution_proposed         → 3 points
        practical_steps_mentioned → 2 points
        keyword_coverage          → 2 points  (0.0–1.0 scaled to 0–2)
        ─────────────────────────────────────
        Total                     → 10 points

    Returns:
        float score between 0.0 and 10.0
    """
    score = 0.0
    if llm_eval.problem_identified:
        score += 3.0
    if llm_eval.solution_proposed:
        score += 3.0
    if llm_eval.practical_steps_mentioned:
        score += 2.0

    score += keyword_coverage_score * 2.0  # scale 0–1 to 0–2

    return round(score, 2)


def calc_job_understanding_result(
    extraction : JobKeyPointsSchema,
    llm_eval   : JobUnderstandingEvalSchema,
    proposal_text: str,
    threshold  : float = JOB_UNDERSTANDING_THRESHOLD
) -> dict:
    """
    Full processing pipeline:
        1. Compute keyword metrics via string matching
        2. Compute final score from LLM flags + keyword coverage
        3. Apply rule-based acceptance decision

    Args:
        extraction    : Output of JobKeyPointsExtractor
        llm_eval      : Output of JobUnderstandingEvaluator
        proposal_text : The freelancer's proposal (needed for keyword matching)
        threshold     : Minimum passing score (default: 5.0)

    Returns:
        dict with all fields needed for the SuperAgent or rejection report.
    """
    # Step 1 — keyword metrics (pure code, no LLM)
    kw_metrics = calc_keyword_metrics(
        key_keywords=extraction.key_keywords,
        proposal_text=proposal_text
    )

    # Step 2 — final score
    score = calc_job_understanding_score(
        llm_eval=llm_eval,
        keyword_coverage_score=kw_metrics["keyword_coverage_score"]
    )

    # Step 3 — rule-based decision
    accepted = score >= threshold

    rejection_reason = None
    if not accepted:
        rejection_reason = (
            f"The proposal did not demonstrate sufficient understanding of the job. "
            f"Score: {score:.1f}/10 (threshold: {threshold}). "
            f"{llm_eval.summary}"
        )

    return {
        "score"                  : score,
        "confidence"             : llm_eval.confidence_score,
        "accepted"               : accepted,
        "problem_identified"     : llm_eval.problem_identified,
        "solution_proposed"      : llm_eval.solution_proposed,
        "practical_steps"        : llm_eval.practical_steps_mentioned,
        "matched_keywords"       : kw_metrics["matched_keywords"],
        "missing_keywords"       : kw_metrics["missing_keywords"],
        "keyword_coverage_score" : kw_metrics["keyword_coverage_score"],
        "rejection_reason"       : rejection_reason,
        "summary"                : llm_eval.summary,
    }
