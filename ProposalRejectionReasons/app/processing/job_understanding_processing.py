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

    total    = len(key_keywords)
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


def build_rejection_reasons(
    llm_eval  : JobUnderstandingEvalSchema,
    kw_metrics: dict
) -> list[str]:
    """
    Build a list of specific, actionable rejection reasons from the flags and metrics.
    Each reason maps directly to one failed check.
    This list is what the SuperAgent will consume to generate recommendations.

    Args:
        llm_eval   : Output of JobUnderstandingEvaluator.
        kw_metrics : Output of calc_keyword_metrics.

    Returns:
        List of specific rejection reason strings. Empty if everything passed.
    """
    reasons = []

    if not llm_eval.problem_identified:
        reasons.append(
            "Your proposal doesn't show you understood what the client needs."
        )
    if not llm_eval.solution_proposed:
        reasons.append(
            "You didn't propose how you would solve the problem."
        )
    if not llm_eval.practical_steps_mentioned:
        reasons.append(
            "You didn't explain your approach or methodology."
        )
    if kw_metrics["keyword_coverage_score"] < 0.5:
        missing = ", ".join(kw_metrics["missing_keywords"])
        reasons.append(
            f"You missed key topics the client cares about: {missing}."
        )

    return reasons


def calc_job_understanding_result(
    extraction   : JobKeyPointsSchema,
    llm_eval     : JobUnderstandingEvalSchema,
    proposal_text: str,
    threshold    : float = JOB_UNDERSTANDING_THRESHOLD
) -> dict:
    """
    Full processing pipeline:
        1. Compute keyword metrics via string matching
        2. Compute final score from LLM flags + keyword coverage
        3. Build specific rejection reasons from flags
        4. Apply rule-based acceptance decision

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

    # Step 3 — specific reasons from flags (used by SuperAgent)
    reasons = build_rejection_reasons(
        llm_eval=llm_eval,
        kw_metrics=kw_metrics
    )

    # Step 4 — rule-based acceptance decision
    accepted = score >= threshold

    rejection_reason = None
    if not accepted:
        rejection_reason = " ".join(reasons) if reasons else llm_eval.summary

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
        "reasons"                : reasons,          # specific per-flag reasons → SuperAgent
        "rejection_reason"       : rejection_reason, # combined string → rejection report
        "summary"                : llm_eval.summary, # LLM explanation → human readability
    }

