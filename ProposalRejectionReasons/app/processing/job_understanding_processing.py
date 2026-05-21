from schemas.job_understanding.job_understanding_schema import JobUnderstandingSchema
import helpers.config as CFG


def calc_job_understanding_result(evaluation: JobUnderstandingSchema) -> dict:
    """
    Apply rule-based decisions on top of the JobUnderstandingAgent output.

    Rules:
        score < JOB_UNDERSTANDING_THRESHOLD  -> rejection_reason is populated
        score >= JOB_UNDERSTANDING_THRESHOLD -> accepted, no rejection reason
    """
    threshold = CFG.JOB_UNDERSTANDING_THRESHOLD
    accepted  = evaluation.score >= threshold

    rejection_reason = None
    if not accepted:
        rejection_reason = (
            f"The proposal did not demonstrate sufficient understanding of the job. "
            f"Score: {evaluation.score:.1f}/10 (threshold: {threshold}). "
            f"{evaluation.summary}"
        )

    return {
        "score"            : evaluation.score,
        "confidence"       : evaluation.confidence_score,
        "accepted"         : accepted,
        "rejection_reason" : rejection_reason,
        "summary"          : evaluation.summary,
    }
