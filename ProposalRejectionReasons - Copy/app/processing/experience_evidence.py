from schemas.experience_evidence import ExperienceEvidenceSchema
from schemas import FinalSubagentResult



EXPERIENCE_EVIDENCE_THRESHOLD = 0.5



# ---------------------------------------------------- Pre-Processing -------------------------------------------- #
def prepare_experience_evidence_evaluator_ip(job_desc: str, proposal_text: str) -> str:
    formatted = f"""##Job Description:\n{job_desc}\n\n##Proposal:\n{proposal_text}"""
    return formatted


# ---------------------------------------------------- Post-Processing -------------------------------------------- #
def calc_experience_score(
    has_experience_evidence: bool,
    extracted_projects     : list
) -> float:
    if not has_experience_evidence or not extracted_projects:
        return 0.0

    scores = [project.relevance_score for project in extracted_projects]
    max_score = max(scores) if scores else 0.0

    return round(max_score, 2)


def build_experience_rejection_reasons(
    has_experience_evidence: bool,
    score: float,
    threshold: float
) -> list[str]:
    reasons = []

    if not has_experience_evidence:
        reasons.append("No concrete project experience mentioned.")
    elif score < threshold:
        reasons.append("Past projects have low technical relevance.")

    return reasons


def build_experience_acceptance_reasons(score: float) -> list[str]:
    reasons = []
    reasons.append("Highly relevant past project provided.")
    if score >= 0.8:
        reasons.append("Project relevance score meets threshold.")
    return reasons


def get_final_experience_evidence_result(
    llm_audit: ExperienceEvidenceSchema,
    threshold: float = EXPERIENCE_EVIDENCE_THRESHOLD
) -> FinalSubagentResult:
    score = calc_experience_score(
        has_experience_evidence=llm_audit.has_experience_evidence,
        extracted_projects=llm_audit.extracted_projects
    )

    reasons = build_experience_rejection_reasons(
        has_experience_evidence=llm_audit.has_experience_evidence,
        score=score,
        threshold=threshold
    )

    accepted = score >= threshold

    if accepted:
        acceptance_reasons = build_experience_acceptance_reasons(score=score)
        rejection_reasons = None
    else:
        acceptance_reasons = None
        rejection_reasons = reasons

    return FinalSubagentResult(
        score=score,
        accepted=accepted,
        summary=llm_audit.summary, 
        acceptance_reasons=acceptance_reasons,
        rejection_reasons=rejection_reasons
    )