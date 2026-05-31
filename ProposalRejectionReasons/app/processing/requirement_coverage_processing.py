from helpers.config import NECESSITY_LEVEL_WEIGHTS
from schemas import FinalSubagentResult

REQUIREMENT_THRESHOLD = 0.5


def build_requirement_reasons(accepted: bool) -> list[str]:
    reasons = []
    if accepted:
        reasons.append("Mandatory requirements are fully covered.")
    else:
        reasons.append("Mandatory job requirements were missed.")
    return reasons


def calc_requirement_coverage_score(
    extracted_requirements, 
    final_coverage, 
    threshold: float = REQUIREMENT_THRESHOLD
) -> FinalSubagentResult:
    req_necessity_map = {
        req.id: req.necessity_level 
        for req in extracted_requirements
    }
    
    proposal_score = 0.0
    grd_truth = 0.0

    for req in extracted_requirements:
        if req.necessity_level == "forbidden":
            grd_truth += 1.0  
        else:
            grd_truth += NECESSITY_LEVEL_WEIGHTS[req.necessity_level]

    for req_id in final_coverage.requirements_covered_ids:
        necessity = req_necessity_map.get(req_id)
        if necessity:
            if necessity == "forbidden":
                proposal_score += 1.0  
            else:
                proposal_score += NECESSITY_LEVEL_WEIGHTS[necessity]

    for req_id in final_coverage.missing_requirements_ids:
        necessity = req_necessity_map.get(req_id)
        if necessity == "forbidden":
            proposal_score += NECESSITY_LEVEL_WEIGHTS["forbidden"]

    if grd_truth == 0:
        score = 0.0
    else:
        final_score = proposal_score / grd_truth
        score = max(0.0, round(final_score, 4))
        
    accepted = score >= threshold
    reasons = build_requirement_reasons(accepted)
    
    if accepted:
        acceptance_reasons = reasons
        rejection_reasons = None
    else:
        acceptance_reasons = None
        rejection_reasons = reasons

    summary_text = getattr(final_coverage, "summary", "Requirements coverage evaluation completed.")

    return FinalSubagentResult(
        score=score,
        accepted=accepted,
        summary=summary_text,
        acceptance_reasons=acceptance_reasons,
        rejection_reasons=rejection_reasons
    )