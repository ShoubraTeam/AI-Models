from helpers.config import NECESSITY_LEVEL_WEIGHTS
from schemas import FinalSubagentResult
import json

REQUIREMENT_THRESHOLD = 0.5


# -------------------------------------- Pre-Processing --------------------------------------- #

def prepare_job_requirements_matcher_ip(job_requirements: list, proposal_text: str) -> str:
    mapped_requirements = []
    for req in job_requirements:
        if isinstance(req, dict):
            requirement = {
                "id": str(req.get("id", "")),
                "text": req.get("description", req.get("text", "")),
                "necessity_level": req.get("necessity_level", "mandatory"),
            }
        else:
            requirement = {
                "id": str(req.id),
                "text": req.text,
                "necessity_level": req.necessity_level,
            }

        mapped_requirements.append(requirement)
    
    requirements_json = json.dumps(mapped_requirements, indent=2)
    return f"Job Requirements List:\n{requirements_json}\n\nFreelancer Proposal Text:\n{proposal_text}"


# -------------------------------------- Post-Processing --------------------------------------- #

def build_requirement_reasons(accepted: bool) -> list[str]:
    reasons = []
    if accepted:
        reasons.append("Mandatory requirements are fully covered.")
    else:
        reasons.append("Mandatory job requirements were missed.")
    return reasons


def get_final_requirements_coverage_result(
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
                proposal_score += NECESSITY_LEVEL_WEIGHTS["forbidden"]
            else:
                proposal_score += NECESSITY_LEVEL_WEIGHTS[necessity]

    for req_id in final_coverage.missing_requirements_ids:
        necessity = req_necessity_map.get(req_id)
        if necessity == "forbidden":
            proposal_score += 1.0

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