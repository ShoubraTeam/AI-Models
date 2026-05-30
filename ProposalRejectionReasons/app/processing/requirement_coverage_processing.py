from helpers.config import NECESSITY_LEVEL_WEIGHTS

def calc_requirement_coverage_score(extracted_requirements, final_coverage) -> float:
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
        return 0.0
        
    final_score = proposal_score / grd_truth
    return max(0.0, round(final_score, 4))