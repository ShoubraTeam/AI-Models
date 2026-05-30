# -------------------------------------------------------------------------------
# Functions to pre-process | post-process data from the tools_alignment agents
# -------------------------------------------------------------------------------

from schemas import JobTool, ProposalToolsResponse
from helpers.config import NECESSITY_LEVEL_WEIGHTS, WITH_CONFIDENCE_TOOL_WEIGHT, GENERIC_TOOL_WEIGHT

def format_ip_for_proposal_tools_analyzer(
    job_tools: list[JobTool],
    proposal: str
):
    """
    Preparing data for the agent that analyzes proposal tools

    Args:
        job_tools (list): the list of the tools extracted from the job_description
        proposal  (str) : the proposal text

    return:
        formatted (str): the prepared input formatted as pure string
    """

    formatted = "Job_Tools_List:\n"

    # add tools
    for idx, tool in enumerate(job_tools, start = 1):
        tool_text = f"Tool {idx} => name: {tool.tool_name}, necessity_level: {tool.necessity_level}\n"
        formatted += tool_text

    # add proposal
    formatted += f"\nProposal:\n{proposal}"
    
    return formatted


def calc_tools_alignment_score(
    proposal_tools_response: ProposalToolsResponse
):
    """
    Calculates the score of the tools mentioned in the proposal W.R.T to the job tools.

    Args:
        proposal_tools_response: the response from proposal-tools-analyzer
    
    Return:
        score (float)
    """
    proposal_score = 0.0
    grd_truth = 0.0

    for tool_review in proposal_tools_response.tool_reviews:
        if tool_review.found_in_proposal:
            necessity_level_weight = NECESSITY_LEVEL_WEIGHTS[tool_review.necessity_level]
            with_confidence_weight = WITH_CONFIDENCE_TOOL_WEIGHT if tool_review.with_confidence else GENERIC_TOOL_WEIGHT
            
            proposal_score += 1.0 * necessity_level_weight * with_confidence_weight
            
        grd_truth += 1.0
    
    return proposal_score / grd_truth
