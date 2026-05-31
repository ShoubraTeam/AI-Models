# -------------------------------------------------------------------------------
# Functions to pre-process | post-process data from the tools_alignment agents
# -------------------------------------------------------------------------------

from schemas import JobTool, ProposalToolsResponse
from helpers.config import NECESSITY_LEVEL_WEIGHTS, WITH_CONFIDENCE_TOOL_WEIGHT, GENERIC_TOOL_WEIGHT
from typing import Any
# ------------------------------------- Pre-Processing ---------------------------------------------

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


# ------------------------------------- Post-Processing ---------------------------------------------

def calc_tools_alignment_score(
    proposal_tools_response: ProposalToolsResponse
) -> float | None:
    """
    Calculates how well the proposal tools align with the job-required tools.

    Args:
        proposal_tools_response: response from proposal-tools-analyzer

    Returns:
        float: normalized tools alignment score or None if len(job_tools) == 0
    """

    proposal_score = 0.0
    ground_truth_score = 0.0

    for tool_review in proposal_tools_response.tool_reviews:
        necessity_level_weight = NECESSITY_LEVEL_WEIGHTS[tool_review.necessity_level]

        ground_truth_score += necessity_level_weight

        if not tool_review.found_in_proposal:
            continue

        confidence_weight = WITH_CONFIDENCE_TOOL_WEIGHT if tool_review.with_confidence else GENERIC_TOOL_WEIGHT
        proposal_score += necessity_level_weight * confidence_weight

    if ground_truth_score == 0:
        return None

    return proposal_score / ground_truth_score



def get_final_tool_alignment_result(
    proposal_tools_response: ProposalToolsResponse,

) -> dict[str, Any]:
    """
    Get the final tool alignmnt result. This result will be passed to the Super-Agent.
    """