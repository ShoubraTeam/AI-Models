# -------------------------------------------------------------------------------
# Functions to pre-process | post-process data from the tools_alignment agents
# -------------------------------------------------------------------------------

from schemas import JobTool, ProposalToolsResponse, FinalSubagentResult, ProposalToolReview
from helpers.config import NECESSITY_LEVEL_WEIGHTS, WITH_CONFIDENCE_TOOL_WEIGHT, GENERIC_TOOL_WEIGHT
from helpers.config import TOOL_ALIGNMENT_ACCEPTANCE_THRESHOLD
from typing import Any

MAX_REASON_LENGTH = 100


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))


def fit_reason(reason: str) -> str:
    if len(reason) <= MAX_REASON_LENGTH:
        return reason

    return reason[:MAX_REASON_LENGTH - 3].rstrip() + "..."
# ------------------------------------- Pre-Processing ---------------------------------------------

def prepare_proposal_tools_analyzer_ip(
    job_tools: list[JobTool],
    proposal: str
) -> str:
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
    tool_reviews: list[ProposalToolReview]
) -> float | None:
    """
    Calculates how well the proposal tools align with the job-required tools.

    Args:
        tool_reviews: the list of tool reviews returned by the analyzer sub-agent

    Returns:
        float: normalized tools alignment score or None if len(job_tools) == 0
    """

    proposal_score = 0.0
    max_possible_score = 0.0

    for tool_review in tool_reviews:
        if tool_review.necessity_level == "forbidden":
            necessity_level_weight = 1.0
            max_possible_score += necessity_level_weight

            if not tool_review.found_in_proposal:
                proposal_score += necessity_level_weight

            continue

        necessity_level_weight = NECESSITY_LEVEL_WEIGHTS[tool_review.necessity_level]
        max_possible_score += necessity_level_weight

        if tool_review.found_in_proposal:
            confidence_weight = WITH_CONFIDENCE_TOOL_WEIGHT if tool_review.with_confidence else GENERIC_TOOL_WEIGHT
            proposal_score += necessity_level_weight * confidence_weight

    if max_possible_score == 0:
        return None

    return clamp_score(proposal_score / max_possible_score)



def get_mentioned_tools(
    tool_reviews: list[ProposalToolReview]
) -> tuple[int, int, str, str]:
    """
    Return:
        - number of total job tools
        - number of tools mentioned in the proposal
        - tools mentioned with confidence as a str
        - tools mentioned without confidence as a str
    """
    total_tools = len(tool_reviews)
    mentioned_tools_with_confidence = []
    mentioned_tools_generally = []
    for tool_review in tool_reviews:
        if not tool_review.found_in_proposal:
            continue

        if tool_review.with_confidence == True:
            mentioned_tools_with_confidence.append(tool_review.tool_name)
        else:
            mentioned_tools_generally.append(tool_review.tool_name)
    
    num_of_mentioned_tools = len(mentioned_tools_with_confidence) + len(mentioned_tools_generally)

    return (
        total_tools,
        num_of_mentioned_tools,
        ", ".join(mentioned_tools_with_confidence),
        ", ".join(mentioned_tools_generally),
    )


def get_acceptance_reasons(
    tool_reviews: list[ProposalToolReview],
    score       : float,
    threshold   : float = TOOL_ALIGNMENT_ACCEPTANCE_THRESHOLD
) -> list[str]:
    """
    Return acceptance reasons based on sub-agent results
    """
    reasons = ["Acceptance Reasons:"]

    # score
    reasons.append(f"- Score ({score}) is bigger than the acceptance threshold ({threshold})")

    # tools
    total_tools, num_of_mentioned_tools, mentioned_tools_with_confidence, mentioned_tools_generally = get_mentioned_tools(
        tool_reviews = tool_reviews
    ) 

    reasons.append(f"- Proposal mentioned {num_of_mentioned_tools} of {total_tools} required tools.")
    reasons.append(f"- Confident tool mentions: {mentioned_tools_with_confidence or 'None'}")
    reasons.append(f"- Generic tool mentions: {mentioned_tools_generally or 'None'}")
    
    return [fit_reason(reason) for reason in reasons]
    

def get_rejection_reasons(
    tool_reviews: list[ProposalToolReview],
    score       : float,
    threshold   : float = TOOL_ALIGNMENT_ACCEPTANCE_THRESHOLD
) -> list[str]:
    """
    Return rejections reasons based on sub-agent results
    """
    reasons = ["Rejection Reasons:"]

    # score
    reasons.append(f"- Score ({score}) is less than the acceptance threshold ({threshold})")

    # tools
    total_tools, num_of_mentioned_tools, mentioned_tools_with_confidence, mentioned_tools_generally = get_mentioned_tools(
        tool_reviews = tool_reviews
    ) 

    reasons.append(f"- Proposal mentioned only {num_of_mentioned_tools} of {total_tools} required tools.")
    reasons.append(f"- Confident tool mentions: {mentioned_tools_with_confidence or 'None'}")
    reasons.append(f"- Generic tool mentions: {mentioned_tools_generally or 'None'}")
    
    return [fit_reason(reason) for reason in reasons]


def get_final_tool_alignment_result(
    proposal_tools_response: ProposalToolsResponse,
    threshold: float = TOOL_ALIGNMENT_ACCEPTANCE_THRESHOLD
) -> FinalSubagentResult:
    """
    Get the final tool alignmnt result. This result will be passed to the Super-Agent.

    Args:
        proposal_tools_response: the analysis of the proposal tools returned by the sub-agent

    Returns:
        FinalSubagentResult object: the final result of the sub-agent after post-processings contains:
    """
    # calc score & acceptance
    tool_alignment_score = calc_tools_alignment_score(proposal_tools_response.tool_reviews)
    if tool_alignment_score is None:
        tool_alignment_score = 0.0

    accepted = tool_alignment_score >= threshold

    # summary
    summary = proposal_tools_response.summary

    # reasons
    acceptance_reasons = None
    rejection_reasons  = None

    if accepted:
        acceptance_reasons = get_acceptance_reasons(
            tool_reviews = proposal_tools_response.tool_reviews,
            score        = tool_alignment_score,
            threshold    = threshold
        )

    else:
        rejection_reasons = get_rejection_reasons(
            tool_reviews = proposal_tools_response.tool_reviews,
            score        = tool_alignment_score,
            threshold    = threshold
        )


    return FinalSubagentResult(
        score              = tool_alignment_score,
        accepted           = accepted,
        summary            = summary,
        acceptance_reasons = acceptance_reasons,
        rejection_reasons  = rejection_reasons
    )
