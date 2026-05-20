# -------------------------------------------------------------------------------
# Functions to pre-process | post-process data from the tools_alignment agents
# -------------------------------------------------------------------------------

from schemas import JobTool


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