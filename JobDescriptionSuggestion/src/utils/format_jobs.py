# ---------------------------------------------------------------------------
# Contains the required functions to format the the Job goes to the LLM
# either if RAG was used or not
# 
# Eng. Sara
# ---------------------------------------------------------------------------


def format_job(
    job_title: str,
    job_desc: str,
    use_rag: bool,
    year: int = None,
) -> str:
    """
    Returns:
        formatted_job (str): the job after formatting. It should be ready to feed it into the LLM.
    """
    pass