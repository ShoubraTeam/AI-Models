# ---------------------------------------------------------------------------
# Contains the required functions to format the the Job goes to the LLM
# either if RAG was used or not
# ---------------------------------------------------------------------------
import ast
def format_for_enhancement(
    job_disc: str,
    tools: list = None,
) -> str:
    """
    Formats a job title, description, retrieved tools (optional) into a structured prompt that can be
    sent to an LLM.
    """

    if tools:
        tools = " - ".join(tools)
        formatted = f"""## Job Description:
{job_disc}

Tools: {tools}
"""
        
    else:
        formatted = f"""## Job Description:
{job_disc}
"""
        
    return formatted.strip()


    
    


def format_for_retriever(
    job_title: str,
    job_desc: str,
) -> str:
    """
    Formats a job title and description into a structured prompt that can be fed to a retrieval system

    Args:
        job_title (str)                     : The title of the job position.
        job_desc (str)                      : The original job description text.

    Returns:
        formatted_job (str): The job after formatting. It should be ready to feed to the retriever.
    """


    formatted_job = f"""
## Job Title: {job_title}
## Job Description:
{job_desc}
"""
    return formatted_job.strip()


def format_retrieved_docs(documents: list):
    """
    Format the retrieved documents to a format ready to tools extraction
    """
    formatted = ""
    for idx, doc in enumerate(documents, start = 1):
        job = f"""Job_#{idx}
{doc}


"""
        formatted += job
    
    return formatted