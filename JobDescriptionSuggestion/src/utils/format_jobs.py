# ---------------------------------------------------------------------------
# Contains the required functions to format the the Job goes to the LLM
# either if RAG was used or not
# ---------------------------------------------------------------------------
import ast
def format_job(
    job_info: dict,
    use_rag: bool,
    retrieved_documents :list = None,

) -> str:
    """
    Formats a job title and description into a structured prompt that can be
    sent to an LLM. Optionally appends retrieved context (RAG) to enrich the input.

    Args:
        job_info                            : dictionary contains the job info (title - desc - skills - categories - year)
        use_rag (bool)                      : Whether retrieved context should be included.
        retrieved_documents (list, optional): List of retrieved text documents used as additional context for the model. Defaults to None.

    Returns:
        formatted_job (str): The job after formatting. It should be ready to feed into the LLM.
    """
    job_title = job_info.get("title", None)
    job_desc = job_info.get("description", None)
    job_skills = job_info.get("skills", None)
    job_categories = job_info.get("categories", None)
    job_year = job_info.get("year", None)

    # format skills
    if job_skills is None:
        skills_sentence = ""
    else:
        skills_str = ", ".join(ast.literal_eval(job_skills))
        skills_sentence = f"""
        
Recommended Skills:
{skills_str}
"""
        
    # format categories
    if job_categories is None:
        cat_sentence = ""
    else:
        cat_str = ", ".join(ast.literal_eval(job_categories))
        cat_sentence = f"""
        
The job belong to these categories:
{cat_str}
""" 
        
    
    # format categories
    year_info = f" (Year: {job_year})" if job_year else ""


    # formatting
    formatted_job = f"""
- Ehance this job description:
### Job Title: {job_title}{year_info}

### Original Job Description:
{job_desc}{cat_sentence}{skills_sentence}
"""
    # RAG
    if use_rag and retrieved_documents:
        context = ""
        for idx, doc in enumerate(retrieved_documents, start = 1):
            job = f"Job #{idx}\n{doc}"
            context += job
            context += "\n\n"

        sep = "*" * 50
        formatted_job = f"""{formatted_job}\n\n{sep}\n\n
Add this additional jobs info (Context) to your knowlege base:

{context}
"""

    return formatted_job.strip()
    



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
        formatted_job (str): The job after formatting. It should be ready to feed into the LLM.
    """


    formatted_job = f"""
### Job Title: {job_title}

### Job Description:
{job_desc}
"""
    return formatted_job.strip()