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
    year_info = f" (Year: {year})" if year else ""

    formatted_job = f"""
### Job Title: {job_title}{year_info}

### Original Job Description:
{job_desc}
"""
#-----------------------------------------------------------
#     --- TARGET JOB TO ENHANCE ---

# ### Job Title: Data Analyst (Year: 2024)

# ### Original Job Description:
# محتاجين حد شاطر في الاكسيل والباور بي آي عشان يحلل بيانات المبيعات ويطلع تقارير.

# ----------------------------------------------------------
    if use_rag:
        return f"--- TARGET JOB TO ENHANCE ---\n{formatted_job.strip()}\n----------------------------"
    
    return formatted_job.strip()
    """
    Returns:
        formatted_job (str): the job after formatting. It should be ready to feed it into the LLM.
    """



