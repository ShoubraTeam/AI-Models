# ---------------------------------------------------------------------------
# Contains the required functions to connect to
# - Connect to GROQ
# - Using GROQ models to enhance the the old job desc
# - Using GROQ models to detect if the old desc contains skills or not
# 
# Eng. Hanin
# ---------------------------------------------------------------------------



from groq import Groq


def get_groq_client():
    """
    Returns:
        client: the GROQ API required to use the model
    """
    client = Groq()
    return client



def enhance_old_job(
    client,
    query: str,
    model_name: str,
    system_prompt: str,
    use_rag: bool,
    retrieved_documents: list,
    stream = False,
    **kwargs
) -> str:
    """
    Args:
        client             : the Groq Client
        system_prompt (str): the job enhancement system prompt
        query (str)        : input query goes to the LLM
        model_name (str)   : the LLM name required
        use_rag (bool)     : whether to use RAG or not
        retrieved_documents (list): the list of the retrieved documents if RAG was used
        stream (bool)      : whether to stream the output or not
        kwargs             : keyword args (temperature | max-tokens | top-p | top-k | ...)

    Returns:
        response (str): LLM Response
    """
    pass



def has_skills(
    client,
    query: str,
    model_name: str,
    system_prompt: str,
    **kwargs
):
    """
    Args:
        client             : the Groq Client
        query (str)        : input query goes to the LLM
        model_name (str)   : the LLM name required
        system_prompt (str): the skill detection system prompt
        kwargs             : keyword arg (temperature | max-tokens | top-p | top-k | ...)

    Returns:
        response (str): LLM Response
    """
    
    pass