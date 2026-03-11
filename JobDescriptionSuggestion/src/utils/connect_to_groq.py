# ---------------------------------------------------------------------------
# Contains the required functions to connect to
# - Connect to GROQ
# - Using GROQ models to enhance the the old job desc
# - Using GROQ models to detect if the old desc contains skills or not
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
    stream = False,
    **kwargs
) -> str:
    """
    Enhances an existing job description using an LLM. Optionally augments the
    query with retrieved documents (RAG) before sending it to the model.

    Args:
        client                    : LLM client instance used to call the chat completion API.
        query (str)               : The original job description or prompt to enhance.
        model_name (str)          : Name of the model used for generation.
        system_prompt (str)       : System instruction guiding the model's behavior.
        stream (bool, optional)   : If True, returns a streaming response. Defaults to False.
        **kwargs: Additional parameters passed directly to the model API
            (e.g., temperature, max_tokens).

    Returns:
        str | Stream: 
            - If stream = False → returns the generated text from the model.
            - If stream = True → returns the streaming response object.
    """ 

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model = model_name,
        messages = messages,
        stream = stream,
        **kwargs
    )

    if stream:
        return response
    else:
        return response.choices[0].message.content 


def has_skills(
    client,
    query: str,
    model_name: str,
    system_prompt: str,
    **kwargs
):
    """
    Determines whether a job description or query contains mentioned skills
    using an LLM.

    Args:
        client             : LLM client instance used to call the chat completion API.
        query (str)        : The job description or text to analyze.
        model_name (str)   : Name of the model used for analysis.
        system_prompt (str): System instruction guiding the model's behavior.
        **kwargs: Additional parameters passed directly to the model API
            (e.g., temperature, max_tokens).

    Returns:
        str: The model's response indicating whether skills are present
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model = model_name,
        messages = messages,
        **kwargs
    )

    return response.choices[0].message.content
