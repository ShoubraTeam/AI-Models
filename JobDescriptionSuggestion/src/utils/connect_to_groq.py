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
    
    if use_rag and retrieved_documents:
        context = "\n".join(retrieved_documents)
        query = f"{query}\n\nContext Information:\n{context}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=stream,
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
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **kwargs
    )

    return response.choices[0].message.content
