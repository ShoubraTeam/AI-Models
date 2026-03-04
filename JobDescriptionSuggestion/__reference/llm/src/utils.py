# ------------------------------------------------------------
# contains utility functions
# ------------------------------------------------------------

from groq import Groq
import src.config as CFG

def init_client(
    model_provider: str = 'groq',
):
    """
    Args:
        model_provider: str indicates the provider of the LLM. For ex (groq, ...)

    Returns:
        client: the API to use the model
    """

    if model_provider == 'groq':
        client = Groq()


    return client



def rephrase_with_groq(
    client,
    input: str,
    model_name: str,
    stream = False,
    **kwargs
):
    """
    Generating a response using a Groq Client

    Args:
        client          : the Groq Client
        input (str)     : input goes to the LLM
        model_name (str): the LLM name required
        stream (bool)   : whether to stream the output or not

    Returns:
        response (str): LLM Response
    """
    messages = [
        {"role" : "system", "content" : CFG.rephrasing_system_prompt},
        {"role" : "user", "content": input}
    ]


    completion = client.chat.completions.create(
        model = model_name,
        messages = messages,
        stream = stream,
        **kwargs
    )

    if not stream:
        response = completion.choices[0].message.content
    
    else:
        response = None

    return response



def is_complete_groq(client, input: str, model_name: str):
    """
    Determines whether the input contains skills or not.

    Args:
        client          : the Groq Client
        input (str)     : input goes to the LLM
        model_name (str): the LLM name required

    Returns:
        response (str): LLM Response
    """
    
    messages = [
        {'role' : 'system', 'content' : CFG.is_complete_system_prompt},
        {'role' : 'user', 'content' : input}
    ]

    completion = client.chat.completions.create(
        model = model_name,
        messages = messages,
        stream = False,
        max_tokens = 1
    )

    response = completion.choices[0].message.content
    return response