# -------------------------------------------------------------------
# Contains the required functions to Construct the System Prompts for
# - Suggesting Ehnanced Job Desc
# - Detecting if the the old job desc contains skills
# 
# Eng. Hanin
# -------------------------------------------------------------------


def get_detection_prompt() -> str:
    """
    Constructing the System Prompt required for detecting if the job description contains skills or not.

    Returns:
        prompt (str)
    """
    pass


def get_enhancement_prompt(use_rag: bool, retrieved_documents: list = None) -> str:
    """
    Constructing the System Prompt required for enhancing the job description.

    Args:
        use_rag             (bool): whether to use RAG or not
        retrieved_documents (list): the list of the retrieved documents if RAG was used

    Returns:
        prompt (str)
    """
    if use_rag:
        pass
    else:
        pass