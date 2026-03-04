# ---------------------------------------------------------------------------
# Contains the required class to
# - Initiate the enhancer
# - enhancing the given job desc
# - detect the skills
# 
# Eng. Sara
# ---------------------------------------------------------------------------



class Enhancer:
    """
    General class for Job Description Enhancement / Suggestion

    Attbs:
        enhancement_model (str): the LLM used to re-structure the old description
        detection_model (str)  : the LLM used to detect the skills in the old description
        model_provider (str)   : str indicates the provider of the LLM. For ex (groq, ...)
    """
    def __init__(self, enhancement_model: str, detection_model: str, model_provider = 'groq'):
        pass

    def detect_skills(self, query: str) -> bool:
        """
        Args:
            query (str): input query

        Returns:
            result (bool): Whether if the job_desc contains skills (True) or not (False).
        """
        pass

    def enhance_old_desc(self, query: str, use_rag: bool, stream = False, **kwargs) -> str:
        """
        Args:
            query (str)    : input query
            use_rag (bool) : whether to use RAG or not
            stream (bool)  : whether to stream the output or not
            kwargs         : keyword args (temperature | max-tokens | top-p | top-k | ...)

        Returns:
            response (str): the LLM response
        """
        pass