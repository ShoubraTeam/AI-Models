# ------------------------------------------------------------
# contains the logic required for rephrasing the user input
# ------------------------------------------------------------


from src.utils import init_client, rephrase_with_groq, is_complete_groq

class LLMRephraser:
    """
    A General class for using an LLM to rephrase the user input

    Attbs:
        model_name    : str indicates the LLM name to use
        model_provider: str indicates the provider of the LLM. For ex (groq, ...)
    """

    def __init__(self, model_name: str, model_provider: str = 'groq'):
        self.rephraser_model = model_name
        self.checker_model = model_name
        self.model_provider = model_provider
        self.client = init_client(model_provider = model_provider)


    # ----------------------------------------------------------------------------
    def rephrase(self, input: str, stream = False, **kwargs):
        """
        Rephrasing the user's input
        """
        if self.model_provider == 'groq':
            response = rephrase_with_groq(
                client = self.client,
                input = input,
                model_name = self.rephraser_model,
                stream = stream,
                **kwargs
            )

        return response
    # ----------------------------------------------------------------------------
    def is_complete(self, input: str):
        """
        Determines whether the input contains skills or not.
        """

        response = is_complete_groq(
            client = self.client,
            model_name = self.checker_model,
            input = input
        )

        response = response.strip().lower()

        if response == "no":
            return False
        elif response == "yes":
            return True
        else:
            return "cannot-determine"
    # ----------------------------------------------------------------------------
    def format_job(self, job_title: str, job_description: str, experience_required: str = None, skills = None):
        """
        Formatting the given job

        Returns:
            formatted_input (str): the formatted input
            is_complete: if the input mentioned the skills or not
        """
        # experience part
        exp_sentence = ""
        if experience_required is not None:
            exp_sentence = f"Experience Level Required: {experience_required}."

        
        # skills part
        skills_sentence = ""
        if skills is not None:
            if isinstance(skills, list):
                skills = ", ".join(skills)
            skills_sentence = f"Required Skills: {skills}"
            is_complete = True
        else:
            is_complete = self.is_complete(input = job_description)

        # format
        formatted_input = f"""
Job Title: {job_title}

Job Description: {job_description}

{skills_sentence}

{exp_sentence}
"""
        return formatted_input, is_complete
