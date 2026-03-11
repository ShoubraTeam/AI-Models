# -------------------------------------------------------------------
# Contains the required functions to Construct the System Prompts for
# - Suggesting Ehnanced Job Desc
# - Detecting if the the old job desc contains skills
# 
# Eng. Hanin
# -------------------------------------------------------------------


def get_detection_prompt() -> str:
    
    prompt = """You are an expert HR assistant. 
Your task is to analyze the provided job description and determine if it explicitly lists any technical or soft skills.
Respond ONLY with 'Yes' if it contains skills, and 'No' if it does not contain any skills. Do not provide any further explanation."""
    
    return prompt


def get_enhancement_prompt(use_rag: bool, retrieved_documents: list = None) -> str:
    """
    Constructing the System Prompt required for enhancing the job description.
    """
    if use_rag:
        prompt = """You are an expert HR copywriter. 
Your task is to professionally enhance, structure, and rewrite the provided job description. 
You will be provided with additional 'Context Information'. Please use this context to enrich the job description and make it more accurate and appealing."""
    else:
        prompt = """You are an expert HR copywriter. 
Your task is to professionally enhance, structure, and rewrite the provided job description to make it highly engaging, clear, and attractive to top talent."""
        
    return prompt
