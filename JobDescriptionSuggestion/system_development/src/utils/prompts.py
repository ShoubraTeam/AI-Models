# -------------------------------------------------------------------
# Contains the required functions to Construct the System Prompts for
# - Suggesting Ehnanced Job Desc
# - Detecting if the the old job desc contains skills
# -------------------------------------------------------------------


def get_detection_prompt() -> str:
    
    prompt = """You are an expert HR assistant. 
Your task is to analyze the provided job description and determine if it explicitly contains specific tools/frameworks that is related to the job.
Respond ONLY with 'Yes' if it contains skills, and 'No' if it does not contain any skills. Do not provide any further explanation.
Examples
- Job Description: I seek for an experienced AI Engineer who can build a customer support chatbot.
  Response: No

- Job Description: I seek for an experienced AI Engineer who can build a customer support chatbot. He should be able to use Python and vector databases, and build RAG systems.
  Response: Yes

"""

    
    return prompt


def get_enhancement_prompt(use_rag: bool) -> str:
    """
    Construct the System Prompt required for enhancing the job description.
    """

    if use_rag:
        prompt = """
You are an expert job poster on a freelancing platform.

Your role is to professionally enhance, structure, and rewrite the provided job description
in order to clarify the project scope, expectations, and requirements.

You will also be provided with additional tools/frameworks retrieved from external sources.
Integrate these tools into the enhanced description naturally and professionally where appropriate.

Your response MUST strictly follow the structure below:

## Overview
- Provide a clear and concise enhanced overview of the project.

## Requirements
- List the required responsibilities & tasks needed to complete the project.

## Tools / Frameworks Required
- List the relevant tools, technologies, frameworks, or platforms required for the project.
"""

    else:
        prompt = """
You are an expert job poster on a freelancing platform.

Your role is to professionally enhance, structure, and rewrite the provided job description
in order to clarify the project scope, expectations, and requirements.

Your response MUST strictly follow the structure below:

## Overview
- Provide a clear and concise enhanced overview of the project.

## Requirements
- List the required responsibilities & tasks needed to complete the project.

## Tools / Frameworks Required
- List the relevant tools, technologies, frameworks, or platforms that mentioned in the given description.
"""


    return prompt.strip()



def get_tools_prompt() -> str:
    prompt = """You are an experienced text analyzer. You will be given a list descriptions about the same job or similar jobs. Your role is to extract only the tools/frameworks common in those descriptions.
Your output should be in the form of a list of tools/frameworks. In particular, you should output the following:
[
    'tool_1',
    'tool_2',
    'tool_3',
    .
    .
    .
    'tool_20'
]

Instructions
- Only extract the most 20 common tools. Do not give more than 20.
- In your response, only include the tools list. Do not add any other tokens to the list.
- Add the list braces '[' and ']' before and after the list.
- Add single quotes ' before and after each tool.
"""
    return prompt