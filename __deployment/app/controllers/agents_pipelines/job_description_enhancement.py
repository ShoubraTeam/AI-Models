# -----------------------------------------------
# Required workflow for job_desc_enhancement
# -----------------------------------------------
from typing import Any
from .pipeline import Pipeline
from groq import Groq
from agents.job_description_enhancement import detect_tools, extract_tools, enhance_job
from typing import Callable
from helpers.config import JOB_DESCRIPTION_N_JOBS_TO_RETREIVE, JOB_DESCRIPTION_RETREIVAL_ALPHA
import ast
from weaviate.collections import Collection
class JobDescriptionEnhancementPipeline(Pipeline):
    """
    Job Description Enhancement Pipeline

    Required Methods:
        preprocess(input)        : pre-process the input before calling the agent. If not pre-processing required -> return the input
        call(input)              : invoke/call the agent on the given input
        postprocess(agent_output): post-process the agent output. If no post-processing required -> return the agent_output.
    """
    def __init__(self, 
        agents,
        client: Groq,
        task  : str,
        tools_retreiver: Callable[[Collection, str, str, int, float], list] = None,
        collection: Collection = None
    ) -> None:
        """
        Args:
            agents         : the models used in tools-detection, tools-recommendation, and job-enhancement
            client         : Groq Client
            task           : str determines whether the required task is [tools_detection - tools_recommendation - job_desc_enhancement]
            tools_retreiver: function used to retreive relevant tools
        """
        self.tools_retreiver   = tools_retreiver
        self.tools_detector    = agents["tools_detector"]
        self.tools_recommender = agents["tools_recommender"]
        self.job_desc_enhancer = agents["job_desc_enhancer"]
        self.client            = client
        self.task              = task
        self.collection        = collection
    
    # ------------------------------- driver functions -----------------------------------
    def preprocess(self, input: str | tuple[str, str]):
        if self.task == "tools_detection":
            return self.detect_tools_preprocess(input = input)

        elif self.task == "tools_recommendation":
            return self.recommend_tools_preprocess(input = input)
        
        elif self.task == "job_description_enhancement":
            return self.enhance_job_preprocess(input = input)

        else:
            pass

    
    def call(self, input: str):
        if self.task == "tools_detection":
            return self.detect_tools_call(input = input)

        elif self.task == "tools_recommendation":
            return self.recommend_tools_call(input = input)
        
        elif self.task == "job_description_enhancement":
            return self.enhance_job_call(input = input)

        else:
            pass

    def postprocess(self, agent_output):
        if self.task == "tools_detection":
            return self.detect_tools_postprocess(agent_output = agent_output)

        elif self.task == "tools_recommendation":
            return self.recommend_tools_postprocess(agent_output = agent_output)
        
        elif self.task == "job_description_enhancement":
            return self.enhance_job_postprocess(agent_output = agent_output)

        else:
            pass
    
    # ------------------------------- utils ---------------------------------------------
    def get_detection_prompt(self) -> str:
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
    
    def get_tools_prompt(self) -> str:
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
    
    def get_enhancement_prompt(self, is_rag_used: bool) -> str:
        """
        Construct the System Prompt required for enhancing the job description.
        """

        if is_rag_used:
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
    # ------------------------------- detecting tools -----------------------------------
    def detect_tools_preprocess(self, input: tuple[str, str]) -> tuple[str, str]:
        return input
    
    def detect_tools_postprocess(self, agent_output: str) -> str:
        return agent_output
    
    def detect_tools_call(self, input: str) -> bool:
        """
        Determine if the given job description contains tools or not

        Args:
            job_desc: the original job description
        """
        system_prompt = self.get_detection_prompt()
        response = detect_tools(
            client = self.client,
            query = input,
            model_name = self.tools_detector,
            system_prompt = system_prompt,
            temperature = 0,
            max_tokens = 1 
        )
        
        if response.lower() == 'yes' or response.lower() == "no":
            return "yes" in response.lower()
        else:
            return self.detect_tools(input)
        
    # ------------------------------- tools recommendation -----------------------------------
    def recommend_tools_preprocess(self, input: tuple[str, str]) -> str:
        """
        Retreive Relevant Jobs& format them for extracting tools
        
        Args:
            (job_title, job_desc)
        """
        job_title, job_desc = input
        formatted_job = f"""
        ## Job Title: {job_title}
        ## Job Description:
        {job_desc}
        """
        retrieved = self.tools_retreiver(
            self.collection,
            formatted_job, 
            None, 
            JOB_DESCRIPTION_N_JOBS_TO_RETREIVE,
            JOB_DESCRIPTION_RETREIVAL_ALPHA
        )


        formatted = ""
        for idx, doc in enumerate(retrieved, start = 1):
                job = f"""Job_#{idx}\n{doc}\n\n"""
                formatted += job


        return formatted.strip()
    


    def recommend_tools_call(self, input: str):
        system_prompt = self.get_tools_prompt()
        tools = extract_tools(
            client = self.client,
            query = input,
            model_name = self.tools_recommender,
            system_prompt = system_prompt,
            temperature = 0.0
        )

        return tools
    
    def recommend_tools_postprocess(self, agent_output: str):
        try:
            tools = list(set(ast.literal_eval(agent_output)))[:10]
        except:
            raise

        return tools
    # --------------------------------------- Job Description Enhancement -----------------------------------
    def enhance_job_preprocess(self, input: tuple[str, str, list[str] | None]) -> str:
        job_title, job_desc, tools = input

        if tools is not None:
            tools = " - ".join(tools)
            formatted = f"""##Job Title: {job_title}\n\n## Job Description:\n{job_desc}\n\nTools: {tools}"""
                
        else:
            formatted = f"""## Job Title: {job_title}\n\n## Job Description:\n{job_desc}"""
                
        return formatted.strip()
    

    def enhance_job_call(self, input: tuple[str, bool]) -> str:
        agent_input = input[0]
        is_rag_used = input[1]

        system_prompt = self.get_enhancement_prompt(is_rag_used = is_rag_used)
            
        return enhance_job(
            client = self.client,
            query = agent_input,
            model_name = self.job_desc_enhancer,
            system_prompt = system_prompt,
            stream = False,
            temperature = 0.5,
        )
    
    def enhance_job_postprocess(self, agent_output: str) -> str:
        return agent_output