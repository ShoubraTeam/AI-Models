# -----------------------------------------------
# Required workflow for job_desc_enhancement
# -----------------------------------------------
from typing import Any
from .pipeline import Pipeline
from groq import Groq
from agents.job_description_enhancement import detect_tools, extract_tools, enhance_job

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
        client: Groq
    ) -> None:
        """
        Args:
            agents : the models used in tools-detection, tools-recommendation, and job-enhancement

        """
        
        self.tools_detector    = agents["tools_detector"]
        self.tools_recommender = agents["tools_recommender"]
        self.job_desc_enhancer = agents["job_desc_enhancer"]
        self.client = client
    
    # ------------------------------- driver functions -----------------------------------
    def preprocess(self, input: tuple[str, str], task: str):
        if task == "detect_tools":
            return self.detect_tools_preprocess(input = input)

        elif task == "recommend_tools":
            return self.recommend_tools_preprocess(input = input)
        
        elif task == "enhance_job_description":
            return self.enhance_job_preprocess(input = input)

        else:
            pass

    
    def call(self, input, task: str):
        if task == "detect_tools":
            return self.detect_tools_call(input = input)

        elif task == "recommend_tools":
            return self.recommend_tools_call(input = input)
        
        elif task == "enhance_job_description":
            return self.enhance_job_call(input = input)

        else:
            pass

    def postprocess(self, input, task: str):
        if task == "detect_tools":
            return self.detect_tools_postprocess(input = input)

        elif task == "recommend_tools":
            return self.recommend_tools_postprocess(input = input)
        
        elif task == "enhance_job_description":
            return self.enhance_job_postprocess(input = input)

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
    # ------------------------------- detecting tools -----------------------------------
    def detect_tools_preprocess(self, input: tuple[str, str]) -> tuple[str, str]:
        return input
    
    def detect_tools_postprocess(self, input: str) -> str:
        return input
    
    def detect_tools_call(self, job_desc: str) -> bool:
        """
        Determine if the original job description contains tools or not

        Args:
            job_desc: the original job description
        """
        system_prompt = self.get_detection_prompt()
        response = detect_tools(
            client = self.client,
            query = job_desc,
            model_name = self.detection_model,
            system_prompt = system_prompt,
            temperature = 0,
            max_tokens = 1 
        )
        
        if response.lower() == 'yes' or response.lower() == "no":
            return "yes" in response.lower()
        else:
            return self.detect_tools(job_desc)
        
    # ------------------------------- detecting tools -----------------------------------
