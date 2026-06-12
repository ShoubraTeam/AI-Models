# -----------------------------------------------
# Required workflow for job_desc_enhancement
# -----------------------------------------------
from groq import Groq
from agents.job_description_enhancement import detect_tools, extract_tools, enhance_job
from typing import Callable
from models.config.agents_config import JOB_DESCRIPTION_N_JOBS_TO_RETRIEVE, JOB_DESCRIPTION_RETRIEVAL_ALPHA
import ast
from weaviate.collections import Collection

from prompts import (
    TOOLS_DETECTION_PROMPT, 
    TOOLS_RECOMMENDATION_PROMPT, 
    JOB_DESCRIPTION_ENHANCEMENT_PROMPT_WITH_RAG, 
    JOB_DESCRIPTION_ENHANCEMENT_PROMPT_WITHOUT_RAG
)

from models.config.system_tasks import (
    JOB_DESC_TOOLS_DETECTION,
    JOB_DESC_TOOLS_RECOMMENDATION,
    JOB_DESC_JOB_DESCRIPTION_ENHANCEMENT
)


class JobDescriptionEnhancementPipeline:
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
        tools_retriever: Callable[[Collection, str, str, int, float], list] = None,
        collection: Collection = None
    ) -> None:
        """
        Args:
            agents         : the models used in tools-detection, tools-recommendation, and job-enhancement
            client         : Groq Client
            task           : str determines whether the required task is [tools_detection - tools_recommendation - job_desc_enhancement]
            tools_retriever: function used to retrieve relevant tools
        """
        self.tools_retriever   = tools_retriever
        self.tools_detector    = agents["tools_detector"]
        self.tools_recommender = agents["tools_recommender"]
        self.job_desc_enhancer = agents["job_desc_enhancer"]
        self.client            = client
        self.task              = task
        self.collection        = collection
    
    # ------------------------------- driver functions -----------------------------------
    def preprocess(self, input: str | tuple[str, str]):
        if self.task == JOB_DESC_TOOLS_DETECTION:
            return self.detect_tools_preprocess(input = input)

        elif self.task == JOB_DESC_TOOLS_RECOMMENDATION:
            return self.recommend_tools_preprocess(input = input)
        
        elif self.task == JOB_DESC_JOB_DESCRIPTION_ENHANCEMENT:
            return self.enhance_job_preprocess(input = input)

        else:
            pass

    
    def call(self, input: str):
        if self.task == JOB_DESC_TOOLS_DETECTION:
            return self.detect_tools_call(input = input)

        elif self.task == JOB_DESC_TOOLS_RECOMMENDATION:
            return self.recommend_tools_call(input = input)
        
        elif self.task == JOB_DESC_JOB_DESCRIPTION_ENHANCEMENT:
            return self.enhance_job_call(input = input)

        else:
            pass

    def postprocess(self, agent_output):
        if self.task == JOB_DESC_TOOLS_DETECTION:
            return self.detect_tools_postprocess(agent_output = agent_output)

        elif self.task == JOB_DESC_TOOLS_RECOMMENDATION:
            return self.recommend_tools_postprocess(agent_output = agent_output)
        
        elif self.task == JOB_DESC_JOB_DESCRIPTION_ENHANCEMENT:
            return self.enhance_job_postprocess(agent_output = agent_output)

        else:
            pass
    
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
        response = detect_tools(
            client = self.client,
            query = input,
            model_name = self.tools_detector,
            system_prompt = TOOLS_DETECTION_PROMPT,
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
        Retrieve Relevant Jobs& format them for extracting tools
        
        Args:
            (job_title, job_desc)
        """
        job_title, job_desc = input
        formatted_job = f"""
        ## Job Title: {job_title}
        ## Job Description:
        {job_desc}
        """
        retrieved = self.tools_retriever(
            self.collection,
            formatted_job, 
            None, 
            JOB_DESCRIPTION_N_JOBS_TO_RETRIEVE,
            JOB_DESCRIPTION_RETRIEVAL_ALPHA
        )


        formatted = ""
        for idx, doc in enumerate(retrieved, start = 1):
                job = f"""Job_#{idx}\n{doc}\n\n"""
                formatted += job


        return formatted.strip()
    


    def recommend_tools_call(self, input: str):
        tools = extract_tools(
            client = self.client,
            query = input,
            model_name = self.tools_recommender,
            system_prompt = TOOLS_RECOMMENDATION_PROMPT,
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

        system_prompt = JOB_DESCRIPTION_ENHANCEMENT_PROMPT_WITH_RAG if is_rag_used else JOB_DESCRIPTION_ENHANCEMENT_PROMPT_WITHOUT_RAG
            
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
