
from models.data_config import (
    FEATURE_IDENITY_RECOGNITION,
    FEATURE_JOB_RECOMMENDATION_SYSTEM,
    FEATURE_PROFILE_ANALYSIS,
    FEATURE_JOB_DESCRIPTION_ENHANCEMENT,
)

from typing import Any

class AgentController:
    """
    Orchestrate the logic of agents [LLM agents - trained models]. In paricular:
        - Pre-process input data 
        - Inferene using the agent
        - Post-process agent response 

    Args:
        feature_id (str): feature identifier. Must be one of the following:
            - identity_recognition
            - job_recommendation_system
            - profile_analyzer
            - job_description_enhancement
            - proposal_rejection_reasons 
        
        agents: agents used in features
        kwargs: key word arguments specific to each feature if required
    """
    def __init__(self, feature_id: str, agents: Any, client: Any = None, **kwargs) -> None:
        # setup
        self.feature_id = feature_id
        self.agent_pipeline = self.get_feature_pipeline(agents, client = client, **kwargs)


    def get_feature_pipeline(self, agents: Any, client: Any, **kwargs) -> None:
        """Get the required feature pipeline"""
        
        if self.feature_id == FEATURE_IDENITY_RECOGNITION:
            from .agents_pipelines.identity_recognition import IdentityRecognitionPipeline

            return IdentityRecognitionPipeline(agents)
            
        elif self.feature_id == FEATURE_JOB_RECOMMENDATION_SYSTEM:
            from .agents_pipelines.job_recommendation_system import JobRecommendationSystemPipeline

            return JobRecommendationSystemPipeline(agents, task=kwargs.get("task"))
            
        elif self.feature_id == FEATURE_PROFILE_ANALYSIS:
            from .agents_pipelines.profile_analysis import ProfileAnalysisPipeline

            return JobRecommendationSystemPipeline(agents, task=kwargs.get("task"))
            
        elif self.feature_id == FEATURE_JOB_DESCRIPTION_ENHANCEMENT:
            from .agents_pipelines.job_description_enhancement import JobDescriptionEnhancementPipeline

            return JobDescriptionEnhancementPipeline(agents, client = client, **kwargs) 
            
        else: 
            from .agents_pipelines.proposal_rejection_reasons import ProposalsRejectionReasonsPipeline

            return ProposalsRejectionReasonsPipeline(agents, **kwargs)
    

    def preprocess_input(self, input: Any):
        return self.agent_pipeline.preprocess(input = input)
    
    def call_agent(self, input: Any):
        return self.agent_pipeline.call(input = input)
    
    def postprocess_agent_output(self, agent_output: Any):
        return self.agent_pipeline.postprocess(agent_output = agent_output)
            
        
        
