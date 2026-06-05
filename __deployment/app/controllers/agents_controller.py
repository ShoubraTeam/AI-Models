
from models.data_config import (
    FEATURE_IDENITY_RECOGNITION,
    FEATURE_JOB_RECOMMENDATION_SYSTEM,
    FEATURE_PROFILE_ANALYSIS,
    FEATURE_JOB_DESCRIPTION_ENHANCEMENT,
)

from .agents_pipelines.identity_recognition        import IdentityRecognitionPipeline
from .agents_pipelines.job_description_enhancement import JobDescriptionEnhancementPipeline
from .agents_pipelines.job_recommendation_system   import JobRecommendationSystemPipeline
from .agents_pipelines.profile_analysis            import ProfileAnalysisPipeline
from .agents_pipelines.proposal_rejection_reasons  import ProposalRejectionReasonsPipeline

from typing import Any

class AgentController:
    """
    Orchestrate the logic of agents [LLM agents - trained models]. In paricular:
        - Pre-process input data 
        - Inferene using the agent
        - Post-process agent response 

    Args:
        Args:
        feature_id (str): feature identifier. Must be one of the following:
            - identity_recognition
            - job_recommendation_system

            - profile_analyzer
            - job_description_enhancement
            - proposal_rejection_reasons 
    """
    def __init__(self, feature_id: str, agents: Any) -> None:
        # setup
        self.feature_id = feature_id
        self.agent_pipeline = self.get_feature_pipeline(agents)


    def get_feature_pipeline(self, agents: Any) -> None:
        """Get the required feature pipeline"""
        if self.feature_id == FEATURE_IDENITY_RECOGNITION:
            return IdentityRecognitionPipeline(agents)
            
        elif self.feature_id == FEATURE_JOB_RECOMMENDATION_SYSTEM:
            return JobRecommendationSystemPipeline(agents) 
            
        elif self.feature_id == FEATURE_PROFILE_ANALYSIS:
            return ProfileAnalysisPipeline(agents) 
            
        elif self.feature_id == FEATURE_JOB_DESCRIPTION_ENHANCEMENT:
            return JobDescriptionEnhancementPipeline(agents) 
            
        else: 
            return ProposalRejectionReasonsPipeline(agents)
    

    def preprocess_input(self, input: Any):
        return self.agent_pipeline.preprocess(input = input)
    
    def call_agent(self, input: Any):
        return self.agent_pipeline.call(input = input)
    
    def postprocess_agent_output(self, agent_output: Any):
        return self.agent_pipeline.postprocess(agent_output = agent_output)
            
        
        