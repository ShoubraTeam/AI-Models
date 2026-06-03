
import os
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
    def __init__(self, feature_id: str) -> None:
        # setup
        self.feature_id = feature_id
        self.agent_pipeline = self.get_feature_pipeline()


    def get_feature_pipeline(self) -> None:
        """Get the required feature pipeline"""
        if self.feature_id == FEATURE_IDENITY_RECOGNITION:
            return IdentityRecognitionPipeline
            
        elif self.feature_id == FEATURE_JOB_RECOMMENDATION_SYSTEM:
            return JobRecommendationSystemPipeline 
            
        elif self.feature_id == FEATURE_PROFILE_ANALYSIS:
            return ProfileAnalysisPipeline 
            
        elif self.feature_id == FEATURE_JOB_DESCRIPTION_ENHANCEMENT:
            return JobDescriptionEnhancementPipeline 
            
        else: 
            return ProposalRejectionReasonsPipeline 
            
        
        