import asyncio
from typing import Any
from pathlib import Path
from typing import TypeAlias

from agents import NumericalAnalyzer, BioAnalyzer, SkillsAnalyzer, VisualBrandEvaluator, SuperAgent
from models.pydantic_schemas import SuperAgentSchema

from .pipeline_errors import NumericalAnalyzerError, BioAnalyzerError, ProfileScorerError, SkillsAnalyzerError, VisualBrandEvaluatorError, ProfileSuperAgentError


from models.data_config import (
    PROFILE_SCORER_FEATURES_EXTRACTION,
    PROFILE_SCORER_FINAL_ANALYSIS,
)

ProfileScorer_Type: TypeAlias = (
    NumericalAnalyzer    |
    BioAnalyzer          |
    SkillsAnalyzer       |
    VisualBrandEvaluator |
    SuperAgent
)

class ProfileAnalysisPipeline:
    """
    This pipeline abstracts the Profile Scorer system by:
        - Accepting freelancer profile data and image
        - Running profile text & numerical features extraction
        - Preparing sub-audits for the Master SuperAgent
        - Synthesizing the final Profile Audit Report with actionable recommendations
    """
    # --------------------- Setup -----------------------
    def __init__(self, agents: dict[str, ProfileScorer_Type], task: str):
        self.numerical_analyzer   = agents["numerical_analyzer"]
        self.bio_analyzer         = agents["bio_analyzer"]
        self.skills_analyzer       = agents["skills_analyzer"]
        self.visual_brand_evaluator = agents["visual_brand_evaluator"]
        self.super_agent          = agents["profile_super_agent"]
        
        self.task = task
        
        self.cached_profile_data = None
        self.extracted_profile_features = None

    # ------------------------------- Driver Functions (Standard Compliant) -----------------------------------
    def preprocess(self, input: Any):
        if self.task == PROFILE_SCORER_FEATURES_EXTRACTION:
            return self.profile_features_extraction_preprocess(input = input)
        elif self.task == PROFILE_SCORER_FINAL_ANALYSIS:
            return self.profile_final_analysis_preprocess(input = input)

    def call(self, input: Any):
        if self.task == PROFILE_SCORER_FEATURES_EXTRACTION:
            return self.profile_features_extraction_call(profile_data = input)
        elif self.task == PROFILE_SCORER_FINAL_ANALYSIS:
            return self.profile_final_analysis_call(input = input)

    def postprocess(self, agent_output: Any):
        if self.task == PROFILE_SCORER_FEATURES_EXTRACTION:
            return self.profile_features_extraction_postprocess(agent_output = agent_output)
        elif self.task == PROFILE_SCORER_FINAL_ANALYSIS:
            return self.profile_final_analysis_postprocess(agent_output = agent_output)

    # ---------------------- Stage 1: Profile Features Extraction ----------------------
    def profile_features_extraction_preprocess(self, input: dict) -> dict:
        return input

    async def profile_features_extraction_call(self, profile_data: dict) -> dict:
        """
        Runs Textual and Numerical Sub-Agents in parallel to extract baseline profile features.
        """
        job_role_input = str(profile_data.get("job_role", ""))
        bio_input = str(profile_data.get("bio_text", ""))
        skills_input = list(profile_data.get("declared_skills", []))

        tasks = {
            "numerical_res": asyncio.to_thread(
                self.numerical_analyzer.invoke,
                job_role=job_role_input,
                hourly_rate=float(profile_data.get("hourly_rate", 0.0)),
                rating=float(profile_data.get("rating", 0.0)),
                total_completed_jobs=int(profile_data.get("total_completed_jobs", 0))
            ),
            "bio_res": asyncio.to_thread(
                self.bio_analyzer.invoke, 
                bio_text=bio_input, 
                job_role=job_role_input
            ),
            "skills_res": asyncio.to_thread(
                self.skills_analyzer.invoke, 
                declared_skills=skills_input, 
                job_role=job_role_input
            )
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        profile_features = dict(zip(tasks.keys(), results))
        
        error_mapping = {
            "numerical_res": NumericalAnalyzerError,
            "bio_res": BioAnalyzerError,
            "skills_res": SkillsAnalyzerError
        }
        for k, v in profile_features.items():
            if isinstance(v, Exception):
                error_class = error_mapping.get(k, ProfileScorerError)
                profile_features[k] = error_class(f"{error_class.default_message}: {str(v)}")
            elif v is None:
                profile_features[k] = {"score": 0, "details": "Agent returned empty response"}

        return profile_features

    def profile_features_extraction_postprocess(self, agent_output: dict) -> dict:
        return agent_output

    # ---------------------- Stage 2: Final Analysis & Super Agent ----------------------
    async def profile_final_analysis_preprocess(self, input: tuple[dict, str, dict]) -> dict:
        """
        Combines Stage 1 features with the Live Visual Brand Audit.
        """
        profile_data, img_path, pre_extracted_features = input
        
        try:
            visual_res = await asyncio.to_thread(
                self.visual_brand_evaluator.invoke,
                image_path=img_path, 
                job_role=profile_data.get("job_role", "")
            )
        except Exception as e:
            visual_res = VisualBrandEvaluatorError(f"Visual Brand Evaluator Error: {str(e)}")

        all_sub_audits = {
            **pre_extracted_features,
            "visual_res": visual_res
        }
        return all_sub_audits

    async def profile_final_analysis_call(self, input: tuple[dict, dict]) -> SuperAgentSchema:
        """
        Invokes the Master SuperAgent Orchestrator to compile the final report.
        """
        sample, all_sub_audits = input  
        
        return await asyncio.to_thread(
            self.super_agent.invoke,
            visual_res=all_sub_audits.get("visual_res"),
            bio_res=all_sub_audits.get("bio_res"),
            skills_res=all_sub_audits.get("skills_res"),
            numerical_res=all_sub_audits.get("numerical_res")
        )
    
    def profile_final_analysis_postprocess(self, agent_output: SuperAgentSchema) -> str:
        """
        Converts the structured SuperAgent Schema output into a highly polished, 
        scannable Markdown report for the freelancer dashboard.
        """
        def format_bullet_points(items: list[str] | None) -> str:
            if not items:
                return "None identified."
            return "\n".join(f"- {str(item).strip()}" for item in items if str(item).strip())

        report_sections = [
            "Freelancer Profile Executive Audit Report",
            f"Overall Profile Score: {agent_output.overall_score}/1.0",
            f"Executive Summary\n{agent_output.executive_summary}",
            f"Critical Weaknesses & Gaps\n{format_bullet_points(agent_output.critical_weaknesses)}",
            f"Prioritized Action Plan\n{format_bullet_points(agent_output.prioritized_action_plan)}"
        ]
        
        return "\n\n".join(report_sections)