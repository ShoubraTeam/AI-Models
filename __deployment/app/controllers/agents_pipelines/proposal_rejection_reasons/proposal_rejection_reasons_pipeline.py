# -----------------------------------------
from __future__ import annotations
# Building the system pipeline
# -----------------------------------------

import asyncio
from typing import Any

# sub-agents
from agents.proposal_rejection_reasons.tools_alignment      import JobToolsExtractor, ProposalToolsAnalyzer
from agents.proposal_rejection_reasons.experience_evidence  import ExperienceEvidenceAgent
from agents.proposal_rejection_reasons.language_clarity     import LanguageClarityEvaluator
from agents.proposal_rejection_reasons.job_understanding    import JobUnderstandingEvaluator, JobKeyPointsExtractor
from agents.proposal_rejection_reasons.requirement_coverage import JobRequirementsExtractor, JobRequirementsMatcher
from agents.proposal_rejection_reasons.super_agent import ProposalRejectionSuperAgent
from models.pydantic_schemas import SuperAgentResponse, FinalSubagentResult


ProposalRejectionReasons_Type = (
    JobToolsExtractor           |
    ProposalToolsAnalyzer       |
    JobKeyPointsExtractor       |
    JobUnderstandingEvaluator   |
    JobRequirementsMatcher      |
    JobRequirementsExtractor    |
    LanguageClarityEvaluator    |
    ExperienceEvidenceAgent     |
    ProposalRejectionSuperAgent
)


# final results
from .processing import (
    calc_final_tool_alignment_result,
    calc_job_understanding_result,
    calc_requirement_coverage_score,
    calc_experience_evidence_result,
    calc_language_clarity_result,
    format_ip_for_proposal_tools_analyzer
)


from models.pydantic_schemas import JobToolResponse, JobKeyPointsSchema, ExtractedRequirementsSchema

# errors 
from .pipeline_errors import (
    JobToolsExtractorError,
    ProposalToolsAnalyzerError,
    JobKeyPointsExtractorError,
    JobUnderstandingEvaluatorError,
    JobRequirementExtractorError,
    RequirmentCoverageEvaluatorError,
    ExperienceEvidenceEvaluatorError,
    LanguageClarityEvaluatorError,
)


# Thresholds 
from helpers.config import (
    TA_TOOL_ALIGNMENT_THRESHOLD,
    JD_JOB_UNDERSTANDING_THRESHOLD,
    RQ_REQUIREMENT_COVERAGE_THRESHOLD,
    LANGUAGE_CLARITY_THRESHOLD,
    EXPERIENCE_EVIDENCE_THRESHOLD,
)


from models.data_config import (
    PROPOSAL_REJECTION_REASONS_JOB_FEATURES_EXTRACTION,
    PROPOSAL_REJECTION_REASONS_PROPOSAL_ANALYSIS,
)


class ProposalsRejectionReasonsPipeline:
    """
    This pipeline abstract the whole system by:
        - accepting user input
        - invoking each sub-agent on that input
        - get sub-agents results
        - prepare these results to the super-agent
        - invoke the super-agent to summarize results & give recommendations
    """
    # --------------------- Setup -----------------------
    def __init__(self, agents: dict[str, ProposalRejectionReasons_Type], task: str):
        self.job_tools_extractor           = agents["job_tools_extractor"]
        self.proposal_tools_analyzer       = agents["proposal_tools_analyzer"]
        self.requirement_extractor         = agents["requirement_extractor"]
        self.requirement_matcher           = agents["requirement_matcher"]
        self.job_key_points_extractor      = agents["job_key_points_extractor"]
        self.job_understanding_evaluator   = agents["job_understanding_evaluator"]
        self.experience_evidence_evaluator = agents["experience_evidence_evaluator"]
        self.language_clarity_evaluator    = agents["language_clarity_evaluator"]
        self.super_agent                   = agents["super_agent"]

        
        self.task = task

        self.feature_results = []
        self.extracted_job_desc = None
        self.extracted_job_features = None


    # ------------------------------- driver functions -----------------------------------
    def preprocess(self, input: str | tuple[str, str]):
        if self.task == PROPOSAL_REJECTION_REASONS_JOB_FEATURES_EXTRACTION:
            return self.job_features_extraction_preprocess(input = input)

        elif self.task == PROPOSAL_REJECTION_REASONS_PROPOSAL_ANALYSIS:
            return self.proposal_analysis_preprocess(input = input)

    def call(self, input: str):
        if self.task == PROPOSAL_REJECTION_REASONS_JOB_FEATURES_EXTRACTION:
            return self.job_features_extraction_call(job_desc = input)

        elif self.task == PROPOSAL_REJECTION_REASONS_PROPOSAL_ANALYSIS:
            return self.proposal_analysis_call(input = input)

    def postprocess(self, agent_output):
        if self.task == PROPOSAL_REJECTION_REASONS_JOB_FEATURES_EXTRACTION:
            return self.job_features_extraction_postprocess(agent_output = agent_output)

        elif self.task == PROPOSAL_REJECTION_REASONS_PROPOSAL_ANALYSIS:
            return self.proposal_analysis_postprocess(agent_output = agent_output)
        
    # ------------------------ Utils ---------------------------
    def _raise_pipeline_error(self, error_type, error: Exception):
        if isinstance(error, error_type):
            raise error

        raise error_type(str(error)) from error


    def _get_job_feature(self, job_features: dict, feature_name: str, error_type):
        feature = job_features.get(feature_name)

        if isinstance(feature, Exception):
            self._raise_pipeline_error(error_type, feature)

        if feature is None:
            raise error_type(f"Missing extracted job feature: {feature_name}")

        return feature

    # ---------------------- Job Features Extraction ----------------------
    def job_features_extraction_preprocess(self, input: Any) -> Any:
        return input
    

    def job_features_extraction_postprocess(self, agent_output: Any) -> Any:
        return agent_output


    async def _extract_job_tools(self, job_desc: str):
        try:
            return await self.job_tools_extractor.ainvoke(input = job_desc)
        except Exception as e:
            self._raise_pipeline_error(JobToolsExtractorError, e)


    async def _extract_job_key_points(self, job_desc: str):
        try:
            return await self.job_key_points_extractor.ainvoke(input = job_desc)
        except Exception as e:
            self._raise_pipeline_error(JobKeyPointsExtractorError, e)


    async def _extract_job_requirements(self, job_desc: str):
        try:
            return await self.requirement_extractor.ainvoke(input = job_desc)
        except Exception as e:
            self._raise_pipeline_error(JobRequirementExtractorError, e)


    
    async def job_features_extraction_call(self, job_desc: str) -> dict:
        """
        Run all job-features-extractors agents & save job features
        """
        if self.extracted_job_desc == job_desc and self.extracted_job_features is not None:
            return self.extracted_job_features

        tasks = {
            "job_tools"       : self._extract_job_tools(job_desc = job_desc),
            "job_key_points"  : self._extract_job_key_points(job_desc = job_desc),
            "job_requirements": self._extract_job_requirements(job_desc = job_desc),
        }

        results = await asyncio.gather(
            *tasks.values(),
            return_exceptions = True,
        )

        job_features = dict(zip(tasks.keys(), results))

        if not any(isinstance(feature, Exception) for feature in job_features.values()):
            self.extracted_job_desc = job_desc
            self.extracted_job_features = job_features

        return job_features

    # --------------------------------- Proposal Analysis & Super Agent ----------------------------------
    async def get_tools_alignment_results_from_features(
        self,
        job_features: dict,
        proposal    : str
    ) -> FinalSubagentResult:
        try:
            job_tools_response = self._get_job_feature(
                job_features = job_features,
                feature_name = "job_tools",
                error_type   = JobToolsExtractorError,
            )

            formatted_proposal_analyzer_input = format_ip_for_proposal_tools_analyzer(
                job_tools = job_tools_response.tools,
                proposal  = proposal
            )

            proposal_analysis = await self.proposal_tools_analyzer.ainvoke(
                input = formatted_proposal_analyzer_input
            )

            return calc_final_tool_alignment_result(
                proposal_tools_response = proposal_analysis,
                threshold = TA_TOOL_ALIGNMENT_THRESHOLD
            )
        
        except Exception as e:
            self._raise_pipeline_error(ProposalToolsAnalyzerError, e)


    async def get_job_understanding_results_from_features(
        self,
        job_features: dict,
        proposal    : str
    ) -> FinalSubagentResult:
        try:
            job_key_points = self._get_job_feature(
                job_features = job_features,
                feature_name = "job_key_points",
                error_type   = JobKeyPointsExtractorError,
            )

            understanding_evaluation = await self.job_understanding_evaluator.ainvoke(
                core_problem          = job_key_points.core_problem,
                required_deliverables = job_key_points.required_deliverables,
                key_keywords          = job_key_points.key_keywords,
                proposal_text         = proposal
            )

            return calc_job_understanding_result(
                understanding_evaluation,
                threshold = JD_JOB_UNDERSTANDING_THRESHOLD
            )
        
        except Exception as e:
            self._raise_pipeline_error(JobUnderstandingEvaluatorError, e)


    async def get_requirement_coverage_results_from_features(
        self,
        job_features: dict,
        proposal    : str
    ) -> FinalSubagentResult:
        try:
            job_requirements_response = self._get_job_feature(
                job_features = job_features,
                feature_name = "job_requirements",
                error_type   = JobRequirementExtractorError,
            )

            requirement_matching = await self.requirement_matcher.ainvoke(
                job_requirements = job_requirements_response.requirements,
                proposal_text    = proposal
            )

            return calc_requirement_coverage_score(
                extracted_requirements = job_requirements_response.requirements,
                final_coverage = requirement_matching,
                threshold = RQ_REQUIREMENT_COVERAGE_THRESHOLD
            )
        except Exception as e:
            self._raise_pipeline_error(RequirmentCoverageEvaluatorError, e)


    async def get_experience_evidence_results_async(self, job_desc: str, proposal: str) -> FinalSubagentResult:
        try:
            experience_evidence_evaluation = await self.experience_evidence_evaluator.ainvoke(
                job_desc      = job_desc,
                proposal_text = proposal
            )

            return calc_experience_evidence_result(
                llm_audit = experience_evidence_evaluation,
                threshold = EXPERIENCE_EVIDENCE_THRESHOLD
            )
        except Exception as e:
            self._raise_pipeline_error(ExperienceEvidenceEvaluatorError, e)


    async def get_language_clarity_results_async(self, proposal: str) -> FinalSubagentResult:
        try:
            language_clarity_evaluation = await self.language_clarity_evaluator.ainvoke(
                input = proposal
            )

            return calc_language_clarity_result(
                llm_eval      = language_clarity_evaluation,
                proposal_text = proposal,
                threshold     = LANGUAGE_CLARITY_THRESHOLD
            )
        
        except Exception as e:
            self._raise_pipeline_error(LanguageClarityEvaluatorError, e)


    async def analyze_proposals(
        self,
        job_desc    : str,
        proposal    : str,
        job_features: dict | None = None
    ) -> dict[str, FinalSubagentResult | Exception]:
        """
        Website phase 2:
        Analyze one proposal using previously extracted job features.
        """

        tasks = {
            "tools_alignment": self.get_tools_alignment_results_from_features(
                job_features = job_features,
                proposal     = proposal,
            ),

            "job_understanding": self.get_job_understanding_results_from_features(
                job_features = job_features,
                proposal     = proposal,
            ),

            "requirement_coverage": self.get_requirement_coverage_results_from_features(
                job_features = job_features,
                proposal     = proposal,
            ),

            "experience_evidence": self.get_experience_evidence_results_async(
                job_desc = job_desc,
                proposal = proposal,
            ),

            "language_clarity": self.get_language_clarity_results_async(
                proposal = proposal,
            ),
        }

        results = await asyncio.gather(
            *tasks.values(),
            return_exceptions = True,
        )

        return dict(zip(tasks.keys(), results))

    

    def _trim_text(self, text: str, max_length: int = 350) -> str:
        if len(text) <= max_length:
            return text

        return text[:max_length - 3].rstrip() + "..."


    def _format_subagent_error(self, error: Exception) -> str:
        error_type = type(error).__name__
        error_message = self._trim_text(str(error) or "No error message provided.")

        lines = [
            "> Status: unavailable",
            "> Decision: not evaluated",
            "> Error Indicator: yes",
            f"> Error Type: {error_type}",
            f"> Error Message: {error_message}",
            "> Evidence Note: Ignore this section as rejection evidence.",
        ]

        root_error = error.__cause__ or error.__context__
        if root_error is not None:
            lines.extend([
                f"> Root Error Type: {type(root_error).__name__}",
                f"> Root Error Message: {self._trim_text(str(root_error))}",
            ])

        return "\n".join(lines)


    def _format_reasons(self, title: str, reasons: list[str] | None) -> str:
        if not reasons:
            return f"{title}: None"

        formatted_reasons = "\n".join(f"- {reason}" for reason in reasons)
        return f"{title}:\n{formatted_reasons}"


    def parse_subagents_results(
        self,
        results: dict[str, FinalSubagentResult | Exception | None]
    ) -> str:
        """
        Convert parallel sub-agent results into a stable text block for a super-agent.

        Failed or malformed entries are included as unavailable diagnostics only.
        The super-agent should use completed sections as proposal evidence and
        should not treat unavailable sections as rejection reasons.
        """
        if not results:
            return (
                "# Sub-Agent Evaluation Report\n\n"
                "No sub-agent results were produced. The final report should state "
                "that there is not enough evidence to evaluate the proposal."
            )

        completed_count = sum(
            1
            for feature_result in results.values()
            if isinstance(feature_result, FinalSubagentResult)
        )
        unavailable_count = len(results) - completed_count

        formatted_sections = [
            "# Sub-Agent Evaluation Report",
            "Use completed sections as evidence. Treat unavailable sections as diagnostics only.",
            f"Completed Evaluators: {completed_count}",
            f"Unavailable Evaluators: {unavailable_count}",
        ]

        for feature_name, feature_result in results.items():
            feature_title = feature_name.replace("_", " ").title()
            formatted_feature = [f"## {feature_title}"]

            if isinstance(feature_result, Exception):
                formatted_feature.append(self._format_subagent_error(feature_result))
                formatted_sections.append("\n".join(formatted_feature))
                continue

            if feature_result is None:
                formatted_feature.extend([
                    "- Status: unavailable",
                    "- Decision: not evaluated",
                    "- Error Indicator: yes",
                    "- Error Type: MissingResult",
                    "- Error Message: Sub-agent returned no result.",
                    "- Evidence Note: Ignore this section as rejection evidence.",
                ])
                formatted_sections.append("\n".join(formatted_feature))
                continue

            if not isinstance(feature_result, FinalSubagentResult):
                formatted_feature.extend([
                    "- Status: unavailable",
                    "- Decision: not evaluated",
                    "- Error Indicator: yes",
                    "- Error Type: InvalidResultType",
                    f"- Error Message: Expected FinalSubagentResult, got {type(feature_result).__name__}.",
                    "- Evidence Note: Ignore this section as rejection evidence.",
                ])
                formatted_sections.append("\n".join(formatted_feature))
                continue

            decision = "accepted" if feature_result.accepted else "rejected"
            formatted_feature.extend([
                "- Status: completed",
                f"- Decision: {decision}",
                f"- Score: {feature_result.score}",
            ])

            if feature_result.accepted:
                formatted_feature.append(
                    self._format_reasons(
                        title = "Acceptance Reasons",
                        reasons = feature_result.acceptance_reasons,
                    )
                )
            else:
                formatted_feature.append(
                    self._format_reasons(
                        title = "Rejection Reasons",
                        reasons = feature_result.rejection_reasons,
                    )
                )

            formatted_feature.append(f"Summary: {feature_result.summary}")
            formatted_sections.append("\n".join(formatted_feature))

        return "\n\n".join(formatted_sections)

    # super agent
    async def proposal_analysis_preprocess(
        self,
        input: tuple[str, str, dict[str, JobToolResponse | JobKeyPointsExtractorError | ExtractedRequirementsSchema]]
    ):
        job_desc, proposal, job_features = input

        subagent_results = await self.analyze_proposals(
            job_desc     = job_desc,
            proposal     = proposal,
            job_features = job_features,
        )

        return self.parse_subagents_results(subagent_results)


    async def proposal_analysis_call(self, input: tuple[str, str, str]) -> SuperAgentResponse:
        job_desc, proposal, parsed_subagent_results = input
        return await self.super_agent.ainvoke(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = parsed_subagent_results
        )
    

    def proposal_analysis_postprocess(self, agent_output: SuperAgentResponse) -> str:
        """
        Convert the structured super-agent response into a readable final report.
        """
        def format_list_section(title: str, items: list[str] | None) -> str:
            valid_items = [
                str(item).strip()
                for item in (items or [])
                if str(item).strip()
            ]

            if not valid_items:
                return f"## {title}\nNone"

            formatted_items = "\n".join(f"- {item}" for item in valid_items)
            return f"## {title}\n{formatted_items}"

        verdict_labels = {
            "accepted": "Accepted",
            "at_risk": "At Risk",
            "rejected": "Rejected",
        }

        verdict = verdict_labels.get(
            agent_output.verdict,
            agent_output.verdict.replace("_", " ").title()
        )

        report_sections = [
            "# Proposal Rejection Reasons Report",
            f"## Overall Verdict\n{verdict}",
            f"## Summary\n{agent_output.summary_report}",
            format_list_section(
                title = "Strengths",
                items = agent_output.strengths_points,
            ),
            format_list_section(
                title = "Rejection Risks",
                items = agent_output.weakness_points,
            ),
            format_list_section(
                title = "Recommendations",
                items = agent_output.recommendations,
            ),
            format_list_section(
                title = "Evaluation Limitations",
                items = agent_output.evaluation_limitations,
            ),
        ]

        return "\n\n".join(report_sections)

    
    


