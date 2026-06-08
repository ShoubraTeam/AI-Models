# -----------------------------------------
# Building the system pipeline
# -----------------------------------------

import asyncio


# sub-agents
from agents.tools_alignment      import JobToolsExtractor, ProposalToolsAnalyzer
from agents.experience_evidence  import ExperienceEvidenceAgent
from agents.language_clarity     import LanguageClarityEvaluator
from agents.job_understanding    import JobUnderstandingEvaluator, JobKeyPointsExtractor
from agents.requirement_coverage import JobRequirementsExtractor, JobRequirementsMatcher
from .super_agent import ProposalRejectionSuperAgent


# final results
from processing import (
    get_final_tool_alignment_result,
    calc_job_understanding_result,
    calc_requirement_coverage_score,
    calc_experience_evidence_result,
    calc_language_clarity_result,
)

# cfg
from .pipeline_config import (
    TA_JOB_TOOLS_EXTRACTOR_CFG,
    TA_PROPOSAL_TOOLS_ANALYZER_CFG,
    JD_JOB_KEY_POINTS_CFG,
    JD_JOB_UNDERSTANDING_EVALUATOR_CFG,
    RQ_REQUIREMENT_EXTRACTOR_CFG,
    RQ_REQUIREMENT_COVERAGE_EVALUATOR_CFG,
    LANGUAGE_CLARITY_EVALUATOR_CFG,
    EVIDENCE_OF_EXPERIENCE_EVALUATOR_CFG,
    SUPER_AGENT_CFG,
)

from .pipeline_config import (
    TA_TOOL_ALIGNMENT_THRESHOLD,
    JD_JOB_UNDERSTANDING_THRESHOLD,
    RQ_REQUIREMENT_COVERAGE_THRESHOLD,
    EXPERIENCE_EVIDENCE_THRESHOLD,
    LANGUAGE_CLARITY_THRESHOLD
)


# preprocessing
from processing.tool_alignment_processing import format_ip_for_proposal_tools_analyzer

from schemas import FinalSubagentResult


# errors 
from .pipeline_errors import (
    ProposalRejecionReasonsError,
    JobToolsExtractorError,
    ProposalToolsAnalyzerError,
    JobKeyPointsExtractorError,
    JobUnderstandingEvaluatorError,
    JobRequirementExtractorError,
    RequirmentCoverageEvaluatorError,
    ExperienceEvidenceEvaluatorError,
    LanguageClarityEvaluatorError,

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
    def __init__(self):
        self.init_subagents()
        self.feature_results = []


    def init_subagents(self) -> None:
        self.job_tools_extractor            = JobToolsExtractor(**TA_JOB_TOOLS_EXTRACTOR_CFG)
        self.proposal_tools_analyzer        = ProposalToolsAnalyzer(**TA_PROPOSAL_TOOLS_ANALYZER_CFG)
        self.requirement_extractor          = JobRequirementsExtractor(**RQ_REQUIREMENT_EXTRACTOR_CFG)
        self.requirement_matcher            = JobRequirementsMatcher(**RQ_REQUIREMENT_COVERAGE_EVALUATOR_CFG)
        self.job_key_points_extractor       = JobKeyPointsExtractor(**JD_JOB_KEY_POINTS_CFG)
        self.job_understanding_evaluator    = JobUnderstandingEvaluator(**JD_JOB_UNDERSTANDING_EVALUATOR_CFG)
        self.experience_evidence_evaluator  = ExperienceEvidenceAgent(**EVIDENCE_OF_EXPERIENCE_EVALUATOR_CFG)
        self.language_clarity_evaluator     = LanguageClarityEvaluator(**LANGUAGE_CLARITY_EVALUATOR_CFG)
        self.super_agent                    = ProposalRejectionSuperAgent(**SUPER_AGENT_CFG)

    def _raise_pipeline_error(self, error_type, error: Exception):
        if isinstance(error, error_type):
            raise error

        raise error_type(str(error)) from error

    # ---------------------- Workflows --------------------------
    def get_tools_alignment_results(self, job_desc: str, proposal: str) -> FinalSubagentResult:
        try:
            job_tools_response = self.job_tools_extractor.invoke(input = job_desc)
        except Exception as e:
            self._raise_pipeline_error(JobToolsExtractorError, e)

        try:
            formatted_proposal_analyzer_input = format_ip_for_proposal_tools_analyzer(
                job_tools = job_tools_response.tools,
                proposal  = proposal
            )


            proposal_analysis = self.proposal_tools_analyzer.invoke(formatted_proposal_analyzer_input)
        except Exception as e:
            self._raise_pipeline_error(ProposalToolsAnalyzerError, e)

        try:
            tool_alignment_result = get_final_tool_alignment_result(
                proposal_tools_response = proposal_analysis,
                threshold               = TA_TOOL_ALIGNMENT_THRESHOLD
            )
            return tool_alignment_result
        except Exception as e:
            self._raise_pipeline_error(ProposalToolsAnalyzerError, e)
    


    def get_job_understanding_results(self, job_desc: str, proposal: str) -> FinalSubagentResult:
        try:
            job_key_points = self.job_key_points_extractor.invoke(input = job_desc)
        except Exception as e:
            self._raise_pipeline_error(JobKeyPointsExtractorError, e)

        try:
            understanding_evaluation = self.job_understanding_evaluator.invoke(
                core_problem          = job_key_points.core_problem,
                required_deliverables = job_key_points.required_deliverables,
                proposal_text         = proposal
            )
        except Exception as e:
            self._raise_pipeline_error(JobUnderstandingEvaluatorError, e)

        try:
            job_understanding_result = calc_job_understanding_result(
                understanding_evaluation,
                threshold = JD_JOB_UNDERSTANDING_THRESHOLD
            )
            return job_understanding_result
        except Exception as e:
            self._raise_pipeline_error(JobUnderstandingEvaluatorError, e)
    

    def get_requirement_coverage_results(self, job_desc: str, proposal: str) -> FinalSubagentResult:
        try:
            job_requirements_response = self.requirement_extractor.invoke(input = job_desc)
        except Exception as e:
            self._raise_pipeline_error(JobRequirementExtractorError, e)

        try:
            requirement_matching = self.requirement_matcher.invoke(
                job_requirements = job_requirements_response.requirements,
                proposal_text = proposal 
            )
        except Exception as e:
            self._raise_pipeline_error(RequirmentCoverageEvaluatorError, e)

        try:
            requirement_coverage_result = calc_requirement_coverage_score(
                extracted_requirements = job_requirements_response.requirements,
                final_coverage         = requirement_matching,
                threshold              = RQ_REQUIREMENT_COVERAGE_THRESHOLD
            )
            return requirement_coverage_result
        except Exception as e:
            self._raise_pipeline_error(RequirmentCoverageEvaluatorError, e)


    def get_experience_evidence_results(self, job_desc: str, proposal: str) -> FinalSubagentResult:
        try:
            experience_evidence_evaluation = self.experience_evidence_evaluator.invoke(
                job_desc      = job_desc,
                proposal_text = proposal
            )
        except Exception as e:
            self._raise_pipeline_error(ExperienceEvidenceEvaluatorError, e)

        try:
            experience_evidence_results = calc_experience_evidence_result(
                llm_audit = experience_evidence_evaluation,
                threshold = EXPERIENCE_EVIDENCE_THRESHOLD
            )
            return experience_evidence_results
        except Exception as e:
            self._raise_pipeline_error(ExperienceEvidenceEvaluatorError, e)
    
    def get_language_clarity_results(self, proposal: str) -> FinalSubagentResult:
        try:
            language_clarity_evaluation = self.language_clarity_evaluator.invoke(proposal_text = proposal)
        except Exception as e:
            self._raise_pipeline_error(LanguageClarityEvaluatorError, e)

        try:
            language_clarity_results = calc_language_clarity_result(
                llm_eval       = language_clarity_evaluation,
                proposal_text = proposal,
                threshold     = LANGUAGE_CLARITY_THRESHOLD
            )
            return language_clarity_results
        except Exception as e:
            self._raise_pipeline_error(LanguageClarityEvaluatorError, e)
    

    async def get_all_results(self, job_desc: str, proposal: str) -> dict[str, FinalSubagentResult | Exception]:
        tasks = {
            "tools_alignment": asyncio.to_thread(
                self.get_tools_alignment_results,
                job_desc,
                proposal,
            ),
            
            "job_understanding": asyncio.to_thread(
                self.get_job_understanding_results,
                job_desc,
                proposal,
            ),

            "requirement_coverage": asyncio.to_thread(
                self.get_requirement_coverage_results,
                job_desc,
                proposal,
            ),

            "experience_evidence": asyncio.to_thread(
                self.get_experience_evidence_results,
                job_desc,
                proposal,
            ),

            "language_clarity": asyncio.to_thread(
                self.get_language_clarity_results,
                proposal,
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
            "Status: unavailable",
            "Decision: not evaluated",
            "Error Indicator: yes",
            f"Error Type: {error_type}",
            f"Error Message: {error_message}",
            "Evidence Note: Ignore this section as rejection evidence.",
        ]

        root_error = error.__cause__ or error.__context__
        if root_error is not None:
            lines.extend([
                f"Root Error Type: {type(root_error).__name__}",
                f"Root Error Message: {self._trim_text(str(root_error))}",
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
                    "Status: unavailable",
                    "Decision: not evaluated",
                    "Error Indicator: yes",
                    "Error Type: MissingResult",
                    "Error Message: Sub-agent returned no result.",
                    "Evidence Note: Ignore this section as rejection evidence.",
                ])
                formatted_sections.append("\n".join(formatted_feature))
                continue

            if not isinstance(feature_result, FinalSubagentResult):
                formatted_feature.extend([
                    "Status: unavailable",
                    "Decision: not evaluated",
                    "Error Indicator: yes",
                    "Error Type: InvalidResultType",
                    f"Error Message: Expected FinalSubagentResult, got {type(feature_result).__name__}.",
                    "Evidence Note: Ignore this section as rejection evidence.",
                ])
                formatted_sections.append("\n".join(formatted_feature))
                continue

            decision = "accepted" if feature_result.accepted else "rejected"
            formatted_feature.extend([
                "Status: completed",
                f"Decision: {decision}",
                f"Score: {feature_result.score}",
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


    async def get_super_agent_report(self, job_desc: str, proposal: str) -> str:
        subagent_results = await self.get_all_results(
            job_desc = job_desc,
            proposal = proposal
        )
        parsed_subagent_results = self.parse_subagents_results(
            results = subagent_results
        )

        return self.super_agent.invoke(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = parsed_subagent_results
        )
    
    
