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
from .super_agent import ProposalRejectionSuperAgent, SuperAgentResponse


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
        self.extracted_job_desc = None
        self.extracted_job_features = None


    def init_subagents(self) -> None:
        self.job_tools_extractor            = JobToolsExtractor(**TA_JOB_TOOLS_EXTRACTOR_CFG)
        self.proposal_tools_analyzer        = ProposalToolsAnalyzer(**TA_PROPOSAL_TOOLS_ANALYZER_CFG)
        self.requirement_extractor          = JobRequirementsExtractor(**RQ_REQUIREMENT_EXTRACTOR_CFG)
        self.requirement_matcher            = JobRequirementsMatcher(**RQ_REQUIREMENT_COVERAGE_EVALUATOR_CFG)
        self.job_key_points_extractor       = JobKeyPointsExtractor(**JD_JOB_KEY_POINTS_CFG)
        self.job_understanding_evaluator    = JobUnderstandingEvaluator(**JD_JOB_UNDERSTANDING_EVALUATOR_CFG)
        self.experience_evidence_evaluator  = ExperienceEvidenceAgent(**EVIDENCE_OF_EXPERIENCE_EVALUATOR_CFG)
        self.language_clarity_evaluator     = LanguageClarityEvaluator(**LANGUAGE_CLARITY_EVALUATOR_CFG)

        self.super_agent                    = ProposalRejectionSuperAgent(
            **SUPER_AGENT_CFG
        )


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

    
    async def extract_job_features(self, job_desc: str) -> dict:
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

            return get_final_tool_alignment_result(
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
                key_keywords          = getattr(job_key_points, "key_keywords", []),
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


    async def analyze_proposal(
        self,
        job_desc    : str,
        proposal    : str,
        job_features: dict | None = None
    ) -> dict[str, FinalSubagentResult | Exception]:
        """
        Website phase 2:
        Analyze one proposal using previously extracted job features.
        """
        if job_features is None:
            job_features = await self.extract_job_features(job_desc = job_desc)

        tasks = {
            "tools_alignment": self.get_tools_alignment_results_from_features(
                job_features = job_features,
                proposal = proposal,
            ),

            "job_understanding": self.get_job_understanding_results_from_features(
                job_features = job_features,
                proposal = proposal,
            ),

            "requirement_coverage": self.get_requirement_coverage_results_from_features(
                job_features = job_features,
                proposal = proposal,
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

    
    
    async def get_all_results(self, job_desc: str, proposal: str) -> dict[str, FinalSubagentResult | Exception]:
        job_features = await self.extract_job_features(
            job_desc = job_desc
        )

        return await self.analyze_proposal(
            job_desc     = job_desc,
            proposal     = proposal,
            job_features = job_features
        )
    

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


    async def get_super_agent_report(self, job_desc: str, proposal: str) -> SuperAgentResponse:
        subagent_results = await self.get_all_results(
            job_desc = job_desc,
            proposal = proposal
        )
        parsed_subagent_results = self.parse_subagents_results(
            results = subagent_results
        )

        return await self.super_agent.ainvoke(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = parsed_subagent_results
        )
    

    def format_final_result(self, super_agent_response: SuperAgentResponse) -> str:
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
            super_agent_response.verdict,
            super_agent_response.verdict.replace("_", " ").title()
        )

        report_sections = [
            "# Proposal Rejection Reasons Report",
            f"## Overall Verdict\n{verdict}",
            f"## Summary\n{super_agent_response.summary_report}",
            format_list_section(
                title = "Strengths",
                items = super_agent_response.strengths_points,
            ),
            format_list_section(
                title = "Rejection Risks",
                items = super_agent_response.weakness_points,
            ),
            format_list_section(
                title = "Recommendations",
                items = super_agent_response.recommendations,
            ),
            format_list_section(
                title = "Evaluation Limitations",
                items = super_agent_response.evaluation_limitations,
            ),
        ]

        return "\n\n".join(report_sections)

    
