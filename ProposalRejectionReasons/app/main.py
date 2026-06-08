# ---------------------------------------------------------------------
# The Main Workflow
# ---------------------------------------------------------------------

# agents
from agents import JobToolsExtractor, ProposalToolsAnalyzer
from agents import JobRequirementsExtractor, JobRequirementsMatcher
from agents import JobKeyPointsExtractor, JobUnderstandingEvaluator
from agents import ExperienceEvidenceAgent
from agents import LanguageClarityEvaluator

# schemas
from schemas import ExperienceEvidenceSchema
from schemas import JobToolResponse, ProposalToolsResponse
from schemas import JobKeyPointsSchema, JobUnderstandingEvalSchema
from schemas import ExtractedRequirementsSchema, RequirementCoverageSchema
from schemas import LanguageClarityEvalSchema

# prompts
from prompts import JOB_TOOLS_EXTRACTION_PROMPT, PROPOSAL_TOOLS_EXTRACTION_PROMPT
from prompts import JOB_KEY_POINTS_EXTRACTION_PROMPT, JOB_UNDERSTANDING_EVALUATOR_PROMPT
from prompts import REQUIREMENT_EXTRACTOR_PROMPT, REQUIREMENT_MATCHER_PROMPT
from prompts import EXPERIENCE_EVIDENCE_PROMPT
from prompts import LANGUAGE_CLARITY_EVALUATOR_PROMPT

# data processing
# from processing.tool_alignment_processing import format_ip_for_proposal_tools_analyzer, calc_tools_alignment_score
# from processing.job_understanding_processing import calc_job_understanding_result
from processing.requirement_coverage_processing import calc_requirement_coverage_score
# from processing.language_clarity_processing import calc_language_clarity_result
#from processing.experience_evidence import calc_experience_evidence_result

# others
import os
from pathlib import Path
import logging
import traceback
import helpers.config as CFG
import helpers.functional as F
from dotenv import load_dotenv


load_dotenv()

DATA_PATH = os.path.join(
    Path(__file__).parent.parent,
    "eval_data"
)



#     # # --------------------------------------------
#     # F.print_subtitle("Language Clarity")

#     # lc_sample = language_clarity_data_samples[0]
#     # proposals = lc_sample["proposals"]

#     # print("\t>> Evaluating Language Clarity")
#     # for idx, proposal in enumerate(proposals, start=1):
#     #     print(f"--- Analyzing Proposal {idx} ---")
#     #     llm_eval = language_clarity_evaluator.invoke(proposal_text=proposal)
#     #     F.print_structured_response(llm_eval)

#     #     print("Final Result (text metrics + scoring): ")
#     #     result = calc_language_clarity_result(
#     #         llm_eval      = llm_eval,
#     #         proposal_text = proposal
#     #     )
#     #     for key, value in result.items():
#     #         print(f"  {key} => {value}")
#     #     print()

#     # --------------------------------------------
#     # --------------------------------------------




import time
from agent_pipelines.proposal_rejection_reasons_pipeline import ProposalsRejectionReasonsPipeline
EXAMPLES_DATA_PATH = "/mnt/d/Education/College/______GraduationProject/AI-Models/ProposalRejectionReasons/app/data_examples"
from schemas import FinalSubagentResult
from typing import Callable


LOGS_PATH = Path(__file__).parent / "logs"
LOG_FILE_PATH = LOGS_PATH / "pipeline_test.log"


def setup_logging() -> logging.Logger:
    LOGS_PATH.mkdir(
        parents = True,
        exist_ok = True
    )

    logger = logging.getLogger("proposal_rejection_pipeline_test")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt = "%(asctime)s | %(levelname)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(
        LOG_FILE_PATH,
        encoding = "utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def print_and_log(title: str, content, logger: logging.Logger) -> None:
    F.print_subtitle(title)
    print(content)
    logger.info("%s\n%s", title, content)


def stringify_result(result) -> str:
    if isinstance(result, Exception):
        return format_error_details(result)

    if hasattr(result, "model_dump_json"):
        return result.model_dump_json(indent = 2)

    return str(result)


def format_error_details(error: Exception) -> str:
    error_lines = []
    current_error = error
    depth = 0

    while current_error is not None:
        prefix = "Root Error" if depth == 0 else f"Caused By #{depth}"
        error_lines.append(f"{prefix}: {type(current_error).__name__}")
        error_lines.append(f"Message: {str(current_error)}")

        if current_error.args:
            error_lines.append(f"Args: {current_error.args}")

        next_error = current_error.__cause__ or current_error.__context__
        if next_error is not None:
            error_lines.append("")

        current_error = next_error
        depth += 1

    traceback_text = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    error_lines.extend(["", "Traceback:", traceback_text])

    return "\n".join(error_lines)


def test_feature(feature_name: str, data_file_name: str, feature_func: Callable[[str, str], FinalSubagentResult]):
    F.print_subtitle(feature_name)

    print(">> Loading Data: ")
    try:
        data = F.load_json(os.path.join(EXAMPLES_DATA_PATH, data_file_name))
    except FileNotFoundError as e:
        F.print_error_message("Data File Not Found")
        return
    else:
        sample = data[0]
        F.print_success_message("Data Loaded Successfully")

        if feature_name == "Language Clarity":
            proposal = sample["proposal"]
            job_desc = None
        
        else:
            sample = data[0]
            job_desc = sample.get("job_desc", None)
            proposal = sample['proposals'][0]
            
    

    print(">> Testing: ")
    try:
        if job_desc is not None:
            result = feature_func(job_desc, proposal)
        else:
            result = feature_func(proposal)
    except Exception as e:
        F.print_error_message("Agent Error Details:")
        print(format_error_details(e))
        return 
    F.print_data(result)

    F.print_success_message(f"Feature {feature_name} Worked Successfuly")
    time.sleep(3)

async def test_full_pipeline_with_logging():
    logger = setup_logging()

    F.print_title("1.0 Starting Full Pipeline Test")
    logger.info("Starting full pipeline test")

    F.print_subtitle("Initiating Agents")
    logger.info("Initiating pipeline and all agents")
    pipeline = ProposalsRejectionReasonsPipeline()
    logger.info("Pipeline initiated successfully")

    F.print_subtitle("Loading Test Data")
    samples = F.load_json(
        os.path.join(EXAMPLES_DATA_PATH, "requirement_coverage_samples.json")
    )
    logger.info("Loaded %s samples", len(samples))

    sample = samples[0]
    job_desc = sample.get("job_desc", None)
    proposal = sample['proposals'][0]

    print_and_log(
        title = "Job Description",
        content = job_desc,
        logger = logger
    )
    print_and_log(
        title = "Proposal Under Test",
        content = proposal,
        logger = logger
    )

    F.print_subtitle("Running Subagents In Parallel")
    logger.info("Calling get_all_results")
    subagent_results = await pipeline.get_all_results(
        job_desc = job_desc,
        proposal = proposal
    )
    logger.info("Subagents finished")

    F.print_subtitle("Raw Subagent Results")
    for feature_name, feature_result in subagent_results.items():
        result_text = stringify_result(feature_result)
        print_and_log(
            title = f"Raw Result - {feature_name}",
            content = result_text,
            logger = logger
        )

    F.print_subtitle("Parsing Subagent Results For Super Agent")
    logger.info("Calling parse_subagents_results")
    parsed_subagent_results = pipeline.parse_subagents_results(
        results = subagent_results
    )
    print_and_log(
        title = "Parsed Super-Agent Input",
        content = parsed_subagent_results,
        logger = logger
    )

    F.print_subtitle("Calling Super Agent")
    logger.info("Calling super agent")
    try:
        super_agent_report = pipeline.super_agent.invoke(
            job_desc = job_desc,
            proposal = proposal,
            subagents_results = parsed_subagent_results
        )
    except Exception as e:
        error_details = format_error_details(e)
        F.print_error_message("Super Agent Error Details:")
        print(error_details)
        logger.error("Super agent failed\n%s", error_details)
        return

    print_and_log(
        title = "Final Super-Agent Report",
        content = super_agent_report,
        logger = logger
    )

    F.print_success_message("Full pipeline test completed")
    logger.info("Full pipeline test completed")


async def main():
    await test_full_pipeline_with_logging()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    
    

    
