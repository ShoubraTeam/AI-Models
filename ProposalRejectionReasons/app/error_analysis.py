# ------------- Imports
from dotenv import load_dotenv
load_dotenv()

import json
import os
from dataclasses import dataclass
from io import TextIOWrapper
from typing import Any, Callable

import helpers.functional as F
import helpers.config as CFG

from agents import (
    ExperienceEvidenceAgent,
    JobKeyPointsExtractor,
    JobRequirementsExtractor,
    JobRequirementsMatcher,
    JobToolsExtractor,
    JobUnderstandingEvaluator,
    LanguageClarityEvaluator,
    ProposalToolsAnalyzer,
)
from Groq_Native import get_response_format
from processing.tool_alignment_processing import format_ip_for_proposal_tools_analyzer
from prompts.groq_native_prompts import (
    EXPERIENCE_EVIDENCE_PROMPT,
    JOB_KEY_POINTS_EXTRACTION_PROMPT,
    JOB_TOOLS_EXTRACTION_PROMPT,
    JOB_UNDERSTANDING_EVALUATOR_PROMPT,
    LANGUAGE_CLARITY_EVALUATOR_PROMPT,
    PROPOSAL_TOOLS_EXTRACTION_PROMPT,
    REQUIREMENT_EXTRACTOR_PROMPT,
    REQUIREMENT_MATCHER_PROMPT,
)
from schemas import (
    ExperienceEvidenceSchema,
    ExtractedRequirementsSchema,
    JobKeyPointsSchema,
    JobTool,
    JobToolResponse,
    JobUnderstandingEvalSchema,
    LanguageClarityEvalSchema,
    ProposalToolsResponse,
    RequirementCoverageSchema,
)
from helpers.logging import (
    write_in_file,
    TEXT_DICT,
    TEXT_NORMAL,
    TEXT_TITLE,
)


# ------------------- CFG ------------------------------------ #
TASK_JOB_TOOLS_EXTRACTION = "job_tools_extraction"
TASK_PROPOSAL_TOOLS_ANALYSIS = "proposal_tools_analysis"
TASK_JOB_KEY_POINTS_EXTRACTION = "job_key_points_extraction"
TASK_JOB_UNDERSTANDING_EVALUATION = "job_understanding_evaluation"
TASK_REQUIREMENT_EXTRACTION = "requirement_extraction"
TASK_REQUIREMENT_COVERAGE = "requirement_coverage"
TASK_EXPERIENCE_EVIDENCE = "experience_evidence"
TASK_LANGUAGE_CLARITY = "language_clarity"

BASE_DIR = "/mnt/d/Education/College/______GraduationProject/AI-Models/ProposalRejectionReasons/app"
DATA_PATH = os.path.join(BASE_DIR, "assets", "eval_data")
ERROR_ANALYSIS_PATH = os.path.join(BASE_DIR, "assets", "error_analysis")

ERROR_ANALYSIS_TASKS = [
    TASK_EXPERIENCE_EVIDENCE,
    # TASK_JOB_KEY_POINTS_EXTRACTION,
    # TASK_JOB_TOOLS_EXTRACTION,
    # TASK_PROPOSAL_TOOLS_ANALYSIS,
    # TASK_JOB_UNDERSTANDING_EVALUATION,
    # TASK_REQUIREMENT_EXTRACTION,
    # TASK_REQUIREMENT_COVERAGE,
    # TASK_LANGUAGE_CLARITY,
]

MODELS_CFGS = [
    {
        "temperature": 0.0,
        "max_tokens": 2048,
        "timeout": 30,
    }
]

MODELS = [
    CFG.GROQ_GPT_20b,
    CFG.GROQ_QWEN_32b,
    CFG.GROQ_LLAMA_8b,
]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    agent_cls: type
    prompt: Any
    schema: type
    runner: Callable[[TextIOWrapper, Any, dict], int]


# ------------------------ Output helpers ------------------------------ #
def print_sub(file: TextIOWrapper, n_dashes: int = 3) -> None:
    write_in_file(file, "")
    write_in_file(file, "")
    write_in_file(file, "-" * n_dashes, TEXT_NORMAL)
    write_in_file(file, "")
    write_in_file(file, "")


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, indent=4, ensure_ascii=False)
    except TypeError:
        return str(value)


def print_native_error(file: TextIOWrapper, error: Exception) -> None:
    write_in_file(file, "")
    write_in_file(file, "LLM Native Structured Output Failure", TEXT_NORMAL)
    write_in_file(file, f"Error Type : {type(error).__name__}", TEXT_NORMAL, n_identation=1)
    write_in_file(file, f"Error      : {error}", TEXT_NORMAL, n_identation=1)

    body = getattr(error, "body", None)
    if body is not None:
        write_in_file(file, "Groq Error Body:", TEXT_NORMAL, n_identation=1)
        write_in_file(file, _safe_json(body), TEXT_DICT, 2)

    raw_output = getattr(error, "raw_output", None)
    if raw_output is not None:
        write_in_file(file, "Raw Model Output:", TEXT_NORMAL, n_identation=1)
        try:
            parsed = json.loads(raw_output) if isinstance(raw_output, str) else raw_output
            write_in_file(file, _safe_json(parsed), TEXT_DICT, 2)
        except Exception:
            write_in_file(file, str(raw_output), TEXT_NORMAL, 2)

    root_error = error.__cause__ or error.__context__
    if root_error is not None:
        write_in_file(file, "Root Cause:", TEXT_NORMAL, n_identation=1)
        write_in_file(file, f"{type(root_error).__name__}: {root_error}", TEXT_NORMAL, 2)

    write_in_file(file, "")


def write_response(file: TextIOWrapper, response: Any) -> None:
    write_in_file(file, "Model Response:", TEXT_NORMAL)
    if hasattr(response, "model_dump"):
        write_in_file(file, response.model_dump(), TEXT_DICT, 1)
    else:
        write_in_file(file, str(response), TEXT_NORMAL, 1)


# ------------------------ Task runners ------------------------------ #
def run_job_desc_only(file: TextIOWrapper, agent: Any, sample: dict) -> int:
    job_desc = sample["job_desc"]
    write_in_file(file, f"\n\nJob Desc:\n{job_desc}\n", TEXT_NORMAL)

    try:
        response = agent.invoke(input=job_desc)
    except Exception as error:
        print_native_error(file, error)
        print_sub(file)
        return 1

    write_response(file, response)
    print_sub(file)
    return 0


def run_job_key_points(file: TextIOWrapper, agent: JobKeyPointsExtractor, sample: dict) -> int:
    job_desc = sample["job_desc"]
    write_in_file(file, f"\n\nJob Desc:\n{job_desc}\n", TEXT_NORMAL)

    try:
        response = agent.invoke(job_desc=job_desc)
    except Exception as error:
        print_native_error(file, error)
        print_sub(file)
        return 1

    write_response(file, response)
    print_sub(file)
    return 0


def run_proposal_tools(file: TextIOWrapper, agent: ProposalToolsAnalyzer, sample: dict) -> int:
    job_tools = [JobTool(**tool) for tool in sample.get("job_data", {}).get("tools", [])]
    proposals = sample.get("proposals", [])
    n_errors = 0

    write_in_file(file, f"\n\nJob Tools:\n{[tool.model_dump() for tool in job_tools]}\n", TEXT_NORMAL)

    for proposal_idx, proposal_sample in enumerate(proposals, start=1):
        proposal = proposal_sample["proposal"]
        write_in_file(file, f">> Proposal #{proposal_idx}:", TEXT_NORMAL)
        write_in_file(file, proposal, TEXT_NORMAL, 1)

        formatted_input = format_ip_for_proposal_tools_analyzer(
            job_tools=job_tools,
            proposal=proposal,
        )
        try:
            response = agent.invoke(input=formatted_input)
        except Exception as error:
            print_native_error(file, error)
            n_errors += 1
        else:
            write_response(file, response)

        print_sub(file)

    return n_errors


def run_job_understanding(file: TextIOWrapper, agent: JobUnderstandingEvaluator, sample: dict) -> int:
    job_data = sample.get("job_data", {})
    proposals = sample.get("proposals", [])
    n_errors = 0

    write_in_file(file, f"\n\nJob Data:\n{job_data}\n", TEXT_NORMAL)

    for proposal_idx, proposal_sample in enumerate(proposals, start=1):
        proposal = proposal_sample["proposal"]
        write_in_file(file, f">> Proposal #{proposal_idx}:", TEXT_NORMAL)
        write_in_file(file, proposal, TEXT_NORMAL, 1)

        try:
            response = agent.invoke(
                core_problem=job_data.get("core_problem", ""),
                required_deliverables=job_data.get("required_deliverables", []),
                key_keywords=job_data.get("key_keywords", []),
                proposal_text=proposal,
            )
        except Exception as error:
            print_native_error(file, error)
            n_errors += 1
        else:
            write_response(file, response)

        print_sub(file)

    return n_errors


def run_requirement_coverage(file: TextIOWrapper, agent: JobRequirementsMatcher, sample: dict) -> int:
    requirements = sample.get("job_data", {}).get("requirements", [])
    proposals = sample.get("proposals", [])
    n_errors = 0

    write_in_file(file, f"\n\nRequirements:\n{requirements}\n", TEXT_NORMAL)

    for proposal_idx, proposal_sample in enumerate(proposals, start=1):
        proposal = proposal_sample["proposal"]
        write_in_file(file, f">> Proposal #{proposal_idx}:", TEXT_NORMAL)
        write_in_file(file, proposal, TEXT_NORMAL, 1)

        try:
            response = agent.invoke(
                job_requirements=requirements,
                proposal_text=proposal,
            )
        except Exception as error:
            print_native_error(file, error)
            n_errors += 1
        else:
            write_response(file, response)

        print_sub(file)

    return n_errors


def run_experience_evidence(file: TextIOWrapper, agent: ExperienceEvidenceAgent, sample: dict) -> int:
    job_desc = sample["job_desc"]
    proposals = sample.get("proposals", [])
    n_errors = 0

    write_in_file(file, f"\n\nJob Desc:\n{job_desc}\n", TEXT_NORMAL)

    for proposal_idx, proposal_sample in enumerate(proposals, start=1):
        proposal = proposal_sample["proposal"]
        write_in_file(file, f">> Proposal #{proposal_idx}:", TEXT_NORMAL)
        write_in_file(file, proposal, TEXT_NORMAL, 1)

        try:
            response = agent.invoke(job_desc=job_desc, proposal_text=proposal)
        except Exception as error:
            print_native_error(file, error)
            n_errors += 1
        else:
            write_response(file, response)

        print_sub(file)

    return n_errors


def run_language_clarity(file: TextIOWrapper, agent: LanguageClarityEvaluator, sample: dict) -> int:
    proposals = sample.get("proposals", [])
    n_errors = 0

    for proposal_idx, proposal_sample in enumerate(proposals, start=1):
        proposal = proposal_sample["proposal"]
        write_in_file(file, f">> Proposal #{proposal_idx}:", TEXT_NORMAL)
        write_in_file(file, proposal, TEXT_NORMAL, 1)

        try:
            response = agent.invoke(proposal_text=proposal)
        except Exception as error:
            print_native_error(file, error)
            n_errors += 1
        else:
            write_response(file, response)

        print_sub(file)

    return n_errors


TASK_SPECS = {
    TASK_JOB_TOOLS_EXTRACTION: TaskSpec(
        TASK_JOB_TOOLS_EXTRACTION,
        JobToolsExtractor,
        JOB_TOOLS_EXTRACTION_PROMPT,
        JobToolResponse,
        run_job_desc_only,
    ),
    TASK_PROPOSAL_TOOLS_ANALYSIS: TaskSpec(
        TASK_PROPOSAL_TOOLS_ANALYSIS,
        ProposalToolsAnalyzer,
        PROPOSAL_TOOLS_EXTRACTION_PROMPT,
        ProposalToolsResponse,
        run_proposal_tools,
    ),
    TASK_JOB_KEY_POINTS_EXTRACTION: TaskSpec(
        TASK_JOB_KEY_POINTS_EXTRACTION,
        JobKeyPointsExtractor,
        JOB_KEY_POINTS_EXTRACTION_PROMPT,
        JobKeyPointsSchema,
        run_job_key_points,
    ),
    TASK_JOB_UNDERSTANDING_EVALUATION: TaskSpec(
        TASK_JOB_UNDERSTANDING_EVALUATION,
        JobUnderstandingEvaluator,
        JOB_UNDERSTANDING_EVALUATOR_PROMPT,
        JobUnderstandingEvalSchema,
        run_job_understanding,
    ),
    TASK_REQUIREMENT_EXTRACTION: TaskSpec(
        TASK_REQUIREMENT_EXTRACTION,
        JobRequirementsExtractor,
        REQUIREMENT_EXTRACTOR_PROMPT,
        ExtractedRequirementsSchema,
        run_job_desc_only,
    ),
    TASK_REQUIREMENT_COVERAGE: TaskSpec(
        TASK_REQUIREMENT_COVERAGE,
        JobRequirementsMatcher,
        REQUIREMENT_MATCHER_PROMPT,
        RequirementCoverageSchema,
        run_requirement_coverage,
    ),
    TASK_EXPERIENCE_EVIDENCE: TaskSpec(
        TASK_EXPERIENCE_EVIDENCE,
        ExperienceEvidenceAgent,
        EXPERIENCE_EVIDENCE_PROMPT,
        ExperienceEvidenceSchema,
        run_experience_evidence,
    ),
    TASK_LANGUAGE_CLARITY: TaskSpec(
        TASK_LANGUAGE_CLARITY,
        LanguageClarityEvaluator,
        LANGUAGE_CLARITY_EVALUATOR_PROMPT,
        LanguageClarityEvalSchema,
        run_language_clarity,
    ),
}


# ------------------------ Main ----------------------- #
def main() -> None:
    F.print_title("Starting Groq Native Error Analysis")
    F.print_success_message("See the output in assets/error_analysis/*")

    data_file_path = os.path.join(DATA_PATH, "eval_data.json")
    samples = F.load_json(data_file_path)

    os.makedirs(ERROR_ANALYSIS_PATH, exist_ok=True)

    for task_name in ERROR_ANALYSIS_TASKS:
        task = TASK_SPECS[task_name]
        F.print_subtitle(f"Analyzing: {task.name}")
        task_path = os.path.join(ERROR_ANALYSIS_PATH, task.name)
        os.makedirs(task_path, exist_ok=True)

        output_file_path = os.path.join(task_path, "output.txt")
        with open(output_file_path, mode="w", encoding="utf-8") as file:
            write_in_file(file, f"Starting Groq Native Error Analysis for task: {task.name}", TEXT_TITLE)

            errors: dict[str, int] = {}
            for model in MODELS:
                model_errors = 0
                response_format = get_response_format(model, task.schema, task.schema.__name__)

                print(f">> Model: {model}")
                write_in_file(file, f">> Model: {model}", TEXT_NORMAL)
                write_in_file(file, f">> Response Format: {response_format['type']}", TEXT_NORMAL)

                for cfg in MODELS_CFGS:
                    write_in_file(file, ">> Model Config:", TEXT_NORMAL)
                    write_in_file(file, cfg, TEXT_DICT, 1)
                    print(f">> CFG: {cfg}")

                    agent = task.agent_cls(
                        model_name=model,
                        system_prompt=task.prompt,
                        structured_response=task.schema,
                        **cfg,
                    )

                    for idx, sample in enumerate(samples, start=1):
                        write_in_file(file, f">> Sample #{idx}", TEXT_NORMAL)
                        print(f"\t>> Sample #{idx}")
                        model_errors += task.runner(file, agent, sample)

                errors[model] = model_errors
                print_sub(file, 5)

            print_sub(file, 100)
            write_in_file(file, "Num of Errors:", TEXT_NORMAL)
            write_in_file(file, errors, TEXT_DICT, 1)


if __name__ == "__main__":
    main()
