# ---------------------------------------------------------------------
# The Main Workflow
# ---------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

import json

# agents
from agents import JobToolsExtractor, ProposalToolsAnalyzer
from agents import JobRequirementsExtractor, JobRequirementsMatcher
from agents import JobKeyPointsExtractor, JobUnderstandingEvaluator
from agents import ExperienceEvidenceAgent
from agents import LanguageClarityEvaluator
from agents import SuperAgent

# schemas
from schemas import ExperienceEvidenceSchema
from schemas import JobToolResponse, ProposalToolsResponse
from schemas import JobKeyPointsSchema, JobUnderstandingEvalSchema
from schemas import ExtractedRequirementsSchema, RequirementCoverageSchema
from schemas import LanguageClarityEvalSchema
from schemas import SuperAgentResponse

# prompts
from prompts import JOB_TOOLS_EXTRACTION_PROMPT, PROPOSAL_TOOLS_EXTRACTION_PROMPT
from prompts import JOB_KEY_POINTS_EXTRACTION_PROMPT, JOB_UNDERSTANDING_EVALUATOR_PROMPT
from prompts import REQUIREMENT_EXTRACTOR_PROMPT, REQUIREMENT_MATCHER_PROMPT
from prompts import EXPERIENCE_EVIDENCE_PROMPT
from prompts import LANGUAGE_CLARITY_EVALUATOR_PROMPT
from prompts import SUPER_AGENT_SYSTEM_PROMPT

# data processing
from processing.tool_alignment_processing import get_final_tool_alignment_result
from processing.job_understanding_processing import get_final_job_understanding_result
from processing.requirement_coverage_processing import get_final_requirements_coverage_result
from processing.language_clarity_processing import get_final_language_clarity_result
from processing.experience_evidence import get_final_experience_evidence_result

# others
import os
from pathlib import Path
import helpers.config as CFG
import helpers.functional as F

# pipeline
# from pipeline_core import PrrPipeline

# ---------------------- CFG --------------------------
DATA_PATH = os.path.join(
    Path(__file__).parent,
    "assets",
    "data_examples"
)

TASK_TOOL_ALIGNMENT = "tools_alignment"
TASK_JOB_REQUIREMENTS = "requirements_coverage"
TASK_JOB_UNDERSTANDING = "job_understanding"
TASK_EXPERIENCE_EVIDENCE = "experience_evidence"
TASK_LANGUAGE_CLARITY = "language_clarity"
TASK_SUPER_AGENT = "super_agent"
TASK_FULL_PIPELINE = "full_pipeline"

TASKS = [
    TASK_TOOL_ALIGNMENT,
    TASK_JOB_REQUIREMENTS,
    TASK_JOB_UNDERSTANDING,
    TASK_EXPERIENCE_EVIDENCE,
    TASK_LANGUAGE_CLARITY,
    # TASK_SUPER_AGENT
    # TASK_FULL_PIPELINE
]

# ------------------------- Utils -----------------------

def get_task_agents(task: str):
    cfg = {
        "temperature": 0,
        "max_tokens": 1024
    }

    if task == TASK_TOOL_ALIGNMENT:
        return (
            JobToolsExtractor(
                model_name          = CFG.GROQ_LLAMA_8b,
                system_prompt       = JOB_TOOLS_EXTRACTION_PROMPT,
                structured_response = JobToolResponse,
                **cfg
            ),

            ProposalToolsAnalyzer(
                model_name          = CFG.GROQ_GPT_120b,
                system_prompt       = PROPOSAL_TOOLS_EXTRACTION_PROMPT,
                structured_response = ProposalToolsResponse,
            )
        )
    
    elif task == TASK_JOB_REQUIREMENTS:
        return (
            JobRequirementsExtractor(
                model_name          = CFG.GROQ_LLAMA_8b,
                system_prompt       = REQUIREMENT_EXTRACTOR_PROMPT,
                structured_response = ExtractedRequirementsSchema,
            ),

            JobRequirementsMatcher(
                model_name          = CFG.GROQ_GPT_120b,
                system_prompt       = REQUIREMENT_MATCHER_PROMPT,
                structured_response = RequirementCoverageSchema,
            )
        )
    
    elif task == TASK_JOB_UNDERSTANDING:
        return (
            JobKeyPointsExtractor(
                model_name          = CFG.GROQ_GPT_120b,
                system_prompt       = JOB_KEY_POINTS_EXTRACTION_PROMPT,
                structured_response = JobKeyPointsSchema,
            ),

            JobUnderstandingEvaluator(
                model_name          = CFG.GROQ_GPT_120b,
                system_prompt       = JOB_UNDERSTANDING_EVALUATOR_PROMPT,
                structured_response = JobUnderstandingEvalSchema,
            )
        )
    
    elif task == TASK_EXPERIENCE_EVIDENCE:
        return ExperienceEvidenceAgent(
            model_name          = CFG.GROQ_GPT_120b,
            system_prompt       = EXPERIENCE_EVIDENCE_PROMPT,
            structured_response = ExperienceEvidenceSchema,
        )
    
    elif task == TASK_LANGUAGE_CLARITY:
        return LanguageClarityEvaluator(
            model_name          = CFG.GROQ_LLAMA_8b,
            system_prompt       = LANGUAGE_CLARITY_EVALUATOR_PROMPT,
            structured_response = LanguageClarityEvalSchema,
        )

    else:
        return SuperAgent(
            model_name          = CFG.GROQ_GPT_120b,
            system_prompt       = SUPER_AGENT_SYSTEM_PROMPT,
            structured_response = SuperAgentResponse
        )



def get_task_data(task: str):
    if task == TASK_TOOL_ALIGNMENT:
        return F.load_json(
            file_path = os.path.join(DATA_PATH, "tools_alignment_samples.json")
        )

    if task == TASK_JOB_UNDERSTANDING:
        return F.load_json(
            file_path = os.path.join(DATA_PATH, "job_understanding_samples.json")
        )

    if task == TASK_JOB_REQUIREMENTS:
        return F.load_json(
            file_path = os.path.join(DATA_PATH, "requirement_coverage_samples.json")
        )

    if task == TASK_EXPERIENCE_EVIDENCE:
        return F.load_json(
            file_path = os.path.join(DATA_PATH, "experience_samples.json")
        )

    if task == TASK_LANGUAGE_CLARITY:
        return F.load_json(
            file_path = os.path.join(DATA_PATH, "language_clarity_samples.json")
        )
    
    return F.load_json(
        file_path = os.path.join(DATA_PATH, "super_agent_samples.json")
    )


def print_sep(n_sep = 3):
    print()
    print()
    print("-" * n_sep)
    print()
    print()

def test_tool_alignment():
    samples = get_task_data(TASK_TOOL_ALIGNMENT)
    agents  = get_task_agents(TASK_TOOL_ALIGNMENT)

    for idx, sample in enumerate(samples, start = 1):
        print(f">> Sample #{idx}\n")

        job_desc = sample["job_desc"]
        proposals = sample["proposals"]

        print(f">> Job Description:\n{job_desc}\n")

        job_analyzer_response = agents[0].invoke(job_desc)
        print(f"\t>> Job Analysis:\n")
        F.print_data(job_analyzer_response)

        for p_idx, proposal in enumerate(proposals, start = 1):
            print(f">> Proposal #{p_idx}:\n{proposal}\n")
            proposal_analyzer_response = agents[1].invoke(job_analyzer_response.tools, proposal)

            print("\t>> Proposal Analysis:\n")
            F.print_data(proposal_analyzer_response, 1)


            print(f"Final Results:\n")
            final_result = get_final_tool_alignment_result(proposal_analyzer_response)
            F.print_data(final_result, 1)
        
            print_sep()
    
def test_requirement_coverage():
    samples = get_task_data(TASK_JOB_REQUIREMENTS)
    agents  = get_task_agents(TASK_JOB_REQUIREMENTS)

    for idx, sample in enumerate(samples, start = 1):
        print(f">> Sample #{idx}\n")

        job_desc = sample["job_desc"]
        proposals = sample["proposals"]

        print(f">> Job Description:\n{job_desc}\n")

        job_analyzer_response = agents[0].invoke(job_desc)
        print(f"\t>> Job Analysis:\n")
        F.print_data(job_analyzer_response)

        for p_idx, proposal in enumerate(proposals, start = 1):
            print(f">> Proposal #{p_idx}:\n{proposal}\n")
            proposal_analyzer_response = agents[1].invoke(job_analyzer_response.requirements, proposal)

            print("\t>> Proposal Analysis:\n")
            F.print_data(proposal_analyzer_response, 1)


            print(f"\n\nFinal Results:\n")
            final_result = get_final_requirements_coverage_result(
                job_analyzer_response.requirements,
                final_coverage = proposal_analyzer_response
            )

            F.print_data(final_result, 1)
        
            print_sep()



def test_job_understanding():
    samples = get_task_data(TASK_JOB_UNDERSTANDING)
    agents  = get_task_agents(TASK_JOB_UNDERSTANDING)

    for idx, sample in enumerate(samples, start = 1):
        print(f">> Sample #{idx}\n")

        job_desc = sample["job_desc"]
        proposals = sample["proposals"]

        print(f">> Job Description:\n{job_desc}\n")

        job_analyzer_response = agents[0].invoke(job_desc)
        print(f"\t>> Job Analysis:\n")
        F.print_data(job_analyzer_response)

        for p_idx, proposal in enumerate(proposals, start = 1):
            print(f">> Proposal #{p_idx}:\n{proposal}\n")
            proposal_analyzer_response = agents[1].invoke(
                job_analyzer_response.core_problem, 
                job_analyzer_response.required_deliverables, 
                proposal
            )

            print("\t>> Proposal Analysis:\n")
            F.print_data(proposal_analyzer_response, 1)


            print(f"\n\nFinal Results:\n")
            final_result = get_final_job_understanding_result(
                proposal_analyzer_response,
            )

            F.print_data(final_result, 1)
        
            print_sep()


            
def test_experience_evidence():
    samples = get_task_data(TASK_EXPERIENCE_EVIDENCE)
    agent  = get_task_agents(TASK_EXPERIENCE_EVIDENCE)
    for idx, sample in enumerate(samples, start = 1):
        print(f">> Sample #{idx}\n")

        job_desc = sample['job_desc']
        proposals = sample["proposals"]

        print(f">> Job Description:\n{job_desc}\n")



        for p_idx, proposal in enumerate(proposals, start = 1):
            print(f">> Proposal #{p_idx}:\n{proposal}\n")

            response = agent.invoke(job_desc, proposal)

            print("\t>> Proposal Analysis:\n")
            F.print_data(response, 1)

            print(f"\n\nFinal Results:\n")
            final_result = get_final_experience_evidence_result(
                response,
            )

            F.print_data(final_result, 1)
            print_sep()
            


def test_language_clarity():
    samples = get_task_data(TASK_LANGUAGE_CLARITY)
    agent  = get_task_agents(TASK_LANGUAGE_CLARITY)
    proposals = samples["proposals"]

    for p_idx, proposal in enumerate(proposals, start = 1):
        print(f">> Proposal #{p_idx}:\n{proposal}\n")
        response = agent.invoke(proposal)

        print("\t>> Proposal Analysis:\n")
        F.print_data(response, 1)

        
        print(f"\n\nFinal Results:\n")
        final_result = get_final_language_clarity_result(
            response,
            proposal
        )

        F.print_data(final_result, 1)

        print_sep()

def test_task(task: str):
    F.print_subtitle(task.title())
    
    if task == TASK_TOOL_ALIGNMENT:
        test_tool_alignment()
    
    elif task == TASK_JOB_REQUIREMENTS:
        test_requirement_coverage()

    elif task == TASK_JOB_UNDERSTANDING:
        test_job_understanding()

    elif task == TASK_LANGUAGE_CLARITY:
        test_language_clarity()

    elif task == TASK_EXPERIENCE_EVIDENCE:
        test_experience_evidence()

    


if __name__ == "__main__":
    F.print_title("1.0 Starting the APP")

    for task in TASKS:
        test_task(task)

        print_sep(10)


    