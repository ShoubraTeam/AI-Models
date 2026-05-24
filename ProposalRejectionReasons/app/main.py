# ---------------------------------------------------------------------
# The Main Workflow
# ---------------------------------------------------------------------

# agents
from agents import JobToolsExtractor, ProposalToolsAnalyzer
from agents import JobRequirementsExtractor, JobRequirementsMatcher
from agents import JobKeyPointsExtractor, JobUnderstandingEvaluator
from agents import ExperienceEvidenceAgent


# schemas
from schemas import ExperienceEvidenceSchema
from schemas import JobToolResponse, ProposalToolsResponse
from schemas import JobKeyPointsSchema, JobUnderstandingEvalSchema
from schemas import ExtractedRequirementsSchema, RequirementCoverageSchema

# prompts
from prompts import JOB_TOOLS_EXTRACTION_PROMPT, PROPOSAL_TOOLS_EXTRACTION_PROMPT
from prompts import JOB_KEY_POINTS_EXTRACTION_PROMPT, JOB_UNDERSTANDING_EVALUATOR_PROMPT
from prompts import REQUIREMENT_EXTRACTOR_PROMPT, REQUIREMENT_MATCHER_PROMPT
from prompts import EXPERIENCE_EVIDENCE_PROMPT

# data processing
from processing.tool_alignment_processing import format_ip_for_proposal_tools_analyzer, calc_tools_alignment_score
from processing.job_understanding_processing import calc_job_understanding_result
from processing.requirement_coverage_processing import calc_requirement_coverage_score 

# others
import os
from pathlib import Path
import helpers.config as CFG
import helpers.functional as F
from dotenv import load_dotenv



load_dotenv()

DATA_PATH = os.path.join(
    Path(__file__).parent,
    "data_examples"
)

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Agents & Data Initialization
    # -----------------------------------------------------------------
    F.print_title("1.0 Starting the APP")

    F.print_subtitle("Wake up Agents")

    try:
        print("\t>> Tools Alignment Agents")
        job_tool_extractor = JobToolsExtractor(
            model_name          = CFG.GROQ_LLAMA_8b,
            system_prompt       = JOB_TOOLS_EXTRACTION_PROMPT,
            structured_response = JobToolResponse,
        )

        proposal_tools_analyzer = ProposalToolsAnalyzer(
            model_name          = CFG.GROQ_LLAMA_70b,
            system_prompt       = PROPOSAL_TOOLS_EXTRACTION_PROMPT,
            structured_response = ProposalToolsResponse,
        )

        print("\t>> Job Understanding Agents")
        job_key_points_extractor  = JobKeyPointsExtractor(
            model_name          = CFG.GROQ_LLAMA_70b,
            system_prompt       = JOB_KEY_POINTS_EXTRACTION_PROMPT,
            structured_response = JobKeyPointsSchema,
        )

        job_understanding_evaluator  = JobUnderstandingEvaluator(
            model_name          = CFG.GROQ_LLAMA_70b,
            system_prompt       = JOB_UNDERSTANDING_EVALUATOR_PROMPT,
            structured_response = JobUnderstandingEvalSchema,
        )

        print("\t>> Requirement Coverage Agents")
        requirement_extractor  = JobRequirementsExtractor(
            model_name          = CFG.GROQ_LLAMA_8b,
            system_prompt       = REQUIREMENT_EXTRACTOR_PROMPT,
            structured_response = ExtractedRequirementsSchema,
        )

        requirement_matcher  = JobRequirementsMatcher(
            model_name          = CFG.GROQ_LLAMA_70b,
            system_prompt       = REQUIREMENT_MATCHER_PROMPT,
            structured_response = RequirementCoverageSchema,
        )
        print("\t>> Evidence of Experience Agent")  
        experience_evidence_agent = ExperienceEvidenceAgent(
            model_name          = CFG.GROQ_LLAMA_70b,
            system_prompt       = EXPERIENCE_EVIDENCE_PROMPT,
            structured_response = ExperienceEvidenceSchema,
        )

        F.print_success_message("Agents Loaded Successfully")
    
    except Exception as e:
        F.print_error_message("Error While Loading Agents")
        F.print_error_message(e)
        exit()


    F.print_subtitle("Loading Data")
    
    try:
        print("\t>> Tools Alignment Data")
        tools_alignment_data_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "tools_alignment_tools.json")
        )

        print("\t>> Job Understanding Data")
        job_understanding_data_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "job_understanding_samples.json")
        )

        print("\t>> Requirement Coverage Data")
        requirement_data_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "requirement_coverage_samples.json")
        )
        print("\t>> Evidence of Experience Data")
        experience_data_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "experience_samples.json")
        )
        F.print_success_message("Data Loaded Successfully")
    
    except Exception as e:
        F.print_error_message("Error While Loading Data")
        F.print_error_message(e)
        exit()

    # ==================================================================
    # 2.0 Testing Agents
    # ==================================================================
    F.print_title("2.0 Testing Agents")

    # --------------------------------------------
    F.print_subtitle("Tools Alignment")

    tools_sample = tools_alignment_data_samples[0]
    job_desc  = tools_sample["job_desc"]
    proposals = tools_sample["proposals"]

    print("\t>> Extracting Job Tools")
    job_tools_response = job_tool_extractor.invoke(input = job_desc)
    F.print_structured_response(job_tools_response)

    print("\t>> Analyzing Proposal Tools")
    for idx, proposal in enumerate(proposals, start = 1):
        print(f"--- Analyzing Proposal {idx} ---")
        prepared_analysis_tool_ip = format_ip_for_proposal_tools_analyzer(
            job_tools = job_tools_response.tools,
            proposal  = proposal
        )

        proposal_tools_analysis = proposal_tools_analyzer.invoke(
            input = prepared_analysis_tool_ip
        )

        F.print_structured_response(proposal_tools_analysis)
        print()

        print("\t>> Tools Alignment Score: ", end = "")
        print(calc_tools_alignment_score(proposal_tools_analysis))


    # --------------------------------------------
    F.print_subtitle("Requirement Coverage")
    job_desc = requirement_data_samples[0]["job_desc"]
    proposals = requirement_data_samples[0]["proposals"]
    
    print("\t>> Extracting Job Requirements")
    extracted_data = requirement_extractor.invoke(input = job_desc)
    F.print_structured_response(extracted_data)

    print("\t>> Evaluating Requirements in Proposal")
    for idx, proposal in enumerate(proposals, start = 1):
        print(f"--- Analyzing Proposal {idx} ---")
        requirements_matching = requirement_matcher.invoke(
            job_requirements = extracted_data.requirements, 
            proposal_text = proposal
        )

        F.print_structured_response(requirements_matching)
        
        calculated_score = calc_requirement_coverage_score(
            extracted_requirements = extracted_data.requirements,
            final_coverage = requirements_matching
        )
        
        print(f"\t>> Calculated Requirement Coverage Score (Weighted): {calculated_score}\n")


    # --------------------------------------------
    F.print_subtitle("Job Understanding")

    job_desc  = job_understanding_data_samples[0]["job_desc"]
    proposals = job_understanding_data_samples[0]["proposals"]
    
    print("\t>> Extracting Job Key Points")
    job_key_points = job_key_points_extractor.invoke(input = job_desc)
    F.print_structured_response(job_key_points)

    print("\t>> Evaluating Proposal Quality")
    for idx, proposal in enumerate(proposals, start = 1):
        print(f"--- Analyzing Proposal {idx} ---")
        understanding_evaluation = job_understanding_evaluator.invoke(
            core_problem = job_key_points.core_problem,
            required_deliverables = job_key_points.required_deliverables,
            proposal_text = proposal
        )

        F.print_structured_response(understanding_evaluation)
        print()

        print("Final Result (keyword metrics + scoring): ")
        result = calc_job_understanding_result(
            extraction    = job_key_points,
            llm_eval      = understanding_evaluation,
            proposal_text = proposal
        )
        for key, value in result.items():
            print(f"  {key} => {value}")

        print()

    # --------------------------------------------
    F.print_subtitle("Evidence of Experience")  

    exp_job_desc  = experience_data_samples[1]["job_desc"]
    exp_proposals = experience_data_samples[1]["proposals"]

    print("\t>> Auditing Proposals for Past Experience Evidence")
    for idx, proposal in enumerate(exp_proposals, start = 1):
        print(f"--- Analyzing Proposal {idx} ---")
        experience_audit = experience_evidence_agent.invoke(
            job_desc      = exp_job_desc,
            proposal_text = proposal
        )

        F.print_structured_response(experience_audit)
        print()