# ---------------------------------------------------------------------
# The Main Workflow
# ---------------------------------------------------------------------

# agents
from agents import JobToolsExtractor
from agents import ProposalToolsAnalyzer
from agents.requirement_coverage.job_requirements_extractor import JobRequirementsExtractor
from agents.requirement_coverage.job_requirements_matcher import JobRequirementsMatcher
from agents.job_understanding.job_understanding_agent import JobUnderstandingAgent

# schemas
from schemas import JobToolResponse, ProposalToolsResponse
from schemas.requirement_coverage.requirement_extraction_schema import ExtractedRequirementsSchema
from schemas.requirement_coverage.requirement_coverage_schema import RequirementCoverageSchema


# prompts
from prompts import JOB_TOOLS_EXTRACTION_PROMPT, PROPOSAL_TOOLS_EXTRACTION_PROMPT

# data processing
from processing.tool_alignment_processing import format_ip_for_proposal_tools_analyzer, calc_tools_alignment_score
from processing.job_understanding_processing import calc_job_understanding_result

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
    F.print_title("1.0 Starting the APP")

    print("- Waking Up Agents...")

    # -----------------------------------------------------------------
    # Tools Alignment Agents Initialization
    # -----------------------------------------------------------------
    job_tool_extractor = JobToolsExtractor(
        model_name          = CFG.GROQ_LLAMA_8b,
        system_prompt       = JOB_TOOLS_EXTRACTION_PROMPT,
        structured_response = JobToolResponse,
        model_provider      = CFG.PROVIDER_GROQ,
        temperature         = CFG.MODELS_CFG["tools_alignment_pipeline"]["job_tools_extractor_temperature"],
        max_tokens          = CFG.MODELS_CFG["tools_alignment_pipeline"]["job_tools_extractor_max_tokens"],
    )

    proposal_tools_analyzer = ProposalToolsAnalyzer(
        model_name          = CFG.GROQ_LLAMA_70b,
        system_prompt       = PROPOSAL_TOOLS_EXTRACTION_PROMPT,
        structured_response = ProposalToolsResponse,
        model_provider      = CFG.PROVIDER_GROQ,
        temperature         = CFG.MODELS_CFG["tools_alignment_pipeline"]["proposal_tools_analyzer_temperature"],
        max_tokens          = CFG.MODELS_CFG["tools_alignment_pipeline"]["proposal_tools_analyzer_max_tokens"],
    )

    # -----------------------------------------------------------------
    # Requirement Coverage Agents Initialization (Direct Pipeline)
    # -----------------------------------------------------------------
    job_requirements_extractor = JobRequirementsExtractor(
        model_name = CFG.GROQ_LLAMA_8b
    )

    job_requirements_matcher = JobRequirementsMatcher(
        model_name = CFG.GROQ_LLAMA_70b
    )

    # -----------------------------------------------------------------
    # Job Understanding Agent Initialization
    # -----------------------------------------------------------------
    job_understanding_agent = JobUnderstandingAgent()


    print("- Loading data...")
    # Loading Tools Alignment Data
    tools_alignment_data_samples = F.load_json(
        file_path = os.path.join(DATA_PATH, "tools_alignment_tools.json")
    )

    # Loading Requirement Coverage Data
    requirement_data_samples = F.load_json(
        file_path = os.path.join(DATA_PATH, "requirement_coverage_samples.json")
    )

    # Loading Job Understanding Data
    job_understanding_data_samples = F.load_json(
        file_path = os.path.join(DATA_PATH, "job_understanding_samples.json")
    )


    # ==================================================================
    # 2.0 Testing Tools Alignment Agents
    # ==================================================================
    F.print_title("2.0 Testing Tools Alignment Agents")
    tools_sample = tools_alignment_data_samples[0]

    job_description_tools = tools_sample["job_desc"]
    proposal_tools        = tools_sample["proposal1"]

    print("- Extracting Job Tools...")
    job_tools_response = job_tool_extractor.invoke(
        input = job_description_tools
    )
    F.print_structured_response(job_tools_response)

    print("- Analyzing Proposal Tools...")
    prepared_analysis_tool_ip = format_ip_for_proposal_tools_analyzer(
        job_tools = job_tools_response.tools,
        proposal  = proposal_tools
    )

    proposal_tools_analysis = proposal_tools_analyzer.invoke(
        input = prepared_analysis_tool_ip
    )
    F.print_structured_response(proposal_tools_analysis)

    print("- Tool Alignment Score...")
    print(calc_tools_alignment_score(proposal_tools_analysis))


    # ==================================================================
    # 3.0 Testing Requirement Coverage Agents (Direct & Flattened)
    # ==================================================================
    F.print_title("3.0 Testing Requirement Coverage Agents Directly")
    req_sample = requirement_data_samples[0]

    print("- Step 1: Extracting Requirements with IDs...")
    extracted_data = job_requirements_extractor.invoke(
        input = req_sample["job_desc"]
    )
    F.print_structured_response(extracted_data)

    formatted_matcher_input = f"""
    Extracted Requirements (with IDs):
    {extracted_data.model_dump_json(indent=2)}

    Freelancer Proposal Text:
    {req_sample["proposal2"]}
    """

    print("- Step 2: Running Semantic Matching via IDs...")
    final_coverage = job_requirements_matcher.invoke(
        job_requirements = extracted_data.requirements,
        proposal_text    = req_sample["proposal2"]
    )
    F.print_structured_response(final_coverage)

    total_reqs = len(extracted_data.requirements)
    covered_reqs = len(final_coverage.requirements_covered_ids)
    
    calculated_score = covered_reqs / total_reqs if total_reqs > 0 else 0.0
    
    print("\n- Calculated Requirement Coverage Score (Manual):")
    print(f"  Score => {calculated_score}")

    # ==================================================================
    # 4.0 Testing Job Understanding Agent
    # ==================================================================
    F.print_title("4.0 Testing Job Understanding Agent")

    for i, proposal_key in enumerate(["proposal1", "proposal2", "proposal3"], start=1):
        ju_sample = job_understanding_data_samples[0]

        F.print_title(f"4.{i} Proposal #{i}")
        print("- Running Job Understanding Analysis...")

        ju_response = job_understanding_agent.invoke(
            job_description = ju_sample["job_desc"],
            proposal_text   = ju_sample[proposal_key]
        )
        F.print_structured_response(ju_response)

        print("\n- Job Understanding Result...")
        ju_result = calc_job_understanding_result(ju_response)
        for key, value in ju_result.items():
            print(f"  {key} => {value}")

        print()