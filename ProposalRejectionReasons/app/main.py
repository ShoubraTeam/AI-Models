# ---------------------------------------------------------------------
# The Main Workflow
# ---------------------------------------------------------------------

# agents
from agents import JobToolsExtractor
from agents import ProposalToolsAnalyzer
from agents.requirement_coverage.requirement_coverage_agent import RequirementCoverageAgent
from agents.job_understanding.job_understanding_agent import JobUnderstandingAgent

# schemas
from schemas import JobToolResponse, ProposalToolsResponse

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
    # Requirement Coverage Agent Initialization
    # -----------------------------------------------------------------
    requirement_agent = RequirementCoverageAgent()

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
    # 3.0 Testing Requirement Coverage Agent
    # ==================================================================
    F.print_title("3.0 Testing Requirement Coverage Agent")
    req_sample = requirement_data_samples[0]

    print("- Running Requirement Coverage Analysis...")
    req_response = requirement_agent.invoke(
        job_description = req_sample["job_desc"],
        proposal_text   = req_sample["proposal1"]
    )
    F.print_structured_response(req_response)


# ==================================================================
# 4.0 Testing Job Understanding
# ==================================================================
F.print_title("4.0 Testing Job Understanding")

ju_sample       = job_understanding_data_samples[0]
job_desc_ju     = ju_sample["job_desc"]

# Initialize sub-agents independently (no orchestrator class)
extractor  = JobKeyPointsExtractor()
evaluator  = JobUnderstandingEvaluator()

for i, proposal_key in enumerate(["proposal1", "proposal2", "proposal3"], start=1):
    proposal_text = ju_sample[proposal_key]

    F.print_title(f"4.{i} Proposal #{i}")

    # -----------------------------------------------------------------
    # Step 1 — Extract key points (independently testable)
    # -----------------------------------------------------------------
    print("- Extracting Job Key Points...")
    extraction = extractor.invoke(input=job_desc_ju)
    F.print_structured_response(extraction)

    # -----------------------------------------------------------------
    # Step 2 — LLM evaluation (independently testable)
    # -----------------------------------------------------------------
    print("- Running LLM Evaluation (3 questions only)...")
    llm_eval = evaluator.invoke(
        core_problem          = extraction.core_problem,
        required_deliverables = extraction.required_deliverables,
        proposal_text         = proposal_text
    )
    F.print_structured_response(llm_eval)

    # -----------------------------------------------------------------
    # Step 3 — Metrics + scoring in processing layer (no LLM)
    # -----------------------------------------------------------------
    print("- Computing Final Result (keyword metrics + scoring)...")
    result = calc_job_understanding_result(
        extraction    = extraction,
        llm_eval      = llm_eval,
        proposal_text = proposal_text
    )
    for key, value in result.items():
        print(f"  {key} => {value}")

    print()
