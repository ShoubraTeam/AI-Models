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
import helpers.config as CFG
import helpers.functional as F
from dotenv import load_dotenv


load_dotenv()

DATA_PATH = os.path.join(
    Path(__file__).parent.parent,
    "eval_data"
)

if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Agents & Data Initialization
    # -----------------------------------------------------------------
    F.print_title("1.0 Starting the APP")

    F.print_subtitle("Wake up Agents")

    try:
        # print("\t>> Tools Alignment Agents")
        # job_tool_extractor = JobToolsExtractor(
        #     model_name          = CFG.GROQ_LLAMA_8b,
        #     system_prompt       = JOB_TOOLS_EXTRACTION_PROMPT,
        #     structured_response = JobToolResponse,
        # )

        # proposal_tools_analyzer = ProposalToolsAnalyzer(
        #     model_name          = CFG.GROQ_LLAMA_70b,
        #     system_prompt       = PROPOSAL_TOOLS_EXTRACTION_PROMPT,
        #     structured_response = ProposalToolsResponse,
        # )

        # print("\t>> Job Understanding Agents")
        # job_key_points_extractor = JobKeyPointsExtractor(
        #     model_name          = CFG.GROQ_LLAMA_70b,
        #     system_prompt       = JOB_KEY_POINTS_EXTRACTION_PROMPT,
        #     structured_response = JobKeyPointsSchema,
        # )

        # job_understanding_evaluator = JobUnderstandingEvaluator(
        #     model_name          = CFG.GROQ_LLAMA_70b,
        #     system_prompt       = JOB_UNDERSTANDING_EVALUATOR_PROMPT,
        #     structured_response = JobUnderstandingEvalSchema,
        # )

        print("\t>> Requirement Coverage Agents")
        requirement_extractor = JobRequirementsExtractor(
            model_name          = CFG.GROQ_GPT_120b,
            system_prompt       = REQUIREMENT_EXTRACTOR_PROMPT,
            structured_response = ExtractedRequirementsSchema,
        )

        requirement_matcher = JobRequirementsMatcher(
            model_name          = CFG.GROQ_LLAMA_70b,
            system_prompt       = REQUIREMENT_MATCHER_PROMPT,
            structured_response = RequirementCoverageSchema,
        )

        print("\t>> Evidence of Experience Agent")
        experience_evidence_agent = ExperienceEvidenceAgent(
            model_name          = CFG.GROQ_GPT_120b,
            system_prompt       = EXPERIENCE_EVIDENCE_PROMPT,
            structured_response = ExperienceEvidenceSchema,
        )

        # print("\t>> Language Clarity Agent")
        # language_clarity_evaluator = LanguageClarityEvaluator(
        #     model_name          = CFG.GROQ_LLAMA_70b,
        #     system_prompt       = LANGUAGE_CLARITY_EVALUATOR_PROMPT,
        #     structured_response = LanguageClarityEvalSchema,
        # )

        # F.print_success_message("Agents Loaded Successfully")

    except Exception as e:
        F.print_error_message("Error While Loading Agents")
        F.print_error_message(e)
        exit()


    # F.print_subtitle("Loading Data")

    # try:
    #     print("\t>> Tools Alignment Data")
    #     tools_alignment_data_samples = F.load_json(
    #         file_path = os.path.join(DATA_PATH, "tools_alignment_tools.json")
    #     )

    #     print("\t>> Job Understanding Data")
    #     job_understanding_data_samples = F.load_json(
    #         file_path = os.path.join(DATA_PATH, "job_understanding_samples.json")
    #     )

    #     print("\t>> Requirement Coverage Data")
    #     requirement_data_samples = F.load_json(
    #         file_path = os.path.join(DATA_PATH, "requirement_coverage_samples.json")
    #     )

    #     print("\t>> Evidence of Experience Data")
    #     experience_data_samples = F.load_json(
    #         file_path = os.path.join(DATA_PATH, "eval_data.json")
    #     )

    #     print("\t>> Language Clarity Data")
    #     language_clarity_data_samples = F.load_json(
    #         file_path = os.path.join(DATA_PATH, "language_clarity_samples.json")
    #     )

    #     F.print_success_message("Data Loaded Successfully")

    # except Exception as e:
    #     F.print_error_message("Error While Loading Data")
    #     F.print_error_message(e)
    #     exit()


    # ==================================================================
    # 2.0 Testing Agents
    # ==================================================================
    # F.print_title("2.0 Testing Agents")

    
    # F.print_subtitle("Tools Alignment")

    # tools_sample = tools_alignment_data_samples[0]
    # job_desc     = tools_sample["job_desc"]
    # proposals    = tools_sample["proposals"]

    # print("\t>> Extracting Job Tools")
    # job_tools_response = job_tool_extractor.invoke(input=job_desc)
    # F.print_structured_response(job_tools_response)

    # print("\t>> Analyzing Proposal Tools")
    # for idx, proposal in enumerate(proposals, start=1):
    #     print(f"--- Analyzing Proposal {idx} ---")
    #     prepared_analysis_tool_ip = format_ip_for_proposal_tools_analyzer(
    #         job_tools = job_tools_response.tools,
    #         proposal  = proposal
    #     )
    #     proposal_tools_analysis = proposal_tools_analyzer.invoke(
    #         input = prepared_analysis_tool_ip
    #     )
    #     F.print_structured_response(proposal_tools_analysis)
    #     print()
    #     print(">> Tools Alignment Score: ", end="")
    #     print(calc_tools_alignment_score(proposal_tools_analysis))
    #     print()

    # --------------------------------------------
    # F.print_subtitle("Requirement Coverage")

    # job_desc  = requirement_data_samples[0]["job_desc"]
    # proposals = requirement_data_samples[0]["proposals"]

    # print("\t>> Extracting Job Requirements")
    # extracted_data = requirement_extractor.invoke(input=job_desc)
    # F.print_structured_response(extracted_data)

    # print("\t>> Evaluating Requirements in Proposal")
    # for idx, proposal in enumerate(proposals, start=1):
    #     print(f"--- Analyzing Proposal {idx} ---")
    #     requirements_matching = requirement_matcher.invoke(
    #         job_requirements = extracted_data.requirements,
    #         proposal_text    = proposal
    #     )
    #     print("\t>> LLM Raw Matcher Response:")
    #     F.print_structured_response(requirements_matching)
        
    #     final_result = calc_requirement_coverage_score(
    #         extracted_requirements = extracted_data.requirements,
    #         final_coverage         = requirements_matching
    #     )
        
    #     print("\t>> Final Standardized Subagent Result:")
    #     print(f"  score              => {final_result.score}")
    #     print(f"  accepted           => {final_result.accepted}")
    #     print(f"  summary            => {final_result.summary}")
    #     print(f"  acceptance_reasons => {final_result.acceptance_reasons}")
    #     print(f"  rejection_reasons  => {final_result.rejection_reasons}")
    #     print()
    # # --------------------------------------------
    # F.print_subtitle("Job Understanding")

    # job_desc  = job_understanding_data_samples[0]["job_desc"]
    # proposals = job_understanding_data_samples[0]["proposals"]

    # print("\t>> Extracting Job Key Points")
    # job_key_points = job_key_points_extractor.invoke(input=job_desc)
    # F.print_structured_response(job_key_points)

    # print("\t>> Evaluating Proposal Quality")
    # for idx, proposal in enumerate(proposals, start=1):
    #     print(f"--- Analyzing Proposal {idx} ---")
    #     understanding_evaluation = job_understanding_evaluator.invoke(
    #         core_problem          = job_key_points.core_problem,
    #         required_deliverables = job_key_points.required_deliverables,
    #         proposal_text         = proposal
    #     )
    #     F.print_structured_response(understanding_evaluation)

    #     print("Final Result (keyword metrics + scoring): ")
    #     result = calc_job_understanding_result(
    #         extraction    = job_key_points,
    #         llm_eval      = understanding_evaluation,
    #         proposal_text = proposal
    #     )
    #     for key, value in result.items():
    #         print(f"  {key} => {value}")
    #     print()

    # # --------------------------------------------
    # F.print_subtitle("Language Clarity")

    # lc_sample = language_clarity_data_samples[0]
    # proposals = lc_sample["proposals"]

    # print("\t>> Evaluating Language Clarity")
    # for idx, proposal in enumerate(proposals, start=1):
    #     print(f"--- Analyzing Proposal {idx} ---")
    #     llm_eval = language_clarity_evaluator.invoke(proposal_text=proposal)
    #     F.print_structured_response(llm_eval)

    #     print("Final Result (text metrics + scoring): ")
    #     result = calc_language_clarity_result(
    #         llm_eval      = llm_eval,
    #         proposal_text = proposal
    #     )
    #     for key, value in result.items():
    #         print(f"  {key} => {value}")
    #     print()

    # --------------------------------------------
    # --------------------------------------------
    # F.print_subtitle("Evidence of Experience")

    # exp_job_desc  = experience_data_samples[1]["job_desc"]
    # exp_proposals = experience_data_samples[1]["proposals"]

    # print("\t>> Auditing Proposals for Past Experience Evidence")
    # for idx, proposal in enumerate(exp_proposals, start=1):
    #     print(f"--- Analyzing Proposal {idx} ---")
        
    #     experience_audit = experience_evidence_agent.invoke(
    #         job_desc      = exp_job_desc,
    #         proposal_text = proposal
    #     )
    #     print("\t>> LLM Raw Audit Response:")
    #     F.print_structured_response(experience_audit)

    #     print("\t>> Final Standardized Subagent Result:")
    #     final_result = calc_experience_evidence_result(llm_audit=experience_audit)
        
    #     print(f"  score              => {final_result.score}")
    #     print(f"  accepted           => {final_result.accepted}")
    #     print(f"  summary            => {final_result.summary}")
    #     print(f"  acceptance_reasons => {final_result.acceptance_reasons}")
    #     print(f"  rejection_reasons  => {final_result.rejection_reasons}")
    #     print()


    # ==================================================================
    # 3.0 Batch Evaluation Testing
    # ==================================================================
    
    F.print_title("3.0 Batch Evaluation Testing")
    requirement_data_samples = F.load_json(
        file_path = os.path.join(DATA_PATH, "eval_data.json")
    )

    F.print_subtitle("Evaluating: Job Requirements Extractor")
    req_extractor_eval_data = []
    for sample in requirement_data_samples:
         job_data = sample.get("job_data", {}) if isinstance(sample.get("job_data"), dict) else {}
         requirements = job_data.get("requirements", []) if job_data else sample.get("requirements", [])
         
         req_extractor_eval_data.append({
             "desc": sample.get("job_desc", ""),
             "requirements": requirements
         })
     
    extractor_metrics = requirement_extractor.evaluate(eval_data=req_extractor_eval_data)
    print(">> Requirements Extractor Global Metrics:")
    for metric_name, values in extractor_metrics.items():
         if metric_name != "agent_response":
              avg_val = sum(values) / len(values) if values else 0.0
              print(f"   Average {metric_name} => {round(avg_val, 4)}")
    print()

    F.print_subtitle("Evaluating: Job Requirements Matcher")
    
    matcher_metrics = requirement_matcher.evaluate(eval_data=requirement_data_samples)
    
    print(">> Requirements Matcher Global Metrics:")
    for metric_name, value in matcher_metrics.items():
         print(f"   {metric_name} => {value}")
    print()
    

    experience_data_samples = F.load_json(
            file_path = os.path.join(DATA_PATH, "eval_data.json")
        )
    F.print_subtitle("Evaluating: Experience Evidence Agent")
    experience_eval_data = []
    
    for sample in experience_data_samples:
        cleaned_proposals = []
        
        for p in sample.get("proposals", []):
            if isinstance(p, dict):
                proposal_text = p.get("proposal", "")
                true_has_evidence = p.get("has_evidence", False)
                true_projects = p.get("true_projects", [])
            else:
                proposal_text = str(p)
                true_has_evidence = False
                true_projects = []

            cleaned_proposals.append({
                "proposal": proposal_text,
                "has_evidence": true_has_evidence,
                "true_projects": true_projects
            })

        experience_eval_data.append({
            "job_desc": sample.get("job_desc", ""),
            "proposals": cleaned_proposals
        })

    if experience_eval_data:
        experience_metrics = experience_evidence_agent.evaluate(eval_data=experience_eval_data)
        print(">> Experience Evidence Agent Global Metrics:")
        for metric_name, value in experience_metrics.items():
            print(f"  {metric_name} => {value}")
    print()