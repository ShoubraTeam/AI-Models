# ---------------------------------------------------------------------
# The Main Workflow
# ---------------------------------------------------------------------


# agents
from agents import JobToolsExtractor

# schemas
from schemas import JobToolResponse

# prompts
from prompts import JOB_TOOLS_EXTRACTION_PROMPT

# others
import helpers.config as CFG
import helpers.functional as F
from dotenv import load_dotenv


load_dotenv()

if __name__ == "__main__":
    F.print_title("1.0 Starting the APP")

    
    print("- Waking Up Agents...")

    job_tool_extractor = JobToolsExtractor(
        model_name = CFG.GROQ_LLAMA_8b,
        system_prompt = JOB_TOOLS_EXTRACTION_PROMPT,
        structured_response = JobToolResponse,
        model_provider = CFG.PROVIDER_GROQ,
        temperature = CFG.MODELS_CFG["tools_alignment_pipeline"]["job_tools_extractor_temperature"],
        max_tokens = CFG.MODELS_CFG["tools_alignment_pipeline"]["job_tools_extractor_max_tokens"],
    )


    print("- Loading data...")
    job_description = """I’m ready to launch a small business website whose sole purpose is to give visitors clear, trustworthy information about our company. I already know an “About Us” page is essential, and I’m happy to hear your recommendations on whether a Home, Contact, or any other page would improve navigation and credibility.

You’ll take the project from zero to live, handling design, development, hosting configuration, and fundamental search-engine optimisation so the site is easy to find the moment it goes online. A lightweight, easily editable CMS such as WordPress, Webflow, or a similar platform is preferred for ongoing updates.

Deliverables
• Fully responsive website built on the agreed CMS
• At least one complete “About Us” page, with room for expansion
• On-page SEO: keyword research, meta titles/descriptions, alt tags, schema where appropriate
• XML sitemap and robots.txt, submitted to Google Search Console
• Basic performance tuning to meet core-web-vital standards
• Handover of all credentials, theme files, and a brief how-to guide for future edits

Acceptance criteria
The site must load cleanly on mobile and desktop, score green on Google’s PageSpeed Insights, and be indexed in Google with no critical errors.

If this sounds like your wheelhouse, tell me how you’ll approach the build and SEO rollout, along with a rough timeline.
"""

    F.print_title("2.0 Testing the Agent")
    response = job_tool_extractor.invoke(
        input = job_description
    )


    F.print_title("3.0 Printing the Output")
    F.print_structured_response(response)
    
    
    is_agent_op_ok = job_tool_extractor.validate_agent_output(agent_output = response)
    print(is_agent_op_ok)


