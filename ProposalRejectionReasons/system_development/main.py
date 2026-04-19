 # -----------------------------------------------------------------------------------
# The Main Workflow
# -----------------------------------------------------------------------------------



import src.utils.config as CFG
from src.agents.tool_alignment_agent import ToolAlignmentAgent
from src.agents.structured_response import ToolsAlignment
from langchain.tools import tool
from dotenv import load_dotenv
import src.utils.functional as F

load_dotenv()

def format_job_proposal(job_desc: str, proposal: str):
    """
    Formatting the job_desc & the proposal to prepare them for the LLM
    """
    separator = 100 * '='
    formatted = f"Job Description:\n{job_desc}\n{separator}\nProposal:\n{proposal}"
    return formatted

# testing ToolAlignment
tools_alignment_subagent = ToolAlignmentAgent(
    model_name = CFG.LLAMA_70B,
    model_provider = "groq",
    system_prompt = CFG.TOOLS_ALIGNMENT_PROMPT,
    response_format = ToolsAlignment,
    tools = [],
    temperature = 0.1,
    max_tokens = 512,
)

job_desc = f"""We are seeking a skilled AI Engineer to design and develop an intelligent system that evaluates freelancer proposals against job descriptions and provides structured feedback and scoring.

Responsibilities:

Develop an AI system to analyze and compare job descriptions with freelancer proposals
Design and implement scoring mechanisms for proposal quality and relevance
Build modules to evaluate requirement coverage, skill and tool alignment, clarity and language quality, and relevance of past experience
Implement NLP pipelines for text preprocessing and semantic analysis
Develop a feedback generation engine to suggest improvements
Create a scalable and modular architecture (preferably multi-agent or component-based)
Build API endpoints for system interaction and result delivery
Test and validate model performance using real-world examples
Document code, architecture, and usage instructions

Tools & Technologies:

Python
PyTorch, Scikit-learn, TensorFlow (optional)
NLP libraries such as NLTK, spaCy, Sentence Transformers
FastAPI or Flask
PostgreSQL or MongoDB and optionally vector databases like FAISS or Pinecone
Git and GitHub
Optional: Docker

Requirements:

Proven experience in AI, machine learning, and NLP projects
Strong understanding of text similarity, embeddings, and semantic analysis
Experience building or fine-tuning machine learning models
Ability to design scalable architectures
Experience with REST APIs and backend development
Strong analytical and problem-solving skills
Ability to write clean, maintainable, and well-documented code
"""

proposal = f"""
I am very interested in your project and confident in my ability to build a robust AI system that evaluates freelancer proposals and provides actionable insights.

My Approach:
I will design a modular AI system that analyzes the relationship between job descriptions and proposals across multiple dimensions. This includes semantic similarity using embeddings, requirement extraction and matching, skill and tool alignment detection, experience relevance evaluation, and language clarity analysis. Each component will be built as an independent module to ensure scalability and flexibility.

Skills & Tools:

Python for core development
PyTorch and Scikit-learn for model building
NLP tools such as spaCy, NLTK, and Sentence Transformers
Embedding models and similarity techniques such as cosine similarity
FastAPI for building scalable APIs
FAISS or similar vector databases for efficient search
Git for version control
Optional use of Docker for deployment

Deliverables:

A fully functional AI evaluation system
Modular and scalable architecture
API endpoints for evaluating proposals
A structured scoring system
A feedback generation module with actionable insights
Clean and well-documented code
Sample test cases and usage documentation

Why I’m a Good Fit:
I have a strong background in NLP and AI system design and have experience building practical, real-world intelligent systems. I focus on creating solutions that are both effective and interpretable, with clean architecture that can scale over time.
"""


if __name__ == "__main__":
    query = format_job_proposal(job_desc = job_desc, proposal = proposal)
    print(query)
    

    response = tools_alignment_subagent.invoke(query = query)

    F.print_structured_response(response)
    