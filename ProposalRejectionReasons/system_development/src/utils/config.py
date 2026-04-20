# --------------------------------------------------------------------------------------
# System Configuration [System Prompts -- Model Names -- Hyperparameter Values -- ...]
# --------------------------------------------------------------------------------------

# -- System Prompts
TOOLS_ALIGNMENT_PROMPT = f"""You are a professional recruiter. 
You have been employed by a freelancing platform to compare a given job_description posted by a client to a proposal posted by a freelancer on that job.
Your primary role is to extract the technical tools/frameworks in both the job_description & the proposal.
You should extract all the mentioned tools in either. The tools can be used in any of:
- Artificial Intelligence (AI) & Data Science: such as Python, R, TensorFlow, sklearn, matplotlib, ...
- Software Development: such as front-end & back-end languages and frameworks.
- Technical Writing: such as GoogleDocs, overleaf, ...
"""




# -- Models
LLAMA_70B = "llama-3.3-70b-versatile"