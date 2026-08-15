# Proposal Rejection Reasons

An Agentic AI system designed to help freelancers understand **why their proposals may be rejected** and how they can improve them before applying to future jobs.

Instead of asking a single LLM to judge an entire proposal at once, the system breaks the evaluation into several specialized tasks. Each sub-agent analyzes a specific aspect of the relationship between the client's job description and the freelancer's proposal, while a final **SuperAgent** combines the findings into a structured assessment.

The goal is to provide freelancers with concrete, explainable feedback rather than generic suggestions such as *“make your proposal more professional.”*

---

## Why This Project?

Freelancers often submit proposals without knowing why clients reject them.

A proposal may fail for very different reasons:

* it does not address the actual problem;
* important requirements are missing;
* required tools are not mentioned;
* the freelancer provides no evidence of relevant experience;
* the proposal is unclear or unprofessional.

Treating all of these as one classification problem makes the result difficult to explain.

This project therefore uses a **multi-agent architecture**, where each agent specializes in one evaluation dimension.

---

# System Architecture

The system evaluates a job description and a freelancer proposal through five specialized analysis dimensions:

1. **Tools Alignment**
2. **Requirement Coverage**
3. **Job Understanding**
4. **Experience Evidence**
5. **Language Clarity**

Their outputs are normalized into a common structured format and passed to a **SuperAgent**, which produces the final proposal assessment.

```text
              Job Description
                     |
          +----------+----------+
          |          |          |
     Extract Tools  Extract   Extract
                   Key Points Requirements
          |          |          |
          +----------+----------+
                     |
                  Proposal
                     |
     +---------------+---------------+
     |               |               |
 Tools Alignment   Job          Requirement
                  Understanding   Coverage
     |               |               |
     +----------+----+----+----------+
                |         |
        Experience     Language
         Evidence       Clarity
                |         |
                +----+----+
                     |
                SuperAgent
                     |
                     v
          Final Proposal Report
```

---

# 1. Tools Alignment

The Tools Alignment workflow determines how well the technologies mentioned in the freelancer's proposal match those requested by the client.

## Job Tools Extraction

An LLM first extracts tools and technologies from the job description.

Each extracted tool includes a **necessity level**:

* Mandatory
* Recommended
* Optional
* Forbidden

This is important because failing to mention a mandatory tool should have more impact than omitting an optional one.

## Proposal Tool Analysis

A second agent checks each extracted tool against the proposal and determines:

* whether the tool is mentioned;
* whether it is mentioned with confidence and relevant context;
* or whether it is only referenced generically.

For example:

```text
"I built several FastAPI services..."
```

provides stronger evidence than:

```text
"I know FastAPI."
```

## Weighted Scoring

A deterministic scoring layer calculates tool alignment using:

* tool necessity;
* whether it appeared in the proposal;
* confidence of the mention.

Forbidden technologies are handled separately: the proposal is rewarded for **not** using a forbidden tool.

This combines LLM semantic analysis with explicit programmatic scoring rather than relying entirely on subjective model judgment.

---

# 2. Requirement Coverage

The Requirement Coverage component determines whether the proposal actually addresses what the client requested.

## Requirement Extraction

A dedicated agent converts the raw job description into atomic requirements.

Each requirement contains:

* a unique ID;
* a clear requirement statement;
* a necessity level.

The extractor focuses on functional requirements and constraints while separating them from technology-specific analysis handled by the Tools Alignment module.

Example:

```text
REQ_1: Build an authentication system — mandatory
REQ_2: Add social login — recommended
REQ_3: Do not use external authentication providers — forbidden
```

## Semantic Requirement Matching

Another agent compares those requirements against the proposal and classifies them as covered or missing.

The matching is semantic rather than relying on exact keyword overlap, allowing the system to identify requirements even when the freelancer uses different wording.

## Weighted Coverage Score

The result is processed through a deterministic scoring function.

Missing important requirements lower the score automatically.

Forbidden requirements receive stronger treatment: explicitly violating a forbidden constraint applies an active penalty rather than simply contributing zero points.

This helps distinguish between:

> not mentioning an optional feature

and:

> proposing an architecture that directly violates the client's constraints.

---

# 3. Job Understanding

A technically strong proposal can still fail if the freelancer misunderstood the problem.

The Job Understanding module evaluates whether the proposal demonstrates a clear understanding of what the client actually wants.

## Job Key-Point Extraction

The first agent extracts:

* the core problem;
* required deliverables;
* important keywords and domain concepts.

## Proposal Evaluation

A second agent evaluates whether the proposal:

* identifies the client's main problem;
* proposes an appropriate solution;
* explains practical implementation steps;
* addresses important keywords;
* avoids irrelevant content.

The resulting score is normalized and processed using a configurable acceptance threshold.

This evaluates **problem comprehension**, not merely keyword matching.

---

# 4. Evidence of Experience

Freelancers frequently make generic claims such as:

> “I have five years of experience.”

or:

> “I am an expert Python developer.”

These claims do not necessarily prove that the freelancer has solved a similar problem before.

The Experience Evidence Agent therefore applies a stricter standard.

Evidence is considered valid only when the proposal contains concrete previous work, such as:

* a named project;
* a specific system previously built;
* a relevant case study;
* an implementation with identifiable deliverables.

For each detected project, the agent extracts:

* project title/type;
* project description;
* tools used;
* relevance to the current job;
* relevance score.

The final experience score is derived from the strongest relevant project rather than from unsupported claims.

---

# 5. Language Clarity

The Language Clarity module evaluates whether the proposal is easy for a client to read and trust.

It uses both **LLM-based evaluation and deterministic text metrics**.

## LLM Evaluation

The agent evaluates:

* clarity;
* professionalism;
* misleading or vague phrasing.

## Programmatic Metrics

The pipeline additionally calculates:

* word count;
* average sentence length;
* proposal length sufficiency;
* lightweight grammar-related heuristics.

The final language score combines these dimensions using explicit weighting.

This hybrid design prevents the entire language assessment from depending on one subjective LLM response.

---

# Parallel Agent Execution

Performance is improved by running independent evaluation tasks asynchronously.

Job-level information is extracted first and cached:

* tools;
* key points;
* requirements.

These features can then be reused when evaluating multiple proposals against the same job description.

Proposal-specific evaluators are executed concurrently using Python's asynchronous workflow.

This reduces redundant LLM calls and makes the architecture more suitable for evaluating multiple freelancers applying to the same job.

---

# Structured Outputs

All agents return structured responses validated with **Pydantic** models.

Instead of allowing free-form text such as:

```text
"The proposal seems okay but maybe..."
```

agents return predictable fields such as:

```text
score
accepted
summary
acceptance_reasons
rejection_reasons
```

Specialized schemas are also defined for:

* job tools;
* proposal tool analysis;
* extracted requirements;
* requirement coverage;
* job key points;
* job understanding;
* experience evidence;
* language clarity;
* final SuperAgent reports.

This makes agent outputs easier to validate, combine, evaluate, and integrate into downstream workflows.

---

# SuperAgent

After all specialized evaluators finish, their results are transformed into a consistent evaluation report and passed to a final **SuperAgent**.

The SuperAgent does not independently redo all previous analysis. Instead, it synthesizes the evidence produced by completed sub-agents.

The final structured response contains:

* **Verdict**

  * Accepted
  * At Risk
  * Rejected

* **Summary**

* **Strengths**

* **Rejection Risks**

* **Recommendations**

* **Evaluation Limitations**

Failed sub-agents are explicitly marked as unavailable and are not treated as evidence against the proposal.

This makes the final system more robust to partial failures.

---

# Error Handling and Resilience

The pipeline contains dedicated error types for the different agent stages.

When an individual evaluator fails:

* the failure is captured;
* the remaining independent evaluators can continue;
* the unavailable module is included as a diagnostic;
* the SuperAgent is instructed not to interpret missing results as rejection evidence.

This avoids turning one failed model call into an incorrect negative assessment.

The system also includes recovery logic for malformed structured LLM outputs by extracting and validating recoverable JSON payloads where possible.

---

# LLM Integration

The system currently uses **Groq-hosted language models** through a reusable agent abstraction.

Each agent can independently configure:

* model name;
* system prompt;
* temperature;
* token limits;
* structured-response schema.

The current pipeline mainly uses Llama-family models and supports model-specific structured output handling.

A reusable `BaseAgent` abstraction handles:

* model invocation;
* structured-output configuration;
* JSON extraction;
* Pydantic validation;
* synchronous and asynchronous invocation.

---

# Evaluation Framework

A dedicated evaluation framework is included for testing individual agents rather than judging only the final system manually.

Evaluation tasks include:

* Job Tools Extractor
* Proposal Tools Analyzer
* Job Requirements Extractor
* Job Requirements Matcher
* Job Key Points Extractor
* Job Understanding Evaluator
* Experience Evidence Evaluator
* Language Clarity Evaluator

Depending on the task, measured metrics include:

* Accuracy
* Precision
* Recall
* F1 Score
* Tool-necessity accuracy
* Confidence-detection accuracy
* Keyword precision/recall
* Deliverable recall
* Overall understanding accuracy
* Language-assessment accuracy
* Error rate
* Agent invocation latency

Different models and generation configurations can therefore be compared empirically.

---

# Technologies

### Agentic & Generative AI

* Agentic AI
* Multi-Agent Systems
* Large Language Models
* Groq
* Llama Models
* Prompt Engineering
* Structured LLM Outputs

### Python

* Python
* AsyncIO

### Validation & Data Contracts

* Pydantic

### Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Error Analysis
* Latency Evaluation

### Engineering Concepts

* Modular Agent Architecture
* Parallel Agent Execution
* Feature Reuse / Caching
* Deterministic Post-Processing
* Weighted Scoring
* Error Isolation
* Structured Output Validation

---

# Project Structure

```text
ProposalRejectionReasons/
│
├── app/
│   ├── agents/
│   │   ├── tools_alignment/
│   │   ├── requirement_coverage/
│   │   ├── job_understanding/
│   │   ├── experience_evidence/
│   │   ├── language_clarity/
│   │   └── super_agent/
│   │
│   ├── pipeline_core/
│   │   ├── prr_pipeline.py
│   │   ├── pipeline_config.py
│   │   └── pipeline_errors.py
│   │
│   ├── processing/
│   │   ├── tool_alignment_processing.py
│   │   ├── requirement_coverage_processing.py
│   │   ├── job_understanding_processing.py
│   │   ├── experience_evidence.py
│   │   └── language_clarity_processing.py
│   │
│   ├── schemas/
│   ├── prompts/
│   ├── evaluation/
│   ├── assets/
│   └── main.py
│
├── requirements.txt
└── .env.example
```

---

# Setup

## 1. Clone the Repository

```bash
git clone https://github.com/ShoubraTeam/AI-Models.git
cd AI-Models/ProposalRejectionReasons
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Groq

Copy the environment example:

```bash
cp .env.example .env
```

Then provide your Groq API key:

```env
GROQ_API_KEY=your_api_key
```

## 5. Run the Pipeline

```bash
cd app
python main.py
```

---

# Final Outcome

The project demonstrates how an agentic architecture can make proposal evaluation more **explainable, modular, and actionable**.

Rather than generating one vague quality score, the system identifies specific rejection risks across job understanding, requirements, technologies, previous experience, and communication quality.

The combination of specialized LLM agents, deterministic scoring, structured outputs, asynchronous orchestration, and a final synthesis layer provides freelancers with feedback they can actually use to improve future proposals.
