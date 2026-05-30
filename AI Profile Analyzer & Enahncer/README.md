# GoFreelance: AI Freelancer Profile Auditor & Enhancer

A comprehensive system designed to evaluate, audit, and enhance freelancer profiles to maximize their marketability and client acquisition rates. It leverages a modular, agentic AI framework based on an object-oriented multi-agent architecture to critically review profile assets. The system analyzes visual presentation, marketing copy, skill alignments, and pricing strategies relative to specific industry roles, providing targeted, actionable recommendations to help freelancers stand out in a competitive marketplace.

## System Description

The system evaluates the **overall profile health and role-suitability**, measuring how effectively a freelancer's public profile attracts premium clients. It is built on a highly scalable, object-oriented architecture designed to accommodate five specialized sub-agents, each inheriting from a unified `BaseAgent` and responsible for a distinct dimension of the audit:

* **Visual Brand Agent**: A vision-driven model that directly reviews the profile picture, evaluating lighting, composition, and domain-specific professionalism based on the target role.
* **Copywriting & Positioning Agent**: Analyzes the profile's Bio/Summary text for grammatical accuracy, engaging narrative hooks, professional tone, and value proposition.
* **Skill & Domain Alignment Agent**: Maps declared skills against the target role category to identify skill gaps, keyword optimization defects, and technical consistency.
* **Market Pricing & Credibility Agent**: Assesses numerical metrics (hourly rates, review counts, average ratings) against market standards to optimize pricing positioning.
* **Aggregator & Report Generator**: Coordinates the workflow, synthesizes sub-agent reports, performs semantic deduplication, and compiles the final structured enhancement audit.

## Workflow

### 1.0 Visual Brand Agent (Vision-Driven)

* **Multimodal Encoding**: The system accepts a local image path and the freelancer's specific job role. The `VisualBrandEvaluator` sub-agent intercepts the image, automatically detects its MIME type, and encodes the raw image bytes into a secure Base64 data URI string.
* **Domain-Aware Vision Evaluation**: Utilizing **Gemini 2.5 Flash** as a true vision language model, the agent processes a unified multimodal payload (the encoded image and textual role context). The model dynamically adjusts its strict evaluation rules according to 9 core platform roles grouped into two distinct domains:
* **Technical & Engineering Roles** `[AI Engineer, Backend Developer, Frontend Developer, Mobile Developer, Data Analyst, DevOps Engineer]`: Evaluated against corporate standards requiring clean, neutral, clutter-free backgrounds and neat corporate/smart-casual attire.
* **Creative & Media Roles** `[Graphic Designer, Content Writer, Video Producer]`: Welcomes expressive, artistic, and visually stylish environments (e.g., design studios, production setups) and modern-casual outfits, provided they look premium and professional.


* **Conciseness & Validation Control**: The prompt enforces strict output constraints to eliminate AI fluff. Every critique is limited to a single, direct, action-oriented sentence. The evaluation is then parsed and forced into a structured Pydantic data object (`VisualBrandEvaluationSchema`) returning:
* `score` (0.0 to 1.0): An objective measure of professional visual suitability.
* `feedback` (List of strings): Punchy, short, and direct bullet points evaluating lighting, background cleanliness, attire, and facial expression.

---

## Requirements

* Python 3.10 or later
* Google AI Studio API key (`Gemini 2.5 Flash` access)
* Python virtual environment recommended

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Freelancer-Profile-Auditor

```

### 2. Create a virtual environment (Recommended)

* Windows

```bash
python -m venv venv
venv\Scripts\activate

```

* Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install langchain-google-genai

```

### 4. Configure the environment file

Create a `.env` file in the project root:

```bash
touch .env

```

Add your Google API key inside the `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key_here

```

### 5. Run the app

```bash
cd app
python main.py

```
