# Job Description Suggestion

An AI-powered system that improves incomplete or poorly written job descriptions and, when necessary, suggests relevant tools and technologies using Retrieval-Augmented Generation (RAG).

The project was designed for freelancing platforms, where clients may understand the project they want to build but may not know the technical requirements needed to describe it clearly.

Instead of relying only on an LLM to guess those requirements, the system retrieves information from thousands of real job postings and uses that context to produce more relevant and grounded suggestions.

---

## The Problem

Clients on freelancing platforms often create job posts such as:

> I need an AI engineer to build a customer-support chatbot.

The general goal is understandable, but important information may be missing:

* What technologies are relevant?
* What skills should the freelancer have?
* What responsibilities should be included?
* How should the project scope be structured?

An LLM can rewrite the text, but blindly generating technical requirements introduces the risk of recommending irrelevant tools.

This project addresses that problem by combining **LLMs with RAG**.

---

## How It Works

The system follows two possible paths depending on the original job description.

### Case 1 — Technical Tools Are Already Mentioned

If the description already contains relevant technologies or frameworks:

1. The system detects their presence.
2. No retrieval is required.
3. The LLM restructures and enhances the original description while preserving its stated technical requirements.

### Case 2 — Technical Tools Are Missing

If the description does not contain sufficient technical tools:

1. The job title and description are converted into a retrieval query.
2. Similar real-world job postings are retrieved from the vector database.
3. Retrieved jobs are reranked using a cross-encoder.
4. An LLM analyzes the most relevant jobs and extracts commonly required tools/frameworks.
5. The extracted tools are supplied as context to the enhancement model.
6. The final job description is rewritten with clearer requirements and grounded technical suggestions.

The resulting pipeline can be summarized as:

```text
Job Title + Description
          |
          v
   Tool Detection
          |
     +----+----+
     |         |
 Tools      No Tools
 Present       |
     |         v
     |    Retrieve Similar Jobs
     |         |
     |       Rerank
     |         |
     |    Extract Common Tools
     |         |
     +----+----+
          |
          v
   Enhance Description
          |
          v
 Structured Job Posting
```

---

# RAG Pipeline

## 1. Data Collection and Preparation

The retrieval knowledge base was built from multiple job-posting datasets with different schemas and formats.

The data preparation process included:

* standardizing column names;
* normalizing job titles and descriptions;
* preserving available skills and job categories;
* assigning publication years when available;
* converting manually collected text-based job descriptions into structured records;
* removing duplicates and low-information descriptions;
* filtering irrelevant data;
* chunking unusually long job descriptions.

The original sources were merged into approximately **35,595 job records**.

Descriptions containing insufficient information were removed, leaving approximately **33,590 valid jobs**.

Long descriptions were then split into smaller semantic units, producing a final retrieval corpus of approximately:

> **34,060 job documents**

---

## 2. Semantic Job Classification

Part of the collected freelancing data contained jobs from many different domains.

To retain jobs relevant to the target platform, semantic classification was performed using **SentenceTransformers**.

Job titles and descriptions were embedded and compared against predefined role representations such as:

* AI Engineer
* Backend Developer
* Frontend Developer
* DevOps Engineer
* Data Analyst
* Mobile Developer
* Content Writer
* Video Producer
* Graphic Designer

Cosine similarity was used to identify the closest category, and jobs below the relevance threshold were filtered out.

This provided a cleaner domain-specific knowledge base than simple keyword matching.

---

## 3. Document Construction

Each cleaned job record was transformed into a retrieval document containing available information such as:

```text
Job Title

Job Description

Job Category

Recommended Skills
```

This provides the retriever with richer semantic context than embedding the job description alone.

---

## 4. Embeddings

The primary embedding model used by the system is:

**BAAI/bge-base-en-v1.5**

Job documents are converted into dense **768-dimensional embeddings** using Hugging Face tooling.

These embeddings represent semantic meaning and allow jobs with similar requirements to be retrieved even when they do not use exactly the same words.

---

## 5. Vector Database

The embedded job documents are stored in **Weaviate**.

Each object contains:

* job document;
* publication year;
* vector embedding.

The application performs **hybrid retrieval**, combining:

* lexical matching;
* vector similarity.

This helps balance exact terminology with semantic relevance.

Retrieved results are also ordered with preference toward newer job information when appropriate.

---

## 6. Reranking

Initial vector retrieval returns a broader candidate set.

To improve precision before sending context to the LLM, the candidates are reranked using a **CrossEncoder**.

The production configuration uses:

**cross-encoder/ms-marco-MiniLM-L-6-v2**

Rather than independently embedding the query and document, the cross-encoder evaluates them together, producing a stronger relevance score.

The workflow therefore becomes:

```text
Query
  ↓
Hybrid Retrieval
  ↓
Top Candidate Jobs
  ↓
Cross-Encoder Reranking
  ↓
Most Relevant Jobs
```

Only the strongest results are passed to the next stage.

---

# Tool Detection and Extraction

The system does not unnecessarily invoke RAG for every request.

An LLM-based detector first determines whether the submitted job description already specifies relevant technologies.

For example:

```text
I need an AI Engineer to build a customer-support chatbot.
```

would trigger tool suggestion.

However:

```text
I need an AI Engineer to build a RAG chatbot using Python
and a vector database.
```

already provides technical requirements, so retrieval can be skipped.

When retrieval is required, the most relevant job descriptions are passed to a dedicated extraction prompt.

The extractor identifies commonly occurring technologies and returns a structured list of relevant tools/frameworks.

This gives the enhancement model grounded technical context rather than asking it to invent requirements from scratch.

---

# Job Description Enhancement

The final generation stage transforms the original description into a professional, structured job post.

Depending on whether RAG was required, the model receives:

```text
Original Job Description
```

or:

```text
Original Job Description
+
Retrieved Technical Suggestions
```

The output is organized into sections such as:

### Overview

A clearer explanation of the project and its objective.

### Requirements

Responsibilities and technical expectations for the freelancer.

### Tools / Frameworks Required

Technologies explicitly provided by the client or suggested from retrieved job evidence.

This separation helps make generated job posts easier for both clients and freelancers to understand.

---

# LLM Integration

The system integrates LLMs through **Groq**.

Different models can be assigned specialized responsibilities:

* detecting whether tools already exist;
* extracting technologies from retrieved jobs;
* enhancing the final description.

The primary implementation uses Llama-family models, while the evaluation framework was designed to compare multiple model combinations rather than assuming that one model performs best for every stage.

---

# Evaluation

A dedicated evaluation pipeline was developed for both the **retrieval system** and the **LLM components**.

## Retrieval Evaluation

Different combinations of embedding models and rerankers were evaluated.

Embedding models included:

* `BAAI/bge-base-en-v1.5`
* `nomic-ai/nomic-embed-text-v1.5`

Rerankers included:

* `cross-encoder/ms-marco-MiniLM-L-6-v2`
* `mixedbread-ai/mxbai-rerank-large-v1`

The retrieval pipeline was measured using:

* Recall@K
* Precision@K
* Mean Reciprocal Rank (MRR)
* Embedding latency

For example, the evaluated BGE + Mixedbread configuration reached approximately:

```text
Precision@K: 0.579
Recall@K:    0.347
MRR:         0.498
```

while the Nomic + Mixedbread experiment produced the highest measured MRR of approximately:

```text
MRR: 0.512
```

The production configuration balances retrieval quality, model size, and runtime efficiency rather than selecting models based on one metric alone.

---

## LLM Evaluation

Different models were evaluated for the individual stages of the pipeline.

The evaluation included metrics for:

### Tool Detection

* classification accuracy;
* inference time.

### Tool Extraction

* precision;
* recall;
* F1 score;
* inference time.

### Enhancement

* generation time;
* semantic answer similarity.

Across tested configurations, tool-extraction F1 reached approximately **0.87** in the strongest experiment, while generated-description semantic similarity reached approximately **0.87** in several configurations.

This evaluation helped treat model selection as an engineering decision rather than relying solely on subjective output inspection.

---

# Technologies

### Generative AI & RAG

* Large Language Models
* Retrieval-Augmented Generation (RAG)
* Groq
* Llama Models
* Prompt Engineering

### Retrieval & NLP

* Hugging Face
* SentenceTransformers
* BGE Embeddings
* CrossEncoder
* Semantic Search
* Hybrid Search
* Reranking
* Cosine Similarity

### Vector Database

* Weaviate

### Data & Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch

### Evaluation

* Precision
* Recall
* F1 Score
* Recall@K
* Precision@K
* Mean Reciprocal Rank (MRR)
* Semantic Answer Similarity

---

# Project Structure

```text
JobDescriptionSuggestion/
│
├── app/
│   ├── data/
│   └── src/
│       ├── job_enhancer.py
│       ├── vector_database.py
│       └── utils/
│
├── data/
│   ├── concatenating_datasets.ipynb
│   ├── data_preparation.py
│   ├── data_profiling_1.ipynb
│   ├── data_profiling_2.ipynb
│   └── data/
│
├── evaluation/
│   ├── data/
│   ├── results/
│   └── src/
│       ├── evaluate_llms.py
│       ├── evaluate_rag.py
│       └── vector_database.py
│
├── system_development/
│   ├── main.py
│   └── src/
│       ├── job_enhancer.py
│       ├── vector_database.py
│       └── utils/
│
└── requirements.txt
```

---

# Key Engineering Ideas

This project explores several practical Generative AI and information-retrieval concepts:

* Retrieval-Augmented Generation
* Semantic Search
* Hybrid Retrieval
* Dense Embeddings
* Vector Databases
* Cross-Encoder Reranking
* Dynamic RAG Invocation
* LLM Task Specialization
* Prompt Engineering
* Semantic Job Classification
* Data Cleaning and Standardization
* LLM and Retriever Evaluation
* Grounded Technical Recommendation

---

# Final Outcome

The project demonstrates how RAG can be used for more than simply answering questions over documents.

Here, retrieval acts as a **technical recommendation mechanism**: similar real-world jobs provide evidence about the technologies commonly associated with a client's requested project.

By combining large-scale job data, semantic retrieval, reranking, specialized LLM stages, and systematic evaluation, the system can transform minimal job descriptions into clearer and more technically informative job postings while reducing reliance on unsupported LLM suggestions.
