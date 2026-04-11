# ------------------------------------------------
# File to automate the evaluation process
# ------------------------------------------------
#!/bin/bash


# Variables

# -- Data Paths
VENV_PATH="D:\Focus\_____Active_______\AI\venvs\rag_langchain_venv\Scripts\activate"
EVAL_DATA_PATH="./data/eval_data.json"
RELEVANT_CHUNKS_PATH="./data/relevant_chunks.json"

# -- Output Paths
RAG_lOG_FILE_PATH="./results/log/rag_log_file.txt"
RAG_CSV_PATH="./results/csv/rag_results.csv"

LLM_lOG_FILE_PATH="./results/log/llm_log_file.txt"
LLM_CSV_PATH="./results/csv/llm_results.csv"

# -- Models
EMBEDDING_MODELS=("bge" "nomic")
RERANKER_MODELS=("minilm" "mixedbread")
TOOLS_DETECTORS=("meta-llama" "llama")
TOOLS_EXTRACTORS=("llama" "gpt")
ENHANCEMENT_MODELS=("llama" "qwen" "meta-llama")



## Setup

# -- empty result files [UNCOMMENT YOUR SPECIFIC FILES "Do not empty all"]
# > $RAG_lOG_FILE_PATH
# > $RAG_CSV_PATH
# > $LLM_lOG_FILE_PATH
# > $LLM_CSV_PATH

# echo "embedder,reranker,metric,value" > $RAG_CSV_PATH
# echo "detector,extractor,enhancer,metric,value" > $LLM_CSV_PATH

source "$VENV_PATH"
run_id=1


# [UNCOMMENT YOUR SPECIFIC FILES "Do not write in all"]


# Evaluate RAG Components
# echo "Start RAG Evaluation Process" > $RAG_lOG_FILE_PATH
# for emb in "${EMBEDDING_MODELS[@]}"; do
#     for r in "${RERANKER_MODELS[@]}"; do

#         python main.py \
#             --run_id $run_id \
#             --repeats 5 \
#             --component RAG \
#             --eval_data_path $EVAL_DATA_PATH \
#             --relevant_doc_path $RELEVANT_CHUNKS_PATH \
#             --embedding_model $emb \
#             --reranker $r \
#             --log_file $RAG_lOG_FILE_PATH \
#             --csv_file $RAG_CSV_PATH

#         ((run_id++))
#     done
# done

# echo "Finished Evaluating" >> $RAG_lOG_FILE_PATH




# # Evaluate LLMs Components
# echo "Start LLM Evaluation Process" >> $LLM_lOG_FILE_PATH
# for detector in "${TOOLS_DETECTORS[@]}"; do
#     for extractor in "${TOOLS_EXTRACTORS[@]}"; do
#         for enhancer in "${ENHANCEMENT_MODELS[@]}"; do

#             python main.py \
#                 --run_id $run_id \
#                 --component LLM \
#                 --eval_data_path $EVAL_DATA_PATH \
#                 --relevant_doc_path $RELEVANT_CHUNKS_PATH \
#                 --detector $detector \
#                 --extractor $extractor \
#                 --enhancer $enhancer \
#                 --log_file $LLM_lOG_FILE_PATH \
#                 --csv_file $LLM_CSV_PATH

#         ((run_id++))
#         done
#     done
# done


# echo "Finished Evaluating" >> $LLM_lOG_FILE_PATH

