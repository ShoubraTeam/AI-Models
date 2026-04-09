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
lOG_FILE_PATH="./results/log_file.txt"
CSV_PATH="./results/results.csv"

# -- Models
EMBEDDING_MODELS=("bge" "nomic")
RERANKER_MODELS=("minilm" "mixedbread")
TOOLS_DETECTORS=("llama" "gpt")
TOOLS_EXTRACTORS=("gpt" "qwen")
ENHANCEMENT_MODELS=("llama-big" "deepseek")



## Setup

# -- empty result files
> $lOG_FILE_PATH
> $CSV_PATH

echo "Start Evaluation Process" > $lOG_FILE_PATH
echo "type,embedder,reranker,model,metric,value" > $CSV_PATH

source "$VENV_PATH"
run_id=1


# Evaluate RAG Components
for emb in "${EMBEDDING_MODELS[@]}"; do
    for r in "${RERANKER_MODELS[@]}"; do
        python main.py \
            --run_id $run_id \
            --repeats 5 \
            --component RAG \
            --eval_data_path $EVAL_DATA_PATH \
            --relevant_doc_path $RELEVANT_CHUNKS_PATH \
            --embedding_model $emb \
            --reranker $r \
            --log_file $lOG_FILE_PATH \
            --csv_file $CSV_PATH
        ((run_id++))
    done
done
