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


# -- Models
EMBEDDING_MODELS=("bge" "nomic")
RERANKER_MODELS=("minilm" "mixedbread")


# -- empty result files 
> $RAG_lOG_FILE_PATH
> $RAG_CSV_PATH


echo "embedder,reranker,metric,value" > $RAG_CSV_PATH

source "$VENV_PATH"
run_id=1


# Evaluate RAG Components
echo "Start RAG Evaluation Process" > $RAG_lOG_FILE_PATH
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
            --log_file $RAG_lOG_FILE_PATH \
            --csv_file $RAG_CSV_PATH

        ((run_id++))
    done
done

echo "Finished Evaluating" >> $RAG_lOG_FILE_PATH





