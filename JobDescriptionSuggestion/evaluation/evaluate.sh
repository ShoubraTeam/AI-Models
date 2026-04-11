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
LLM_lOG_FILE_PATH="./results/log/llm_log_file.txt"
LLM_CSV_PATH="./results/csv/llm_results.csv"

# -- Models
TOOLS_DETECTORS=("meta-llama" "llama")
TOOLS_EXTRACTORS=("llama" "gpt")
ENHANCEMENT_MODELS=("llama" "qwen" "meta-llama")



## Setup

# -- empty result files 
> $LLM_lOG_FILE_PATH
> $LLM_CSV_PATH

echo "detector,extractor,enhancer,metric,value" > $LLM_CSV_PATH

source "$VENV_PATH"
run_id=




# # Evaluate LLMs Components
echo "Start LLM Evaluation Process" >> $LLM_lOG_FILE_PATH
for detector in "${TOOLS_DETECTORS[@]}"; do
    for extractor in "${TOOLS_EXTRACTORS[@]}"; do
        for enhancer in "${ENHANCEMENT_MODELS[@]}"; do

            python main.py \
                --run_id $run_id \
                --component LLM \
                --eval_data_path $EVAL_DATA_PATH \
                --relevant_doc_path $RELEVANT_CHUNKS_PATH \
                --detector $detector \
                --extractor $extractor \
                --enhancer $enhancer \
                --log_file $LLM_lOG_FILE_PATH \
                --csv_file $LLM_CSV_PATH

        ((run_id++))
        done
    done
done


echo "Finished Evaluating" >> $LLM_lOG_FILE_PATH

