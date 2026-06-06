#!/bin/bash

TASKS=(
    # "bio_analysis"
    #"skills_analysis"
    "visual_brand_analysis"
    #"super_agent"
)


MODELS_BIO_ANALYSIS=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "QWEN_32B"
    "LLAMA_70B"
)

MODELS_SKILLS_ANALYSIS=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "QWEN_32B"
)

MODELS_VISUAL_BRAND_ANALYSIS=(
    "GEMINI_FLASH"
    "GEMINI_FLASH_LITE"
)

MODELS_SUPER_AGENT=(
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
)


TEMPERATUREs=(
    0.0
)

MAX_TOKENSs=(
    1024
)

TOP_Ps=(
    0.9
)

run_id=1

echo "======================================= Starting Profile Suite Evaluation ======================================="

for task in "${TASKS[@]}"
do
    echo ">> Starting Task [$task]"

    if [ "$task" = "bio_analysis" ]; then
        MODELS_LIST=("${MODELS_BIO_ANALYSIS[@]}")

    elif [ "$task" = "skills_analysis" ]; then
        MODELS_LIST=("${MODELS_SKILLS_ANALYSIS[@]}")

    elif [ "$task" = "visual_brand_analysis" ]; then
        MODELS_LIST=("${MODELS_VISUAL_BRAND_ANALYSIS[@]}")

    elif [ "$task" = "super_agent" ]; then
        MODELS_LIST=("${MODELS_SUPER_AGENT[@]}")

    else
        echo "Unknown task: $task"
        exit 1
    fi

    for model in "${MODELS_LIST[@]}"
    do
        echo -e "\t>> Model: $model"
        for temp in "${TEMPERATUREs[@]}"
        do
            for max_tokens in "${MAX_TOKENSs[@]}"
            do
                for top_p in "${TOP_Ps[@]}"
                do
                    echo -e "\t\t>>>>> Run ID: $run_id | Task: $task | Model: $model <<<<<"
                    
                    python evaluate.py \
                        --run_id      "$run_id" \
                        --task        "$task" \
                        --model       "$model" \
                        --temperature "$temp" \
                        --max_tokens  "$max_tokens" \
                        --top_p       "$top_p"

                    ((run_id++))
                    
                    echo "--------------------------------------------------------------"
                done
            done
        done
    done
    run_id=1
    echo
done

echo "======================================= Evaluation Suite Completed Successfully ======================================="