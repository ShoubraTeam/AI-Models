#!/bin/bash

VENV_PATH="/mnt/d/Focus/_____Active_______/AI/venvs/GP/PRR/bin/activate"


TASKS=(
    # "job_tools_extractor"
    # "proposal_tools_analyzer"
    "job_requirements_extractor"
    "job_requirements_matcher"
    # "job_key_points_extractor"
    # "job_understanding_evaluator"
    "experience_evidence_finder"
    # "language_clarity_evaluator"
)

# models
MODELS_TOOLS_ALIGNMENT_JOB_TOOLS_EXTRACTOR=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "GPT_OSS_20B"
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)


MODELS_TOOLS_ALIGNMENT_PROPOSAL_ANALYZER=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "GPT_OSS_20B"
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)


MODELS_JOB_UNDERSTANDING_KEY_POINTS_EXTRACTOR=(
    "QWEN_32B"
    "LLAMA_70B"
    "GPT_OSS_120B"
)


MODELS_JOB_UNDERSTANDING_MATCHER=(
    "GPT_OSS_20B"
    "QWEN_32B"
    "GPT_OSS_120B"
)


MODELS_REQUIREMENT_COVERAGE_EXTRACTOR=(
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)


MODELS_REQUIREMENT_COVERAGE_MATCHER=(
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
)

MODELS_EVIDENCE_EXPERIENCE_FINDER=(
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)

MODELS_LANGUAGE_CLARITY=(
    "GPT_OSS_20B"
    "QWEN_32B"
    "GPT_OSS_120B"
)


TEMPERATUREs=(
    0.0
    0.1
    0.2
)

MAX_TOKENSs=(
    512
    1024
    2048
)

TOP_Ps=(
    0.9
    0.95
    1.0
)

run_id=1

echo "======================================= Starting Evaluation ======================================="


echo "======================================= Activating Venv ======================================="
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
else
    echo ">> VENV Path does not exist"
    exit 1
fi


for task in "${TASKS[@]}"
do
    echo ">> Starting Task [$task]"
    if [ "$task" = "job_tools_extractor" ]; then
        MODELS_LIST=("${MODELS_TOOLS_ALIGNMENT_JOB_TOOLS_EXTRACTOR[@]}")

    elif [ "$task" = "proposal_tools_analyzer" ]; then
        MODELS_LIST=("${MODELS_TOOLS_ALIGNMENT_PROPOSAL_ANALYZER[@]}")

    elif [ "$task" = "job_requirements_extractor" ]; then
        MODELS_LIST=("${MODELS_JOB_UNDERSTANDING_KEY_POINTS_EXTRACTOR[@]}")

    elif [ "$task" = "job_requirements_matcher" ]; then
        MODELS_LIST=("${MODELS_JOB_UNDERSTANDING_MATCHER[@]}")

    elif [ "$task" = "job_key_points_extractor" ]; then
        MODELS_LIST=("${MODELS_REQUIREMENT_COVERAGE_EXTRACTOR[@]}")

    elif [ "$task" = "job_understanding_evaluator" ]; then
        MODELS_LIST=("${MODELS_REQUIREMENT_COVERAGE_MATCHER[@]}")

    elif [ "$task" = "experience_evidence_finder" ]; then
        MODELS_LIST=("${MODELS_EVIDENCE_EXPERIENCE_FINDER[@]}")

    elif [ "$task" = "language_clarity_evaluator" ]; then
        MODELS_LIST=("${MODELS_LANGUAGE_CLARITY[@]}")

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
                    echo -e "\t\t>>>>> Run ID: $run_id <<<<<"
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
    

