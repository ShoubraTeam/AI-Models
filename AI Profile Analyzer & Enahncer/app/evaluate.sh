#!/bin/bash

VENV_PATH="/mnt/d/Focus/_____Active_______/AI/venvs/GP/PRR/bin/activate"

# agents with providers
TOOLS_ALIGNMENT_JOB_TOOLS_EXTRACTOR=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "GPT_OSS_20B"
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)

TOOLS_ALIGNMENT_PROPOSAL_ANALYZER=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "GPT_OSS_20B"
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)

JOB_UNDERSTANDING_KEY_POINTS_EXTRACTOR=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "GPT_OSS_20B"
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)

JOB_UNDERSTANDING_MATCHER=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "GPT_OSS_20B"
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)


REQUIREMENT_COVERAGE_EXTRACTOR=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "GPT_OSS_20B"
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)


REQUIREMENT_COVERAGE_MATCHER=(
    "LLAMA_8B"
    "GEMINI_FLASH_LITE"
    "GPT_OSS_20B"
    "QWEN_32B"
    "GEMINI_FLASH"
    "LLAMA_70B"
    "GPT_OSS_120B"
)


# activate venv
source "$VENV_PATH"

python evaluate.py \
    --task     tools_alignment \
    --models   QWEN_32B QWEN_32B \
    --temperature 0.0 \

