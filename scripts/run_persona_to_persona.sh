#!/bin/bash

# Define variables for better maintainability
PYTHON_SCRIPT="src/data/persona_2_persona.py"
LANGUAGE="sw"
MODEL="together_ai/meta-llama/Llama-3-8b-chat-hf"
PROVIDER="together"
BATCH_SIZE=8
MAX_TOKENS=512
TEMPERATURE=1.0
DATA_DIR="files/wikipedia_personas/"


while [ $# -gt 0 ]; do
    case "$1" in
        --data_directory=*)
            DATA_DIR="${1#*=}"
            ;;
        --language=*)
            LANGUAGE="${1#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            ;;
        --model=*)
            MODEL="${1#*=}"
            ;;
        --model_provider=*)
            PROVIDER="${1#*=}"
            ;;
        --batch_size=*)
            BATCH_SIZE="${1#*=}"
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
    shift
done

# Print run parameters
echo "=== Please verify the following parameters ==="
echo "Python Script: ${PYTHON_SCRIPT}"
echo "Language: ${LANGUAGE}"
echo "Model: ${MODEL}"
echo "Model Provider: ${PROVIDER}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Max Tokens: ${MAX_TOKENS}"
echo "Temperature: ${TEMPERATURE}"
echo "Data Directory: ${DATA_DIR}"
echo "==========================================="

# Ask for confirmation
read -p "Do you want to proceed with these parameters? (y/n): " confirm

if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo "Starting script..."
    python3 "${PYTHON_SCRIPT}" \
        --language "${LANGUAGE}" \
        --model "${MODEL}" \
        --model_provider "${PROVIDER}" \
        --batch_size "${BATCH_SIZE}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --data_directory "${DATA_DIR}"
else
    echo "Script execution cancelled."
    exit 1
fi