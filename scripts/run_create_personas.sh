#!/bin/bash

# Default values
PYTHON_SCRIPT="src/data/create_personas_wikipedia.py"  # Replace with actual script name
DATA_DIR="files/wikipedia_personas"
LANGUAGE="sw"
BATCH_SIZE=10
MODEL="azure_ai/newgpt4o"
PROVIDER="azure"

# Parse named arguments
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
echo "Data Directory: ${DATA_DIR}"
echo "Language: ${LANGUAGE}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Model: ${MODEL}"
echo "Model Provider: ${PROVIDER}"
echo "==========================================="

# Ask for confirmation
read -p "Do you want to proceed with these parameters? (y/n): " confirm

if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo "Starting script..."
    python3 "${PYTHON_SCRIPT}" \
        --data_directory "${DATA_DIR}" \
        --language "${LANGUAGE}" \
        --batch_size "${BATCH_SIZE}" \
        --model "${MODEL}" \
        --model_provider "${PROVIDER}"
else
    echo "Script execution cancelled."
    exit 1
fi