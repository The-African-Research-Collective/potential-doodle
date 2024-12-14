volume=$PWD/llm_data
model=google/gemma-2-9b-it

export PREFIX_CACHING=0

docker run --gpus all --shm-size 1g -e HF_TOKEN=$token -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:3.0.1 --model-id $model  --quantize bitsandbytes