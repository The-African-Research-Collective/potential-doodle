export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=8B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

echo "Training model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file configs/deep_speed/stage3_no_offloading_accelerate.conf \
    src/training/sft.py configs/grade_school_math_llama_3_8b.yaml