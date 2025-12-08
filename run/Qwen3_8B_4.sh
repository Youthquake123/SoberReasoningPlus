#!/bin/bash
#SBATCH --job-name=qwen3_8b_4_3
#SBATCH -A m4899
#SBATCH -C gpu&hbm40g
#SBATCH -q regular
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH --gpus-per-task=1
#SBATCH -o /pscratch/sd/z/zax/SoberReasoningPlus/results/logs/qwen3_8b_4_3_%j.out  
#SBATCH -e /pscratch/sd/z/zax/SoberReasoningPlus/results/logs/qwen3_8b_4_3_%j.err 

module load cuda/12.9
export HF_HOME="/pscratch/sd/z/zax/huggingface_cache"
export VLLM_CACHE_ROOT="/pscratch/sd/z/zax/vllm_cache"
echo "VLLM_CACHE_ROOT is set to $VLLM_CACHE_ROOT"
echo "HF_HOME is set to $HF_HOME"
source /pscratch/sd/z/zax/env/test/bin/activate

unset VLLM_ATTENTION_BACKEND

export NCCL_DEBUG=TRACE
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export TORCH_CUDA_ARCH_LIST="80"

LOCAL_DIR="/pscratch/sd/z/zax/SoberReasoningPlus"
OUTPUT_DIR="/pscratch/sd/z/zax/SoberReasoningPlus/results"

# Define prompts directly in the shell script
export SYSTEM_PROMPT="You are a helpful assistant."
export MATH_QUERY_TEMPLATE="Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: \$\\boxed{{ANSWER}}\$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{Question}
"

MODELS=(
    Qwen/Qwen3-8B
)

MAX_NUM_SEQUENCES=(128)
MAX_NUM_BATCHED_TOKENS=(131072)
DTYPES=("bfloat16")
MAX_MODEL_LENGTHS=(34816)
MAX_TOKENS_LIST=(32768)

TEMPS=(0 0.2 0.4 0.6)
TOP_PS=(0.7 0.8 0.9 0.95 1.0)
# Define how many seeds to run for each task
# Format: "task_string:num_seeds"
TASK_NUM_SEEDS_CONFIG=(
    "custom|aime24|0|0:16"
    "custom|aime25|0|0:16"
    "custom|amc23|0|0:16"
    "custom|math_500|0|0:3"
    "custom|minerva|0|0:3"
    "custom|olympiadbench|0|0:3"
)

# The outer loops for model configuration remain the same
for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
for MODEL in "${MODELS[@]}"; do
for DTYPE in "${DTYPES[@]}"; do
cd $LOCAL_DIR

# REMOVED: The loops for TASK, TEMP, TOP_P, and RUN are gone from here.
# MODIFIED: A single call to python passes all the parameter lists.

gpu_idx=0
for i in ${!TEMPS[@]}; do
    TEMP=${TEMPS[i]}
    echo "Starting temperature $TEMP on GPU $gpu_idx"

    # Run each temperature setting on a different GPU in the background
    CUDA_VISIBLE_DEVICES=$gpu_idx python main_diff.py \
        --model $MODEL \
        --task_num_seeds "${TASK_NUM_SEEDS_CONFIG[@]}" \
        --temperature $TEMP \
        --top_p "${TOP_PS[@]}" \
        --output_dir "${OUTPUT_DIR}/temp_${TEMP}" \
        --max_new_tokens $MAX_TOKENS \
        --max_model_length $MAX_MODEL_LENGTH \
        --custom_tasks_directory lighteval_tasks.py \
        --system_prompt "$SYSTEM_PROMPT" \
        --use_chat_template \
        --dtype $DTYPE \
        --max_num_seqs $MAX_NUM_SEQUENCES \
        --max_num_batched_tokens $MAX_NUM_BATCHED_TOKENS \
        --tensor_parallel_size 1 \
        --pipeline_parallel_size 1 \
        --data_parallel_size 1 &

    gpu_idx=$((gpu_idx + 1))
done

# Wait for all background jobs to finish
wait

done
done
done
done