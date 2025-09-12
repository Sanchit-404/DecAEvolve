############## RUN BASED ON KAZEM CODE  ###########################

# Start VLLM server with 2 GPUs
echo "Starting VLLM server..."
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model Qwen/Qwen2.5-0.5B-Instruct &

# Wait a moment for server to start
sleep 50

# Run GRPO training on remaining GPUs
echo "Starting GRPO training for LLMSR..."
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
  --config_file accelerate_deepspeed_config.yaml \
  main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_offline_grpo True \
               --grpo_learning_rate 1e-6 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct" \

# Wait for all processes to complete
wait

echo "Training completed!"