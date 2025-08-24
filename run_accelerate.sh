############## RUN BASED ON KAZEM CODE  ###########################

# Start VLLM server with 2 GPUs
echo "Starting VLLM server..."
CUDA_VISIBLE_DEVICES=0,1,2 trl vllm-serve \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 &

# Wait a moment for server to start
sleep 50

# Run GRPO training on remaining GPUs
echo "Starting GRPO training for LLMSR..."
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch \
  --config_file accelerate_deepspeed_config.yaml \
  main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --use_offline_grpo True \
               --grpo_learning_rate 1e-6 \
               --hf_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
               --log_path "./logs/oscillator1_v16" \
               --use_wandb True

# Wait for all processes to complete
wait

echo "Training completed!"