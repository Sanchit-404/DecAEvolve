cd llm-sr2l

pip install -r requirements.txt

python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --use_offline_grpo True \
               --grpo_learning_rate 2e-5 \
               --hf_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
               --log_path "./logs/oscillator1_v16"