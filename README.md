levels.fparcd llm-sr2l

pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=2 python main.py --spec_path ./specs/specification_oscillator1_numpy.txt \
               --problem_name oscillator1 \
               --use_offline_grpo True \
               --grpo_learning_rate 1e-6 \
               --hf_model "Qwen/Qwen2.5-0.5B-Instruct"