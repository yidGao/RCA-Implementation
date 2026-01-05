#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

# 8B 模型路径
MODEL_PATH="/root/DeCoRe/my_models/LLM-Research/Meta-Llama-3-8B-Instruct"
ALPHA=0.04
LOG_FILE="run_musique_8b.log"

echo "🚀 开始执行 MuSiQue (8B) 矩阵测试" > $LOG_FILE

# 1. Closed Book (Direct)
echo -e "\n>>> [1/4] 8B: Closed Book (No CoT)..." | tee -a $LOG_FILE
python scripts/main.py \
    experiment=musique/baseline/llama3_8b_instruct \
    decoder=aca \
    data.num_samples=-1 \
    data.variation=direct_closed_book \
    data_loader.batch_size=1 \
    debug=True \
    decoder.configs.alpha=$ALPHA decoder.configs.beta=1.0 decoder.configs.gamma=0.0 \
    model.configs.model_name_or_path="$MODEL_PATH" \
    >> $LOG_FILE 2>&1

# 2. Open Book (Direct)
echo -e "\n>>> [2/4] 8B: Open Book (No CoT)..." | tee -a $LOG_FILE
python scripts/main.py \
    experiment=musique/baseline/llama3_8b_instruct \
    decoder=aca \
    data.num_samples=-1 \
    data.variation=direct_open_book \
    data_loader.batch_size=1 \
    debug=True \
    decoder.configs.alpha=$ALPHA decoder.configs.beta=1.0 decoder.configs.gamma=0.0 \
    model.configs.model_name_or_path="$MODEL_PATH" \
    >> $LOG_FILE 2>&1

# 3. Closed Book (CoT)
echo -e "\n>>> [3/4] 8B: Closed Book (CoT)..." | tee -a $LOG_FILE
python scripts/main.py \
    experiment=musique/baseline/llama3_8b_instruct \
    decoder=aca \
    data.num_samples=-1 \
    data.variation=cot_closed_book \
    data_loader.batch_size=1 \
    debug=True \
    decoder.configs.alpha=$ALPHA decoder.configs.beta=1.0 decoder.configs.gamma=0.0 \
    model.configs.model_name_or_path="$MODEL_PATH" \
    >> $LOG_FILE 2>&1

# 4. Open Book (CoT)
echo -e "\n>>> [4/4] 8B: Open Book (CoT)..." | tee -a $LOG_FILE
python scripts/main.py \
    experiment=musique/baseline/llama3_8b_instruct \
    decoder=aca \
    data.num_samples=-1 \
    data.variation=cot_open_book \
    data_loader.batch_size=1 \
    debug=True \
    decoder.configs.alpha=$ALPHA decoder.configs.beta=1.0 decoder.configs.gamma=0.0 \
    model.configs.model_name_or_path="$MODEL_PATH" \
    >> $LOG_FILE 2>&1

echo "✅ 任务完成！" | tee -a $LOG_FILE