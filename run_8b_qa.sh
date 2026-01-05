#!/bin/bash

# ================= 配置区域 =================
export HF_ENDPOINT=https://hf-mirror.com
# 关键：禁用 WandB，防止 debug=False 时卡住，或者 debug=True 时上传慢
export WANDB_MODE=disabled

# 日志文件
LOG_FILE="final_8b_mix.log"

echo "==========================================" | tee -a $LOG_FILE
echo "🚀 开始执行 8B 混合测试任务 (PopQA + TruthfulQA)" | tee -a $LOG_FILE
echo "📅 开始时间: $(date)" | tee -a $LOG_FILE
echo "⚙️ RCA参数: Alpha=0.04, Beta=1.0, Gamma=0.0" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE


# ================= 任务 3: TruthfulQA (RCA) =================
# 目的：验证 RCA 在事实性任务上是否安全/有效
echo -e "\n\n>>> [1 / 1] Running TruthfulQA (RCA, Alpha=0.04)..." | tee -a $LOG_FILE
python scripts/main.py \
    experiment=truthfulqa/baseline/llama3_8b_instruct \
    decoder=aca \
    data.num_samples=-1 \
    data_loader.batch_size=1 \
    debug=True \
    decoder.configs.alpha=0.04 \
    decoder.configs.beta=1.0 \
    decoder.configs.gamma=0.0 \
    >> $LOG_FILE 2>&1

echo "==========================================" | tee -a $LOG_FILE
echo "✅ 所有任务执行完毕！" | tee -a $LOG_FILE
echo "📅 结束时间: $(date)" | tee -a $LOG_FILE