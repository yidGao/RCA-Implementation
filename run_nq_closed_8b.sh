#!/bin/bash

# ================= 配置区域 =================
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE=disabled

# 8B 模型路径
MODEL_PATH="/root/DeCoRe/my_models/LLM-Research/Meta-Llama-3-8B-Instruct"

# 日志文件
LOG_FILE="compare_nq_closed.log"

echo "==========================================" | tee -a $LOG_FILE
echo "🚀 NQ-Open (闭卷) 对照实验: Baseline vs RCA" | tee -a $LOG_FILE
echo "📅 开始时间: $(date)" | tee -a $LOG_FILE
echo "==========================================" | tee -a $LOG_FILE

# --------------------------------------------------------
# 任务 1: Baseline (基准)
# --------------------------------------------------------
# 预期分数: ~29%
echo -e "\n\n>>> [1/2] Running Baseline (Closed Book)..." | tee -a $LOG_FILE

# 注意: decoder=baseline, 不需要 alpha/beta 参数
python scripts/main.py \
    experiment=nq/baseline/llama3_8b_instruct \
    decoder=baseline \
    data.num_samples=-1 \
    data.variation=closed_book \
    data_loader.batch_size=1 \
    debug=True \
    model.configs.model_name_or_path="$MODEL_PATH" \
    >> $LOG_FILE 2>&1

# --------------------------------------------------------
# 任务 2: RCA (你的方法)
# --------------------------------------------------------
# 参数: alpha=0.04 (通用参数)
# 如果这个分数比 Baseline 低，说明在闭卷任务上 RCA 不适用 (这也是重要结论)
echo -e "\n\n>>> [2/2] Running RCA (Closed Book, Alpha=0.04)..." | tee -a $LOG_FILE

python scripts/main.py \
    experiment=nq/baseline/llama3_8b_instruct \
    decoder=aca \
    data.num_samples=-1 \
    data.variation=closed_book \
    data_loader.batch_size=1 \
    debug=True \
    decoder.configs.alpha=0.04 \
    decoder.configs.beta=1.0 \
    decoder.configs.gamma=0.0 \
    model.configs.model_name_or_path="$MODEL_PATH" \
    >> $LOG_FILE 2>&1

echo "==========================================" | tee -a $LOG_FILE
echo "✅ 对比测试完成！" | tee -a $LOG_FILE
echo "📅 结束时间: $(date)" | tee -a $LOG_FILE