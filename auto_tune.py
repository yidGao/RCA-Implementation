import subprocess
import re
import itertools
import pandas as pd
import os
import time

# ================= 配置区域 =================
# 1. 设置你想测试的参数范围 (网格搜索)
# 建议分两轮跑：
# 第一轮：固定 gamma=0，只测 alpha 和 beta (找 Attention 的最佳点)
# 第二轮：固定最佳 alpha/beta，测 gamma 和 tau (找 MLP 的最佳点)

param_grid = {
    # 🔍 RCA 显微镜模式
    # 我们已知：0.01太弱，0.1有点猛
    # 重点扫描 0.03 - 0.10 这个区间
    # "alpha": [0.15, 0.2, 0.25], 
    
    "alpha": [0.3],
    # 🔒 锁定其他参数 (RCA 不需要这些)
    "beta":  [1.0],
    "gamma": [0.0],
    "tau":   [0.0]
}

# ⚠️ 关键调整：



# 2. 每次测试多少条数据？(建议 50-100 条，太少不准，太多太慢)
NUM_SAMPLES = 1000

# 3. 结果保存文件名
RESULT_FILE = "tuning_stage1_safe.csv"

# ===========================================

def parse_output(output_str):
    """从终端输出中提取指标字典"""
    # 匹配类似 {'rouge1': np.float64(30.5)...} 的结构
    try:
        # 使用正则寻找包含 rouge1 的字典字符串
        match = re.search(r"\{'rouge1':.*\}", output_str)
        if match:
            # 把 np.float64(...) 替换为纯数字，以便 eval 解析
            clean_str = re.sub(r"np\.float64\((.*?)\)", r"\1", match.group(0))
            return eval(clean_str)
    except Exception as e:
        print(f"解析出错: {e}")
    return None

def run_experiment():
    # 生成所有参数组合
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    print(f"🚀 开始全自动调参！共计 {len(combinations)} 组实验。")
    print(f"每组运行 {NUM_SAMPLES} 条样本...\n")

    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] 正在测试参数: {params} ...")
        
        # 构造命令
        cmd = [
            "python", "scripts/main.py",
            "experiment=xsum/baseline/llama3_8b_instruct",
            "decoder=aca",
            f"data.num_samples={NUM_SAMPLES}",
            "debug=True",
            # 动态覆盖参数
            f"decoder.configs.alpha={params['alpha']}",
            f"decoder.configs.beta={params['beta']}",
            f"decoder.configs.gamma={params['gamma']}",
            f"decoder.configs.tau={params['tau']}",
            # 确保使用本地模型路径
            'model.configs.model_name_or_path="/root/DeCoRe/my_models/LLM-Research/Meta-Llama-3-8B-Instruct"'
        ]
        
        # 注入环境变量 (确保 HF 镜像生效)
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://hf-mirror.com"

        start_time = time.time()
        
        try:
            # 执行命令并捕获输出
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=env
            )
            
            # 解析结果
            metrics = parse_output(result.stdout)
            
            if metrics:
                # 计算一个综合得分 (用于排序)
                # 假设我们要平衡 ROUGE-1 和 FactKB
                # ROUGE 约 30分，FactKB 约 0.9 (即 90分)
                # 综合分 = ROUGE1 + FactKB * 30 (让两者权重差不多)
                composite_score = metrics['rouge1'] + (metrics['factKB'] * 30)
                
                record = {
                    **params,
                    "rouge1": round(metrics['rouge1'], 2),
                    "rouge2": round(metrics['rouge2'], 2),
                    "rougeL": round(metrics['rougeL'], 2),
                    "factKB": round(metrics['factKB'], 4),
                    "bert_f1": round(metrics['bertscore_f1'], 4),
                    "composite_score": round(composite_score, 2)
                }
                results.append(record)
                print(f"✅ 完成! ROUGE-1: {record['rouge1']}, FactKB: {record['factKB']}")
            else:
                print("❌ 解析失败，未找到结果。可能报错了。")
                print("最后几行日志:", result.stderr[-500:])
                
        except Exception as e:
            print(f"❌ 运行出错: {e}")

        # 实时保存到 CSV (防止中途断电白跑)
        df = pd.DataFrame(results)
        df.to_csv(RESULT_FILE, index=False)
        print(f"💾 已保存进度到 {RESULT_FILE}\n")

    print("="*50)
    print("🏆 调参结束！最佳结果前 3 名：")
    # 按综合分排序
    if not df.empty:
        df_sorted = df.sort_values(by="composite_score", ascending=False)
        print(df_sorted.head(3))
    else:
        print("没有产生有效数据。")

if __name__ == "__main__":
    run_experiment()