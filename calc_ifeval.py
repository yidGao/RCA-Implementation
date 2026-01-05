import json
import os
import sys

# 1. 把当前目录加入路径，确保能导入 src
sys.path.append(os.getcwd())

# 2. 导入 IFEval 评分工具
from src.metrics.ifeval import IFEval

def calculate_score(json_path):
    print(f"📂 正在读取答题卡: {json_path}")
    
    predictions = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
    except FileNotFoundError:
        print("❌ 找不到文件！请检查路径是否正确。")
        return

    print(f"✅ 成功读取 {len(predictions)} 条预测结果。")
    print("⏳ 正在重新阅卷 (计算分数)...")

    # 3. 初始化评估器并算分
    evaluator = IFEval()
    metrics = evaluator(predictions)

    # 4. 打印结果
    print("\n" + "="*40)
    print("🏆 IFEval 最终成绩单")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("="*40)

if __name__ == "__main__":
    # ==========================================
    # 👇👇👇 请把下面这个路径改成你实际文件的路径 👇👇👇
    # 例如: "outputs/2025-12-20/19-05-00/pred_IFEval_....json"
    file_path = "/root/DeCoRe/outputs/2025-12-20/19-05-02/pred_IFEval_LLaMA3-8b-Instruct__ACA.json" 
    # ==========================================
    
    calculate_score(file_path)