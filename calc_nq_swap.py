import json
import re
import string
import sys
import os

def normalize_answer(s):
    """
    官方标准回答归一化逻辑
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_nq_swap(file_path):
    print(f"📂 正在读取文件: {file_path}")
    
    total = 0
    correct_sub_em = 0  # 答对篡改后的答案 (Good)
    correct_org_em = 0  # 答成原始事实 (Bad - Hallucination)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total += 1
                
                # 1. 获取预测值
                prediction = data.get('predicted_answer', '')
                norm_pred = normalize_answer(prediction)

                # 2. 获取 Sub_Answer (文档里的假答案 - 我们的目标)
                # 结构通常是 [["Answer"]]
                sub_answers_raw = data.get('sub_answer', [])
                if len(sub_answers_raw) > 0 and isinstance(sub_answers_raw[0], list):
                    sub_golds = sub_answers_raw[0]
                else:
                    sub_golds = sub_answers_raw
                
                # 3. 获取 Org_Answer (现实世界的真答案 - 记忆幻觉)
                org_answers_raw = data.get('org_answer', [])
                if len(org_answers_raw) > 0 and isinstance(org_answers_raw[0], list):
                    org_golds = org_answers_raw[0]
                else:
                    org_golds = org_answers_raw

                # 4. 归一化对比
                norm_sub_golds = [normalize_answer(g) for g in sub_golds]
                norm_org_golds = [normalize_answer(g) for g in org_golds]

                # 5. 判定 Sub_EM (包含即可)
                # NQ-Swap 只要生成的句子里包含了实体名就算对
                if any(g in norm_pred for g in norm_sub_golds):
                    correct_sub_em += 1
                
                # 6. 判定 Org_EM
                if any(g in norm_pred for g in norm_org_golds):
                    correct_org_em += 1
                    
    except FileNotFoundError:
        print("❌ 错误：找不到文件！")
        return

    if total == 0:
        print("⚠️ 空文件")
        return

    # 输出结果
    sub_score = (correct_sub_em / total) * 100
    org_score = (correct_org_em / total) * 100
    
    print("\n" + "="*40)
    print(f"🏆 NQ-Swap 最终战报 (N={total})")
    print("="*40)
    print(f"✅ Sub_EM (忠实度): {sub_score:.2f}%  <-- 目标: >60.62%")
    print(f"⚠️ Org_EM (幻觉率): {org_score:.2f}%  <-- 越低越好")
    print("="*40)

if __name__ == "__main__":
    # 请修改为你的真实文件路径
    # 比如: pred_NQSwap_LLaMA3-8b-Instruct__ACA.json
    # 如果不想手动改，可以用下面的代码自动找最新的
    import glob
    
    # 自动查找最近修改的 NQSwap json 文件
    search_path = "/root/DeCoRe/outputs/*/*/pred_NQSwap*.json"
    files = glob.glob(search_path)
    if files:
        # 按修改时间排序，找最新的
        latest_file = max(files, key=os.path.getmtime)
        calculate_nq_swap(latest_file)
    else:
        print("❌ 没找到 NQSwap 的结果文件，请手动指定路径！")