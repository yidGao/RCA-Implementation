import json
import re
import string
import sys

def normalize_answer(s):
    """
    标准化答案：
    1. 转小写
    2. 去除标点
    3. 去除冠词 (a, an, the)
    4. 规范化空白字符
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

def calculate_nq_score(file_path):
    print(f"📂 读取文件: {file_path}")
    
    total = 0
    correct = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                data = json.loads(line)
                total += 1
                
                # 1. 获取预测值
                prediction = data.get('predicted_answer', '')
                
                # 2. 获取标准答案列表
                # 你的数据格式是: "answers": [["May 18, 2018"]]
                # 所以我们需要取 answers[0]
                raw_answers = data.get('answers', [])
                if len(raw_answers) > 0 and isinstance(raw_answers[0], list):
                    gold_answers = raw_answers[0]
                else:
                    gold_answers = raw_answers
                
                # 3. 标准化处理
                norm_prediction = normalize_answer(prediction)
                norm_golds = [normalize_answer(g) for g in gold_answers]
                
                # 4. 判分逻辑 (EM - Exact Match)
                # 只要预测内容里 **包含** 任意一个标准答案，就算对
                # 或者预测内容 **等于** 任意一个标准答案
                is_match = False
                for gold in norm_golds:
                    if gold in norm_prediction:
                        is_match = True
                        break
                
                if is_match:
                    correct += 1
                    
    except FileNotFoundError:
        print("❌ 文件未找到，请检查路径！")
        return

    # 5. 输出结果
    score = (correct / total) * 100 if total > 0 else 0
    
    print("-" * 30)
    print(f"📊 样本总数: {total}")
    print(f"✅ 正确数量: {correct}")
    print(f"🏆 最终得分 (EM): {score:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # 请确认这是你的文件路径
    target_file = "/root/DeCoRe/outputs/2025-12-20/21-30-14/pred_NQ_LLaMA3-8b-Instruct__ACA.json"
    
    calculate_nq_score(target_file)