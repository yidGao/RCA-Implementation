import json
import re
import string
import sys

def normalize_answer(s):
    """标准归一化"""
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

def debug_trivia_score(file_path):
    print(f"📂 正在诊断文件: {file_path}")
    
    total = 0
    correct = 0
    errors = [] # 记录错误案例
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                total += 1
                
                prediction = data.get('predicted_answer', '')
                norm_pred = normalize_answer(prediction)
                
                # 处理嵌套答案列表
                raw_answers = data.get('answers', [])
                golds = []
                if len(raw_answers) > 0:
                    if isinstance(raw_answers[0], list):
                        golds = raw_answers[0]
                    else:
                        golds = raw_answers
                
                norm_golds = [normalize_answer(str(g)) for g in golds]
                
                # 判分
                match = False
                for g in norm_golds:
                    if g in norm_pred or norm_pred == g:
                        match = True
                        break
                
                if match:
                    correct += 1
                else:
                    # 记录前 10 个错误，方便人工检查
                    if len(errors) < 10:
                        errors.append({
                            "Question": data.get('prompted_question', '')[-1][-1] if 'prompted_question' in data else "N/A",
                            "Pred": prediction,
                            "Norm_Pred": norm_pred,
                            "Gold": golds,
                            "Norm_Gold": norm_golds
                        })
                    
    except FileNotFoundError:
        print("❌ 错误：找不到文件")
        return

    score = (correct / total) * 100
    
    print("\n" + "="*50)
    print(f"📊 TriviaQA 诊断报告 (N={total})")
    print(f"🏆 得分: {score:.2f}%")
    print("="*50)
    
    print("\n🧐 **错题抽样分析 (Top 10)**:")
    for i, err in enumerate(errors):
        print(f"\n[Case {i+1}]")
        print(f"❌ 预测: {err['Pred']}")
        print(f"✅ 答案: {err['Gold']}")
        # print(f"   (归一化对比: '{err['Norm_Pred']}' vs {err['Norm_Gold']})")
    
    print("\n" + "="*50)
    print("💡 诊断建议：")
    print("1. 如果【预测】明显是对的（如同义词），说明JSON里的【答案】缺别名 -> 脚本误判（不用担心模型）。")
    print("2. 如果【预测】完全不对（胡说八道），说明 RCA 干扰了记忆 -> 模型崩了。")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_trivia_score(sys.argv[1])
    else:
        # 这里填入你的文件路径
        target_file = "/root/DeCoRe/outputs/2025-12-21/20-31-46/pred_TriviaQA_LLaMA3-8b-Instruct__ACA.json"
        debug_trivia_score(target_file)