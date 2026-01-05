from typing import Optional
import torch
from src.models.base_model import BaseModel
from src.utils.modelling_llama import RCAConfig

class RCAModel(BaseModel):
    def __init__(
        self,
        model_configs,
        decoder_configs,
    ):
        super().__init__(model_configs, decoder_configs)
        
        self.alpha = decoder_configs.configs.get("alpha", 0.04)
        
        print(f"Initialized RCA with alpha={self.alpha}")

    def generate(self, inputs) -> dict:
        self.model.eval()

        prompt = inputs["prompted_question"][0]
        tokenised_inputs = self._verbalise_input(prompt).to(self.model.device)
        
        # 计算 Prompt 长度作为锚点
        # 减1是因为最后一个token是用来预测的，不算在"历史"Prompt里
        prompt_length = tokenised_inputs.shape[1] - 1 

        with torch.inference_mode():
            # 1. 预填充 (Prefill) - 正常跑，建立 KV Cache
            input_logits = self.model(
                input_ids=tokenised_inputs[:, :-1], 
                use_cache=True, 
                return_dict=True
            )
            
            past_kv = input_logits.past_key_values
            last_input_token = tokenised_inputs[:, -1]
            generated_ids = []
            
            # 2. 构造 RCA 配置
            rca_config = RCAConfig(
                enabled=True,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                tau=self.tau,
                prompt_length=prompt_length
            )

            # 3. 自回归解码 (Decoding) - 开启 RCA
            for _ in range(self.max_new_tokens):
                last_input_token = last_input_token.view(1, 1)
                
                # 关键：传入 rca_config
                outputs = self.model(
                    input_ids=last_input_token,
                    past_key_values=past_kv,
                    use_cache=True,
                    attn_mode="torch", # 强制使用 torch 模式以触发我们的修改
                    rca_config=rca_config 
                )
                
                past_kv = outputs.past_key_values
                
                # 贪婪解码
                next_token_logits = outputs.logits[0, -1, :]
                last_input_token = next_token_logits.argmax()
                
                generated_ids.append(last_input_token.item())
                if last_input_token.item() == self.tokenizer.eos_token_id:
                    break
            
            decoded_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )

        return {
            "decoded_text": decoded_text, 
            "attentions": {}, 
            "alphas": []
        }

    def lm_score(self, prompt, answer):
        # 1. 完全复刻 Baseline 的输入处理逻辑 (保留 Chat Template)
        prompted_question = prompt["prompted_question"][0]

        if len(prompt["verbalised_instruction"][0]):
            use_system_prompt = True
        else:
            use_system_prompt = False

        with torch.no_grad():
            # 拼接 Prompt 和 Answer
            if type(prompted_question) == list:
                input_text = prompted_question + [answer]
            else:
                input_text = prompted_question + answer
            
            # 转 Token (使用 base_model 里的标准方法，确保特殊符号正确)
            input_ids = self._verbalise_input(
                input_text,
                use_system_prompt=use_system_prompt,
                add_generation_prompt=False,
            ).to(self.model.device)
            
            prefix_ids = self._verbalise_input(
                prompted_question, use_system_prompt=use_system_prompt
            ).to(self.model.device)
            
            # 确定 Answer 部分的 token 索引
            continue_ids = input_ids[0, prefix_ids.shape[-1] :]
            
            # --- [RCA 核心插入点] ---
            # 构造配置
            prompt_len = prefix_ids.shape[-1]
            rca_config = RCAConfig(
                enabled=True,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                tau=self.tau,
                prompt_length=prompt_len - 1 # 对齐 shift
            )
            
            # 2. 模型前向传播 (注入 rca_config)
            # 注意：这里必须指定 attn_mode="torch" 才能触发我们在 modelling_llama 里改的代码
            outputs = self.model(
                input_ids, 
                attn_mode="torch", 
                rca_config=rca_config,
                use_cache=False 
            )[0].squeeze(0)
            
            outputs = outputs.log_softmax(-1)  # logits -> log probs

            # 3. 切片 (Skip Prompt)
            # 逻辑同 Baseline: 取出对应 Answer 位置的预测结果
            outputs = outputs[prefix_ids.shape[-1] - 1 : -1, :]

            # 4. 求和 (Get logprobs for answer)
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()

        return log_probs