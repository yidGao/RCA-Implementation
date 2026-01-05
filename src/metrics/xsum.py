from typing import Dict

import numpy as np
import sacrebleu
import torch
from rouge_score import rouge_scorer, scoring
import bert_score 

def bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41
    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68
    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}


class XSum:

    def __init__(self):
        self.factkb_tokenizer = None
        self.factkb_model = None
        self.bert_score = None

    def maybe_init_factkb(self):
        if self.factkb_tokenizer is None or self.factkb_model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            ##
            local_factkb_path = "/root/DeCoRe/my_models/bunsenfeng/FactKB"
            print(f"Loading FactKB from local: {local_factkb_path} ...")
            ##
            
            self.factkb_tokenizer = AutoTokenizer.from_pretrained(
                local_factkb_path, padding="max_length", truncation=True
            )
            self.factkb_model = AutoModelForSequenceClassification.from_pretrained(
                local_factkb_path, num_labels=2, device_map="auto"
            )

    def maybe_init_bertscore(self):
        if self.bert_score is None:
            from evaluate import load

            #self.bert_score = load("bertscore")
            ## 修改从本地加载
            local_metric_path = "/root/DeCoRe/metrics/bertscore" 
            
            print(f"Loading BERTScore script from: {local_metric_path}")
            self.bert_score = load(local_metric_path)

    def __call__(self, predictions: dict) -> Dict[str, float]:
        """
        Compute BLEU, ROUGE, FactKB, and BERTScore for XSum.

        Args:
            predictions (dict): Predictions of the model along with the references.

        Returns:
            Dict[str, float]: metrics
        """
        all_results = []
        for prediction in predictions:
            completion = prediction["predicted_answer"]

            document = prediction["document"][0]
            gold_summary = prediction["summary"][0]

            all_refs = [gold_summary]

            # ROUGE-N
            rouge_scores = [rouge([ref], [completion]) for ref in all_refs]
            # ROUGE-1
            rouge1_scores = [score["rouge1"] for score in rouge_scores]
            # ROUGE-2
            rouge2_scores = [score["rouge2"] for score in rouge_scores]
            # ROUGE-L
            rougeL_scores = [score["rougeLsum"] for score in rouge_scores]

            self.maybe_init_factkb()
            input_factkb = [[completion, document]]
            factkb_tokens = self.factkb_tokenizer(
                input_factkb, return_tensors="pt", padding="max_length", truncation=True
            ).to(self.factkb_model.device)
            factkb_logits = self.factkb_model(**factkb_tokens).logits
            factkb_res = torch.softmax(factkb_logits, dim=1)
        
            self.maybe_init_bertscore()

            # === 定义本地路径 ===
            local_deberta_path = "/root/DeCoRe/my_models/bunsenfeng/deberta-xlarge-mnli"

            # === 关键修复：手动注册本地路径到 bert_score 的字典里 ===
            # DeBERTa-xlarge 通常使用第 40 层作为最佳语义表示
            if local_deberta_path not in bert_score.utils.model2layers:
                bert_score.utils.model2layers[local_deberta_path] = 40 


            bert_score_res = self.bert_score.compute(
                predictions=[completion],
                references=[gold_summary],
                model_type="/root/DeCoRe/my_models/bunsenfeng/deberta-xlarge-mnli",
                lang="en",
            )

            res = {
                "rouge1": rouge1_scores[0],
                "rouge2": rouge2_scores[0],
                "rougeL": rougeL_scores[0],
                "factKB": float(factkb_res[0][1]),
                "bertscore_precision": float(bert_score_res["precision"][0]),
                "bertscore_recall": float(bert_score_res["recall"][0]),
                "bertscore_f1": float(bert_score_res["f1"][0]),
            }

            all_results.append(res)

        metrics = {
            key: np.mean([res[key] for res in all_results]) for key in all_results[0]
        }

        return metrics