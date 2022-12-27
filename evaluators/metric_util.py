import spacy
import nltk
from rouge import Rouge
from datasets import load_metric
from typing import List
# import gensim.downloader as api
from sentence_transformers import util as st_utils
from moverscore_v2 import word_mover_score
from collections import defaultdict, Counter
import numpy as np
# model = api.load('word2vec-google-news-300')
nlp_encoder = spacy.load('en_core_web_md')

def calc_text_distance(query_str, value_str, translation_lm, device):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    value_embedding = translation_lm.encode(value_str, convert_to_tensor=True, device=device)
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, value_embedding)[0].detach().cpu().numpy()
    return cos_scores

def calc_textemb_distance(query_emb, value_emb):
    cos_scores = st_utils.pytorch_cos_sim(query_emb, value_emb)[0].detach().cpu().numpy()
    return float(cos_scores)

def get_metric_result(task, predicted_intent, task_eval_groundtruth, task_eval_predict, translation_lm, device):
    task_eval_groundtruth = task_eval_groundtruth.replace('.', ' ')
    task_eval_predict = task_eval_predict.replace('.', ' ')
    sentence_bleu = nltk.translate.bleu_score.sentence_bleu([task_eval_groundtruth.split()], task_eval_predict.split())
    sim = nlp_encoder(task_eval_groundtruth).similarity(nlp_encoder(task_eval_predict))
    rouge = Rouge()
    scores = rouge.get_scores(task_eval_predict, task_eval_groundtruth)
    # total_score_cal[model_type]["rouge-1-f"] += scores[0]["rouge-1"]["f"]
    # total_score_cal[model_type]["rouge-1-p"] += scores[0]["rouge-1"]["p"]
    # total_score_cal[model_type]["rouge-1-r"] += scores[0]["rouge-1"]["r"]
    # use rouge-l instead of rouge-1
    rouge_l_f = scores[0]["rouge-l"]["f"]
    bertscore = load_metric("bertscore")
    try:
        bert_results = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased")
        # bert-score = bert_results["f1"][0]
        # use normalized bert-score
        bert_score_norm = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased", lang="en", verbose=False, rescale_with_baseline=True)["f1"][0]
    except:
        bert_score_norm = 0
    intent_score = calc_text_distance(task, predicted_intent, translation_lm, device)
    return sentence_bleu, sim, rouge_l_f, bert_score_norm, float(intent_score)


# def calculate_total_score(total_score_cal, model_type, task_eval_groundtruth, task_eval_predict): 
#     task_eval_groundtruth = task_eval_groundtruth.replace('.', ' ')
#     task_eval_predict = task_eval_predict.replace('.', ' ')
#     total_score_cal[model_type]["sentence-bleu"] += nltk.translate.bleu_score.sentence_bleu([task_eval_groundtruth.split()], task_eval_predict.split())
#     # total_score_cal[model_type]["wmd"] += model.wmdistance(task_eval_groundtruth, task_eval_predict)
#     total_score_cal[model_type]["sim"] += nlp_encoder(task_eval_groundtruth).similarity(nlp_encoder(task_eval_predict))
#     rouge = Rouge()
#     scores = rouge.get_scores(task_eval_predict, task_eval_groundtruth)
#     total_score_cal[model_type]["rouge-1-f"] += scores[0]["rouge-1"]["f"]
#     total_score_cal[model_type]["rouge-1-p"] += scores[0]["rouge-1"]["p"]
#     total_score_cal[model_type]["rouge-1-r"] += scores[0]["rouge-1"]["r"]
#     bertscore = load_metric("bertscore")
#     try:
#         bert_results = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased")
#         total_score_cal[model_type]["bert-score-f"] += bert_results["f1"][0]
#         total_score_cal[model_type]["bert-score-p"] += bert_results["precision"][0]
#         total_score_cal[model_type]["bert-score-r"] += bert_results["recall"][0]
#     except:
#         pass
#     return total_score_cal

def sentence_score(hypothesis: str, references: List[str], trace=0):
    
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    
    hypothesis = [hypothesis] * len(references)
    
    sentence_score = 0 

    scores = word_mover_score(references, hypothesis, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=False)
    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score

def calculate_total_score(task_eval_groundtruth, task_eval_predict):
    import nltk
    s_bleu = nltk.translate.bleu_score.sentence_bleu([task_eval_groundtruth.split()], task_eval_predict.split())
    # total_score[model_type]["corpus-bleu"] += nltk.translate.bleu_score.corpus_bleu([task_eval_groundtruth.split('.')], task_eval_predict.split('.'))
    sim = nlp_encoder(task_eval_groundtruth).similarity(nlp_encoder(task_eval_predict))
    from moverscore_v2 import get_idf_dict, word_mover_score 
    from collections import defaultdict
    # Source and reference streams have different lengths! use sentence_score
    # mover = corpus_score(task_eval_predict.split('.'), [task_eval_groundtruth.split('.')])
    mover = sentence_score(task_eval_predict, task_eval_groundtruth)
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(task_eval_predict, task_eval_groundtruth)
    rouge_f1 = scores[0]["rouge-1"]["f"]
    rouge_l_f1 = scores[0]["rouge-l"]["f"]
    from datasets import load_metric
    bertscore = load_metric("bertscore")
    bert_results = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased")
    bert_f1 = bert_results["f1"][0]
    bert_f1_norm = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased", lang="en", verbose=False, rescale_with_baseline=True)["f1"][0]
    return s_bleu, sim, bert_f1, rouge_f1, bert_f1_norm, rouge_l_f1, mover


def get_metric_csv_line(model_type, task_eval_groundtruth, task_eval_predict, metric_intent_score):
    s_bleu, sim, bert_f1, rouge_f1, bert_f1_norm, rouge_l_f1, mover = calculate_total_score(task_eval_groundtruth, task_eval_predict)
    avg_length = task_eval_predict.count("step") + task_eval_predict.count("Step")
    # from icecream import ic
    # ic(task_eval_predict, task_eval_predict.count("Step"))
    from icecream import ic
    # ic(s_bleu, sim, bert_f1, rouge_f1, bert_f1_norm, rouge_l_f1, mover, metric_intent_score)
    csv_line = [model_type] + [s_bleu] + [sim] + [bert_f1] + [rouge_f1] + [bert_f1_norm] + [rouge_l_f1] + [mover] + metric_intent_score + [avg_length]
    return csv_line

# def get_csv_line(method_idx, human_program, program_idx, worker_idx):
#     method_type = idx_to_methodname[method_idx]
#     method_program = row['Input.task{}_txt'.format(method_idx+2)] + '. ' + str(row['Input.steps{}_txt'.format(method_idx+2)]).replace('<br>', '. ')
#     # ic(method_type, method_program)
#     s_bleu, wmd, bert_f1, rouge_f1, bert_f1_norm, rouge_l_f1, mover = calculate_total_score(human_program, method_program)
#     human_plan = row['Answer.score{}'.format(method_idx+2)]
#     # ic(program_idx, worker_idx)
#     matched_idx = df_order.loc[df_order['Input.program{}_idx'.format(method_idx+2)] == program_idx].index.tolist()
#     human_order = df_order._get_value(matched_idx[0], 'Answer.score{}'.format(method_idx+2))
#     df_order.loc[matched_idx[0], 'Input.program{}_idx'.format(method_idx+2)] = -1

#     ic(human_plan, human_order, matched_idx, bert_f1_norm)
#     csv_line = [method_type] + [s_bleu] + [wmd] + [bert_f1] + [rouge_f1] + [bert_f1_norm] + [rouge_l_f1] + [mover] + [human_plan] + [human_order]
#     return csv_line