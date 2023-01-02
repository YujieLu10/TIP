import pandas as pd
from icecream import ic
import argparse
import csv
import os
from nltk.translate.bleu_score import sentence_bleu
import spacy
import wmd


parser = argparse.ArgumentParser(description='Intervention')
parser.add_argument('--load_result_path', type=str, default="", help='load result filepath')
parser.add_argument('--data_type', choices=['wikihow', 'robothow', 'wiki_inter', 'robot_inter'], default='robothow', help='choices')
parser.add_argument('--sample_for_eval', type=int, default=50)
parser.add_argument('--sample_per_worker', type=int, default=7)
parser.add_argument('--model_type', choices=['human', 'concept_knowledge', 'task_only_base', 'base', 'base_tune', 'standard_prompt', 'soft_prompt_tuning', 'chain_of_thought', 'chain_of_cause', 'cmle_ipm', 'cmle_epm', 'irm', 'vae_r', 'rwSAM', 'counterfactual_prompt'], default='base', help='choices')
args = parser.parse_args()

root_path = '/local/home/yujielu/project/GoalAgent/causal_planner_human_eval'

idx_to_methodname = ['gpt-chain', 'gpt-planner', 'gpt-concept', 'bart-chain', 'bart-planner', 'bart-concept']
nlp_encoder = spacy.load('en_core_web_md')
df_order = pd.read_csv(os.path.join(root_path, args.data_type, "{}_order.csv".format(args.data_type)))
df_plan = pd.read_csv(os.path.join(root_path, args.data_type, "{}_plan.csv".format(args.data_type)))

from typing import List, Union, Iterable
from itertools import zip_longest
from collections import defaultdict
import numpy as np
from moverscore_v2 import word_mover_score

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
    wmd = nlp_encoder(task_eval_groundtruth).similarity(nlp_encoder(task_eval_predict))
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
    return s_bleu, wmd, bert_f1, rouge_f1, bert_f1_norm, rouge_l_f1, mover


def get_csv_line(method_idx, human_program, program_idx, worker_idx):
    method_type = idx_to_methodname[method_idx]
    method_program = row['Input.task{}_txt'.format(method_idx+2)] + '. ' + str(row['Input.steps{}_txt'.format(method_idx+2)]).replace('<br>', '. ')
    # ic(method_type, method_program)
    s_bleu, wmd, bert_f1, rouge_f1, bert_f1_norm, rouge_l_f1, mover = calculate_total_score(human_program, method_program)
    human_plan = row['Answer.score{}'.format(method_idx+2)]
    # ic(program_idx, worker_idx)
    matched_idx = df_order.loc[df_order['Input.program{}_idx'.format(method_idx+2)] == program_idx].index.tolist()
    human_order = df_order._get_value(matched_idx[0], 'Answer.score{}'.format(method_idx+2))
    df_order.loc[matched_idx[0], 'Input.program{}_idx'.format(method_idx+2)] = -1

    ic(human_plan, human_order, matched_idx, bert_f1_norm)
    csv_line = [method_type] + [s_bleu] + [wmd] + [bert_f1] + [rouge_f1] + [bert_f1_norm] + [rouge_l_f1] + [mover] + [human_plan] + [human_order]
    return csv_line

with open(root_path+"/{}_metric_correlation.csv".format(args.data_type), 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    head_line = ['method_type', 's_bleu', 'wmd', 'bert_f1', 'rouge-1-f1', 'bert_f1_norm', 'rouge_l_f1', 'mover', 'human_plan', 'human_order']
    writer.writerow(head_line)
    
    csv_line = []
    df_order["is_accessed"] = [False] * 150
    for idx, row in df_plan.iterrows():
        human_program = row['Input.task1_txt'] + '. ' + row['Input.steps1_txt'].replace('<br>', '. ')
        for method_idx in range(len(idx_to_methodname)):
            program_idx = row['Input.program{}_idx'.format(method_idx+2)]
            worker_idx = row['WorkerId']
            csv_line = get_csv_line(method_idx, human_program, program_idx, worker_idx)
            writer.writerow(csv_line)
            csv_line = []
