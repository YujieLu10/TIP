import openai
import numpy as np
import torch
import json
from icecream import ic
import pandas as pd
import os
# from dotenv import load_dotenv
import openai
from evaluators.automatic_eval import Automatic_Evaluator

# load_dotenv()
# openai.api_key = opt.api_key #"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"

LMTYPE_TO_LMID = {
    "gpt3": "gpt3",
    "gpt-j-6B": "EleutherAI/gpt-j-6B",
    "gpt2-1.5B": "gpt2",
    "gpt2": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "gpt_neo": "EleutherAI/gpt-neo-1.3B",
    "t5": "t5-3b",
    "bert": "bert-large-uncased",
    "roberta": "roberta-large",
    "bart": "facebook/bart-large"
}

class LM_Engine(object):
    def __init__(self, model, planning_lm_id, device):
        self.model = model
        self.planning_lm_id = planning_lm_id

    def generate(self, prompt, sampling_params):
        response = openai.Completion.create(engine=self.planning_lm_id, prompt=prompt, **sampling_params)
        generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
        # calculate mean log prob across tokens
        mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(sampling_params['n'])]
        return generated_samples, mean_log_probs
            
class LLM_Reasoning(object):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.completion = openai.Completion()
        self.language_model_type = opt.language_model_type
        self.open_loop = opt.open_loop
        self.model_type = opt.model_type
        self.model = None
        self.planning_lm_id = LMTYPE_TO_LMID[opt.language_model_type] #'gpt2-large'  # see comments above for all options
        self.translation_lm_id = 'stsb-roberta-large'  # see comments above for all options
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = LM_Engine(self.model, self.planning_lm_id, self.device)
        self.result_list = []
        self.data_type = opt.data_type
        self.max_tokens = opt.max_tokens
        
        openai.api_key = opt.api_key
        self.sampling_params = \
                {
                    "max_tokens": self.max_tokens,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "n": 10,
                    "logprobs": 1,
                    "presence_penalty": 0.5,
                    "frequency_penalty": 0.3,
                    "stop": '\n'
                }
        if opt.do_eval_each:
            self.total_score_cal = {}
            method_type=opt.model_type
            all_model_type_list = ['{}'.format(method_type)]
            for model_type_item in all_model_type_list:
                self.total_score_cal[model_type_item] = {"sentence-bleu": 0, "wmd": 0, "rouge-1-f": 0, "rouge-1-p": 0, "rouge-1-r": 0, "bert-score-f": 0, "bert-score-p": 0, "bert-score-r": 0, "meteor": 0, "sentence-bert-score": 0, "caption-t-bleu": 0, "LCS": 0, "caption-t-bleu": 0, "caption-vcap-lcs": 0, "gpt3-plan-accuracy": 0, "caption-gpt3-plan-accuracy": 0, "vplan-t-clip-score": 0, "tplan-v-clip-score": 0, "vplan-v-clip-score": 0, "tplan-t-clip-score": 0}
            self.lm_automatic_evaluator = Automatic_Evaluator(self.opt)
    
    def ask_openloop(self, task_eval_predict, curr_prompt):
        if self.model_type == "task_only_base":
            prompt = "what is the step-by-step procedure of " + task_eval_predict + " without explanation "
        else:
            # for robothow PLAN
            # prompt = "What is the step-by-step procedure of " + curr_prompt + "\nWhat is the step-by-step procedure of " + task_eval_predict + " without explanation "
            # prompt = curr_prompt + ", what is the step-by-step procedure of " + task_eval_predict + " without explanation "
            # prompt = "Refer to the possible procedure: " + curr_prompt + ", what is the step-by-step procedure of " + task_eval_predict + " without explanation "
            # prompt = curr_prompt + ", what is the executable step-by-step robot plan of " + task_eval_predict + " without explanation "
            prompt = "A possible procedural plan is: " + curr_prompt + ", think of implementing " + task_eval_predict + " within 5 steps "
        response = self.completion.create(
            prompt=prompt, engine="text-davinci-002", temperature=0.7,
            top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
            max_tokens=120 if self.data_type == "wikihow" else self.max_tokens)
        answer = response.choices[0].text.strip().strip('-').strip('_')
        # ic(answer)
        return answer.split('\n')

    def ask(self, prompt):
        prompt = "what is the step-by-step procedure of " + prompt + " without explanation "
        ic(prompt)
        import time
        time.sleep(3)
        try:
            response = self.completion.create(
                prompt=prompt, engine="text-davinci-002", temperature=0.7,
                top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
                max_tokens=30)
        except:
            time.sleep(10)
            try:
                response = self.completion.create(
                    prompt=prompt, engine="text-davinci-002", temperature=0.7,
                    top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
                    max_tokens=30)
            except:
                time.sleep(20)
                response = self.completion.create(
                    prompt=prompt, engine="text-davinci-002", temperature=0.7,
                    top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
                    max_tokens=30)
        ic(response.choices[0])
        answer = response.choices[0].text.strip().strip('-').strip('_')
        ic(answer)
        return answer

    def language_planning(self, total_score_cal, data_example, epoch_i=0):
        # global example_task_embedding
        # global action_list_embedding

        task = data_example["tasks"]# if not opt.model_type == "concept_knowledge" else "Hang up jacket in bedroom"

        if self.model_type == "task_only_base":
            curr_prompt = task+'.' #+ "<|endoftext|>" 
        # else:
        #     # find most relevant example        
        #     example_idx, _ = find_most_similar(task, example_task_embedding)
        #     example = heldout_available_examples[example_idx]
        #     ic(len(example_task_embedding), example)
        #     # construct initial prompt
        #     curr_prompt = f'{example}\n\n{task}.'
        

        self.result_list.append('\n' + '-'*10 + ' GIVEN EXAMPLE ' + '-'*10+'\n')
        task_eval_groundtruth = task + '. ' + str(data_example["steps"])
        self.result_list.append(task_eval_groundtruth)

        task_eval_predict = task + ". "

        self.result_list.append(f'{task}.')
        step_sequence = []
        if self.open_loop:
            generated_list = self.ask_openloop(task_eval_predict, curr_prompt)
            ic(generated_list)
            translated_list = []
            for step_idx, each_step in enumerate(generated_list):
                # most_similar_idx, matching_score = find_most_similar(each_step, action_list_embedding)
                # translated_action = action_list[most_similar_idx]
                # if matching_score < self.cut_threshold: continue
                
                best_action = each_step # translated_action
                formatted_action = best_action # (best_action[0].upper() + best_action[1:]).replace('_', ' ')
                step_sequence.append(formatted_action)
                step_idx += 1
                translated_list.append(f' Step {step_idx}: {formatted_action}.')
            self.result_list.append(" ".join(translated_list.copy()))
            task_eval_predict += " ".join(translated_list.copy())
        ic(task_eval_groundtruth, task_eval_predict)
        # ic(len(task_eval_groundtruth.split('.')), len(task_eval_predict.split('.')))
        # ic(model_type, task, sentence_bleu([task_eval_groundtruth.split()], task_eval_predict.split()), nlp_encoder(task_eval_groundtruth).similarity(nlp_encoder(task_eval_predict)))
        # ic(task, len(task_eval_groundtruth), len(task_eval_predict))
        return self.lm_automatic_evaluator.calculate_total_score(total_score_cal=total_score_cal, task_eval_groundtruth=task_eval_groundtruth, task_eval_predict=task_eval_predict), self.result_list


    def generate_language_plan(self, opt, task_result_dir, summarize_example_data_list):
        if opt.do_eval_each:
            if not os.path.isdir(task_result_dir): os.makedirs(task_result_dir)
            skip_count = 0
            with open(os.path.join(task_result_dir, "{}_task_result.txt".format(opt.language_model_type)), 'w') as resultfile:
                for data_example in summarize_example_data_list[1:]:
                    
                    self.total_score_cal, result_list = self.language_planning(self.total_score_cal, data_example)

                # mean value
                ic(len(summarize_example_data_list), self.total_score_cal[opt.model_type].keys())
                for score_key in self.total_score_cal[opt.model_type].keys():
                    self.total_score_cal[opt.model_type][score_key] /= (len(summarize_example_data_list)-skip_count)
                resultfile.writelines(result_list)
                json.dump(self.total_score_cal,resultfile)
                ic(skip_count, self.total_score_cal[opt.model_type])
        else:
            # TODO: write down step_n.txt, and full_prediction.txt
            pass
            
    