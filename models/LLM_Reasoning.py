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
from tqdm import tqdm
import glob

MAX_STEPS = 22
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

t2i_template_dict = {
    "t2i-0": "What do I need to draw in the picture to describe the above text?",
    "t2i-1": "What do you see in the figure?",
    "t2i-2": "Let's think about what we need to visualize to present the above idea.",
    "t2i-3": "Describe what the picture corresponding to the text should have.",
    "t2i-4": "What do you usually draw?",
    "t2i-5": "Describe something irrelevant to the above text.",
}
        
i2t_template_dict = {
    "i2t-0": "Rewrite the textual instruction using the knowledge from visualized instruction pair-wisely. Please keep the same number of steps as the provided procedure.",
    "i2t-1": "Revise each step according to the visual imagination. Please keep the same number of steps as the provided procedure.",
    "i2t-2": "Let's revise the procedure using the captions. Please keep the same number of steps as the provided procedure.",
    "i2t-3": "Based on the visual caption, can you revise the step-by-step procedure according to the paired captions? Please keep the same number of steps as the provided procedure.",
    "i2t-4": "Provide an interesting procedure to be irrelevant with the captions. Please keep the same number of steps as the provided procedure.",
    "i2t-5": "Give the textual instruction that disobey the visual captions. Please keep the same number of steps as the provided procedure."
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
        self.data_type = opt.data_type
        self.max_tokens = opt.max_tokens
        
        # m-plan
        self.task_prediced_steps = []
        self.curr_prompt = ""
        self.result_list = []
        self.step_sequence = []
        self.task_eval_predict = ""

        openai.api_key = opt.api_key
        self.sampling_params = \
                {
                    "max_tokens": self.max_tokens,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "n": 10,
                    "logprobs": 1,
                    "presence_penalty": 1,
                    "frequency_penalty": 1,
                    "stop": '\n'
                }
        if opt.do_eval_each:
            self.total_score_cal = {"sentence-bleu": 0, "wmd": 0, "rouge-1-f": 0, "rouge-1-p": 0, "rouge-1-r": 0, "bert-score-f": 0, "bert-score-p": 0, "bert-score-r": 0, "meteor": 0, "sentence-bert-score": 0, "caption-t-bleu": 0, "LCS": 0, "caption-t-bleu": 0, "caption-vcap-lcs": 0, "gpt3-plan-accuracy": 0, "caption-gpt3-plan-accuracy": 0, "vplan-t-clip-score": 0, "tplan-v-clip-score": 0, "vplan-v-clip-score": 0, "tplan-t-clip-score": 0}

    
    def get_revision_plan(self, sample_path, current_revision_sample, template_dict, i2t_bridge):
        template = i2t_template_dict[i2t_bridge]
        task = current_revision_sample["tasks"]
        ori_steps = '\n'.join(current_revision_sample["steps"])
        captions = '\n'.join(current_revision_sample["captions"])
        if i2t_bridge == "i2t-0":
            prompt = f"Textual Instruction:\n{task}\n{ori_steps}\nVisualized Instruction:\n{captions}\n{template}"
        elif i2t_bridge == "i2t-1":
            prompt = f"Plan:\n{task}\n{ori_steps}\nVisual Imagination:\n{captions}\n{template}"
        elif i2t_bridge == "i2t-2":
            prompt = f"{task}\n{ori_steps}\n{captions}\n{template}"
        elif i2t_bridge in ["i2t-3", "i2t-4", "i2t-5"]:
            prompt = f"Step-by-step Procedure:\n{task}\n{ori_steps}\nPaired Captions:\n{captions}\n{template}"
        else:
            prompt = f"{task}\n{ori_steps}\n{captions}\n{template}"
            
        response = self.completion.create(
            prompt=prompt, engine="text-davinci-003", temperature=0.7,
            top_p=1, frequency_penalty=1, presence_penalty=1, best_of=1,
            max_tokens=self.max_tokens)
        answer = response.choices[0].text.strip().strip('-').strip('_').split('\n')
        # ic(i2t_bridge, answer)
        if not "Step" in answer[0]: # i2t_bridge in ["i2t-0", "i2t-1", "i2t-2", "i2t-3"] and 
            answer = answer[1:]
        step_num = len(glob.glob1(sample_path,f"step_[0-9]*_bridget2i-0_caption.txt")) or len(glob.glob1(sample_path,"step_[0-9]*_bridge_caption.txt")) or len(glob.glob1(sample_path,"step_[0-9]*_caption.txt")) or len(glob.glob1(sample_path,"step_[0-9]*.txt"))
        step_num = min(step_num, len(answer))
        for step_idx in range(1, step_num+1):
            try:
                with open(os.path.join(sample_path, "step_{}_bridge{}_tplan.txt".format(str(step_idx), str(i2t_bridge) if (self.opt.t2i_template_check or self.opt.i2t_template_check) else "")), 'w') as f:
                    f.write(answer[step_idx-1])
            except:
                break

    
    def visual_plan_conditioned_textual_plan_revision(self, outpath, before_revision_example_list):
        for task_idx in tqdm(range(self.opt.task_num)):
            sample_path = os.path.join(outpath, f"task_{task_idx}")
            current_revision_sample = before_revision_example_list[task_idx]
            if self.opt.i2t_template_check:
                for i2t_bridge in i2t_template_dict.keys():
                    self.get_revision_plan(sample_path, current_revision_sample, i2t_template_dict, i2t_bridge)
            else:
                self.get_revision_plan(sample_path, current_revision_sample, i2t_template_dict, self.opt.i2t_bridge)
    
    def ask_visual_prompt(self, input_text, t2i_bridge):
        """
        - Physical Action
        Option 1:
        The task is how to surf. Learn to stand on the board properly

        What do I need to draw in the picture to describe the above text to show the action?

        In the picture, you would draw a person standing on a surfboard with their feet spread apart and arms outstretched. You could also draw waves in the background to show the person surfing.
        
        Option 2:
        pick up the wine glass

        What do I need to draw in the picture to describe the above text to show the action?

        You would need to draw a person with their hand reaching out to pick up a wine glass.
        
        - Visual Description
        The task is how to surf. Practice regularly. The more you practice, the better and more confident you will become.

        What do I need to draw in the picture to describe the above text?

        In the picture, you could draw a person standing on a surfboard in the ocean, with waves in the background. You could also draw a sun in the sky to indicate the time of day. Finally, you could add a speech bubble with the words "Practice regularly" to emphasize the importance of regular practice.
        
        - Plan Consistency with incorporation of Task
        use_task_hint
        
        - Negative Prompt
        Prompt: a person stands and a dog sits
        Negative Prompt:a person sits

        """
        prompt = ''.join(input_text) + "\n{}".format(t2i_template_dict[t2i_bridge])
        try:
            response = self.completion.create(
                prompt=prompt, engine="text-davinci-003", temperature=0.7,
                top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
                max_tokens=50) # less max tokens for SD prompt generation
        except:
            import time
            time.sleep(60)
            response = self.completion.create(
                prompt=prompt, engine="text-davinci-003", temperature=0.7,
                top_p=1, frequency_penalty=0, presence_penalty=0, best_of=1,
                max_tokens=50) # less max tokens for SD prompt generation
        answer = response.choices[0].text.strip().strip('-').strip('_').strip('\n')
        if t2i_bridge == "t2i-0" and "draw " in answer:
            answer = answer[answer.index("draw")+4:]
        if t2i_bridge == "t2i-1" and "figure" in answer:
            answer = answer[answer.index("figure")+7:]
        return answer

    def ask_openloop(self, task_eval_predict, curr_prompt):
        if self.model_type == "task_only_base":
            prompt = "what is the step-by-step procedure of " + task_eval_predict + " without explanation "
        else:
            prompt = "A possible procedural plan is: " + curr_prompt + ", think of implementing " + task_eval_predict + " within 5 steps "
        response = self.completion.create(
            prompt=prompt, engine="text-davinci-003", temperature=0.7,
            top_p=1, frequency_penalty=1, presence_penalty=1, best_of=1,
            max_tokens=self.max_tokens)
        answer = response.choices[0].text.strip().strip('-').strip('_').split('\n')
        return [f"Step {item}" for item in answer if len(item) > 3]

    def ask(self, prompt, step_idx):
        if step_idx == 1:
            prompt = prompt + "\n what is the first step of above task in one sentence?"
        else:
            prompt = prompt + "\n what is the next step of the above procedure in one sentence? Reply \"END\" if the procedure is complete and already reached the final step."
            ic(prompt)
        
        import time
        time.sleep(3)
        try:
            response = self.completion.create(
                prompt=prompt, engine="text-davinci-003", temperature=0.7,
                top_p=1, frequency_penalty=1, presence_penalty=1, best_of=1,
                max_tokens=30)
        except:
            time.sleep(10)
            try:
                response = self.completion.create(
                    prompt=prompt, engine="text-davinci-003", temperature=0.7,
                    top_p=1, frequency_penalty=1, presence_penalty=1, best_of=1,
                    max_tokens=30)
            except:
                time.sleep(20)
                response = self.completion.create(
                    prompt=prompt, engine="text-davinci-003", temperature=0.7,
                    top_p=1, frequency_penalty=1, presence_penalty=1, best_of=1,
                    max_tokens=30)
        answer = response.choices[0].text.strip().strip('-').strip('_').strip('\n').strip('\t')
        ic(answer)
        return answer

    def language_planning(self, total_score_cal, data_example, sample_result_dir="", write_step_result=False):
        task = data_example["tasks"]
        if write_step_result:
            with open(os.path.join(sample_result_dir, f"task.txt"), 'w') as f:
                f.write(f"{task}")
        if self.model_type == "task_only_base":
            curr_prompt = task+'.'

        self.result_list.append('\n' + '-'*10 + ' GIVEN EXAMPLE ' + '-'*10+'\n')
        task_eval_groundtruth = task + '. ' + str(data_example["steps"])
        self.result_list.append(task_eval_groundtruth)

        task_eval_predict = task + ". "

        self.result_list.append(f'{task}.')
        step_sequence = []
        if True:
            generated_list = self.ask_openloop(task_eval_predict, curr_prompt)
            ic(generated_list)
            translated_list = []
            for step_idx, each_step in enumerate(generated_list):
                
                best_action = each_step # translated_action
                formatted_action = best_action # (best_action[0].upper() + best_action[1:]).replace('_', ' ')
                step_sequence.append(formatted_action)
                translated_list.append(f' Step {step_idx+1}: {formatted_action}.')
                if write_step_result:
                    with open(os.path.join(sample_result_dir, "step_{}.txt".format(str(step_idx+1))), 'w') as f:
                        f.write(f"{formatted_action}")
            self.result_list.append(" ".join(translated_list.copy()))
            task_eval_predict += " ".join(translated_list.copy())
        return self.result_list


    def generate_language_plan(self, opt, task_result_dir, summarize_example_data_list, sample=None, step_idx=-1):
        if opt.do_eval_each:
            if not os.path.isdir(task_result_dir): os.makedirs(task_result_dir)
            with open(os.path.join(task_result_dir, "{}_task_result.txt".format(opt.language_model_type)), 'w') as resultfile:
                for data_example in summarize_example_data_list[1:]:
                    
                    self.total_score_cal, result_list = self.language_planning(self.total_score_cal, data_example)

                for score_key in self.total_score_cal[opt.model_type].keys():
                    self.total_score_cal[opt.model_type][score_key] /= (len(summarize_example_data_list))
                resultfile.writelines(result_list)
                json.dump(self.total_score_cal,resultfile)
                ic(self.total_score_cal[opt.model_type])
        else:
            if opt.task in ["c-plan"]:
                if step_idx == 1:
                    self.curr_prompt = sample["tasks"]
                    os.makedirs(task_result_dir, exist_ok=True)
                    with open(os.path.join(task_result_dir, "task.txt"), 'w') as f:
                        f.write(f"{self.curr_prompt}")
                best_action = self.ask(self.curr_prompt, step_idx)
                plan_termination = (step_idx > MAX_STEPS) or ("END" in best_action) or ("End" in best_action and len(best_action) < 5)
                if plan_termination: return plan_termination, None
                self.task_prediced_steps.append(best_action)
                formatted_action = f'Step {step_idx}: {best_action}' if not "Step" in best_action else best_action
                with open(os.path.join(task_result_dir, "step_{}.txt".format(str(step_idx))), 'w') as f:
                    f.write(f"{formatted_action}")
                self.curr_prompt += f'\n{formatted_action}'
                self.result_list.append(f'{formatted_action}')
                self.step_sequence.append(best_action)
                self.task_eval_predict += f' {formatted_action}'
                return plan_termination, best_action 
            else:
                for task_idx, data_example in enumerate(summarize_example_data_list):
                    exist_task_num = len(os.listdir(task_result_dir))
                    start_task_idx = (task_idx + exist_task_num - 1) if self.opt.resume else task_idx
                    sample_result_dir = os.path.join(task_result_dir, "task_{}".format(str(start_task_idx)))
                    if not os.path.isdir(sample_result_dir): os.makedirs(sample_result_dir)
                    result_list = self.language_planning(None, data_example, sample_result_dir=sample_result_dir, write_step_result=True)
                