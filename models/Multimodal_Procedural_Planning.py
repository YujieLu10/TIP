import os
import torch
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from stablediffusion.scripts.mpp_utils.prompt_process import load_prompt

import hydra
from omegaconf import DictConfig, OmegaConf
from icecream import ic

import cv2
from PIL import Image
import numpy as np
import json

from LLM_Reasoning import LLM_Reasoning
import openai
from Image_Generation import Image_Generation


class MPP_Planner(object):
    """Option1: Closed-loop Procedural Planning Option2: first generate textual plan candidate pool, and then use captions of text-to-image visual plan and prompt GPT3 whether it can complete the task. and then rerank textual plan according visual plan correctness"""
    def __init__(self, opt, config, outpath) -> None:
        self.outpath = outpath
        self.data, self.task_start_idx_list, self.summarize_example_data_list = load_prompt(opt, config)

        method_type=opt.model_type
        self.total_score = {}
        all_model_type_list = ['{}'.format(method_type)]
        for model_type_item in all_model_type_list:
            self.total_score[model_type_item] = {"sentence-bleu": 0, "bert-score-f": 0, "LCS": 0, "CLIP": 0}
        self.completion = openai.Completion()

        self.task_prediced_steps = []
        self.curr_prompt = ""
        self.result_list = []
        self.step_sequence = []
        self.task_eval_predict = ""

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
        answer = response.choices[0].text.strip().strip('-').strip('_')
        ic(answer)
        return answer
    
    def closed_loop_textual_plan_generation(self, step_idx):
        # TODO: complete implementation
        MAX_STEPS = 20
        # Closed-loop Single Step Textual Plan Generation (LLM, GPT3)
        best_action, probability = self.ask(self.curr_prompt)
        plan_termination = probability < 0.5
        if plan_termination: return plan_termination

        self.task_prediced_steps.append(best_action)
        self.curr_prompt += f'\nStep {step_idx}: {best_action}.'
        self.result_list.append(f'Step {step_idx}: {best_action}.')
        self.step_sequence.append(best_action)
        self.task_eval_predict += f'Step {step_idx}: {best_action}.'        
        return plan_termination


    def closed_loop_visual_plan_generation(self, step_idx):
        # TODO: complete implementation
        # Closed-loop Single Step Visual Plan Generation (text-to-image model, stable diffusion v2)
        opt = self.opt
        task_start_idx_list = []
        batch_size = self.opt.n_samples
        prompt = self.next_prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
        
        image_generator = Image_Generation(opt)
        image_generator.generate_image(opt, data, task_start_idx_list)

 
    def temporal_extended_mpp(self):
        # TODO: complete implementation
        # Temporal-extended multimodal procedural planning
        step_idx = 0
        while True:
            step_idx += 1
            plan_termination = self.closed_loop_textual_plan_generation(step_idx)
            if plan_termination: break
            self.closed_loop_visual_plan_generation(step_idx)
            

    def start_planning(self):
        # TODO: complete implementation
        # load task list
        for task in self.summarize_example_data_list[:self.opt.task_num]:
            self.temporal_extended_mpp()