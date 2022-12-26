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

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

class MPP_Planner(object):
    def __init__(self, opt, config, outpath, data, task_start_idx_list) -> None:
        self.model = load_model_from_config(config, f"{config.model.ckpt}")
        # model = hydra.utils.instantiate(cfg.model)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)
        self.outpath = outpath
        self.data = data
        self.task_start_idx_list = task_start_idx_list
        
        method_type=opt.model_type
        self.total_score = {}
        all_model_type_list = ['{}'.format(method_type)]
        for model_type_item in all_model_type_list:
            self.total_score[model_type_item] = {"sentence-bleu": 0, "bert-score-f": 0, "LCS": 0, "CLIP": 0}

    def closed_loop_visual_plan_generation(self):
        pass
    
    def closed_loop_textual_plan_generation(self):
        pass
            
    def start_planning(self):
        # Closed-loop Single Step Textual Plan Generation (LLM, GPT3)

        # Closed-loop Single Step Visual Plan Generation (text-to-image model, stable diffusion v2)

        # Temporal-extended multimodal procedural planning
        pass