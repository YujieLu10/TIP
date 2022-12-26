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

class Baseline_Planner(object):
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

    def open_loop_visual_plan_generation(self, opt):
        if self.config.mpp_model.task_config.ground_truth_modality == "textual": # text to image generation
            model = self.model
            if opt.plms:
                sampler = PLMSSampler(model)
            elif opt.dpm:
                sampler = DPMSolverSampler(model)
            else:
                sampler = DDIMSampler(model)

            print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
            wm = "SDV2"
            wm_encoder = WatermarkEncoder()
            wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

            batch_size = opt.n_samples
            n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

            sample_path = os.path.join(self.outpath, opt.data_type, opt.task) # os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)
            sample_count = 0
            task_count = 0
            step_count = 0
            all_step_count = 0
            base_count = len(os.listdir(sample_path))
            # grid_count = len(os.listdir(outpath)) - 1

            start_code = None
            if opt.fixed_code:
                start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=self.device)

            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            with torch.no_grad(), \
                precision_scope("cuda"), \
                model.ema_scope():
                    # all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(self.data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)
                            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            samples, _ = sampler.sample(S=opt.steps,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                            for x_sample, x_prompt in zip(x_samples, prompts):
                                sample_path = os.path.join(self.outpath, opt.data_type, opt.task, "task_{}".format(task_count))
                                os.makedirs(sample_path, exist_ok=True)
                                if all_step_count in self.task_start_idx_list:
                                    if opt.save_task_grid:
                                        # save previous task as grid
                                        grid = torch.stack(all_samples[1:], 0)
                                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                                        grid = make_grid(grid, nrow=len(all_samples)-1)

                                        # to image
                                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                                        grid = Image.fromarray(grid.astype(np.uint8))
                                        grid = put_watermark(grid, wm_encoder)
                                        grid.save(os.path.join(sample_path, f'task-grid-{task_count}.png'))
                                        # grid_count += 1
                                    
                                    all_samples = list()
                                    task_count += 1
                                    step_count = 0

                                all_samples.append(x_sample)
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                # img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                img.save(os.path.join(sample_path, f"step_{step_count}.png"))
                                with open(os.path.join(sample_path, f"step_{step_count}.txt"), 'w') as f:
                                    f.write(f"{x_prompt}")
                                step_count += 1
                                all_step_count += 1 
                                base_count += 1
                                sample_count += 1

            print(f"Your samples are ready and waiting for you here: \n{self.outpath} \n"
                f" \nEnjoy.")
        else:
            pass
    
    def open_loop_textual_plan_generation(self, opt, outpath, summarize_example_data_list):
        if self.config.mpp_model.task_config.ground_truth_modality == "visual": # image caption
            pass
        else:
            llm_reasoning_engine = LLM_Reasoning(opt)
            # task_path = opt.data_type + "/" + method_type + "/" + datetime.now().strftime("%Y_%m_%d") + "/" + "demo_{}_{}_inter{}_var{}_heldout{}_oloop{}_".format(opt.language_model_type, opt.model_type, opt.open_loop) + datetime.now().strftime("%H%M%S")
            # task_result_dir = os.path.join("../result", task_path)
            task_result_dir = os.path.join(outpath, opt.data_type, opt.task)
            if not os.path.isdir(task_result_dir): os.makedirs(task_result_dir)
            skip_count = 0
            with open(os.path.join(task_result_dir, "{}_task_result.txt".format(opt.language_model_type)), 'w') as resultfile:
                total_score_cal = self.total_score.copy()
                for data_example in summarize_example_data_list[1:]:
                    
                    total_score_cal, result_list = llm_reasoning_engine.language_planning(total_score_cal, data_example)

                # mean value
                ic(len(summarize_example_data_list), total_score_cal[opt.model_type].keys())
                for score_key in total_score_cal[opt.model_type].keys():
                    total_score_cal[opt.model_type][score_key] /= (len(summarize_example_data_list)-skip_count)
                resultfile.writelines(result_list)
                json.dump(total_score_cal,resultfile)
                ic(skip_count, total_score_cal[opt.model_type])
                
    def start_planning(self):
        if self.opt.task == "u-plan":
            self.open_loop_textual_plan_generation(self.opt, self.outpath, self.summarize_example_data_list)
            self.open_loop_visual_plan_generation(self.opt)
        elif self.opt.task == "tgt-u-plan": # text to image generation
            self.open_loop_visual_plan_generation(self.opt)
        elif self.opt.task == "vgt-u-plan": # image caption
            self.open_loop_textual_plan_generation(self.opt, self.outpath, self.summarize_example_data_list)