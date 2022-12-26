import os
import sys
sys.path.append("submodules")
sys.path.append("submodules/stablediffusion")
sys.path.append("models")
import argparse, os
import torch
from omegaconf import OmegaConf
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
import json

from Multimodal_Procedural_Planning import MPP_Planner
from Baseline_Planning import Baseline_Planner

torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    # LLM argument
    parser.add_argument('--api_key', type=str, default="sk-rXwHNrNvXmaQv9n9OKe3NPCBvH7RGXhCmK3YQNRm", help='api key')
    parser.add_argument('--language_model_type', choices=['gpt-j-6B', 't5-11b', 'gpt2-1.5B', 'gpt2', 'gpt2-xl', 'gpt3', 'gpt_neo', 't5', 'bart', 'bert', 'roberta'], default="gpt3", help='choices')
    parser.add_argument('--model_type', choices=['concept_knowledge', 'task_only_base', 'base', 'base_tune', 'standard_prompt', 'soft_prompt_tuning', 'chain_of_thought', 'chain_of_cause', 'cmle_ipm', 'cmle_epm', 'irm', 'vae_r', 'rwSAM', 'counterfactual_prompt'], default="task_only_base", help='choices')
    parser.add_argument('--variant_type', choices=['wo_symbolic', 'wo_causality', 'full', 'wo_firsttranslation'], default='full', help='choices')
    parser.add_argument('--task_num', type=int, default=5)
    
    parser.add_argument('--n_tokens', type=int, default=20, help='n_tokens')
    parser.add_argument('--init_from_vocab', action='store_true', help='demo')
    parser.add_argument('--open_loop', action='store_true', help='open_loop')
    parser.add_argument('--triplet_similarity_threshold', type=float, default=0.4)
    parser.add_argument('--limit_num', type=int, default=50)
    
    parser.add_argument('--max_tokens', type=int, default=30)

    # stable diffusion argument
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of an astronaut riding a triceratops",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="/share/edc/home/yujielu/MPP_data"
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
        default="from file"
    )
    parser.add_argument(
        "--file_type",
        type=str,
        help="file type",
        default="json"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        help="data type",
        default="wikihow"
    )
    parser.add_argument(
        "--task",
        type=str,
        help="task",
        default="m-plan"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="resolution",
        default=512
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1, # 3
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1, # 3
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config_root",
        type=str,
        default="configs",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat each prompt in file this often",
    )
    parser.add_argument('--save_task_grid', action='store_true', help='demo')
    parser.add_argument('--debug', action='store_true', help='demo')
    opt = parser.parse_args()
    return opt


def main(opt):
    seed_everything(opt.seed)

    # common setup
    resolution_config = "resolution_{}".format(opt.resolution)
    outpath = os.path.join(opt.outdir, "debug_output", resolution_config) if opt.debug else os.path.join(opt.outdir, "experiment_output", resolution_config)
    os.makedirs(outpath, exist_ok=True)
    data, task_start_idx_list, summarize_example_data_list = load_prompt(opt)
    task_config = opt.task + ".yaml"
    config = OmegaConf.load(f"{os.path.join(opt.config_root, resolution_config, task_config)}")
    baseline_planner = Baseline_Planner(opt, config, outpath, data, task_start_idx_list)
    baseline_planner.start_planning()


    mpp_planner = MPP_Planner(opt, config, outpath, data, task_start_idx_list)
    mpp_planner.start_planning()

if __name__ == "__main__":
    import datetime
    opt = parse_args()
    main(opt)
