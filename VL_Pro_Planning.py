import sys
sys.path.append("submodules")
sys.path.append("submodules/stablediffusion")
import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
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

torch.set_grad_enabled(False)

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


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

# @hydra.main(version_base=None, config_path="configs", config_name="config")
def main(opt):
    seed_everything(opt.seed)

    # common setup
    resolution_config = "resolution_{}".format(opt.resolution)
    outpath = os.path.join(opt.outdir, "debug_output", resolution_config) if opt.debug else os.path.join(opt.outdir, "experiment_output", resolution_config)
    os.makedirs(outpath, exist_ok=True)
    data, task_start_idx_list, summarize_example_data_list = load_prompt(opt)
    task_config = opt.task + ".yaml"
    config = OmegaConf.load(f"{os.path.join(opt.config_root, resolution_config, task_config)}")
    # Single Step Textual Plan Generation (LLM, GPT3)
    # result_list = []
    from LLM_Reasoning import LLM_Reasoning
    llm_reasoning_engine = LLM_Reasoning(opt)
    # task_path = opt.data_type + "/" + method_type + "/" + datetime.now().strftime("%Y_%m_%d") + "/" + "demo_{}_{}_inter{}_var{}_heldout{}_oloop{}_".format(opt.language_model_type, opt.model_type, opt.open_loop) + datetime.now().strftime("%H%M%S")
    # task_result_dir = os.path.join("../result", task_path)
    task_result_dir = os.path.join(outpath, opt.data_type, opt.task)
    if not os.path.isdir(task_result_dir): os.makedirs(task_result_dir)
    skip_count = 0
    with open(os.path.join(task_result_dir, "{}_task_result.txt".format(opt.language_model_type)), 'w') as resultfile:
        total_score_cal = total_score.copy()
        for data_example in summarize_example_data_list[1:]:
            
            total_score_cal, result_list = llm_reasoning_engine.language_planning(total_score_cal, data_example)

        # mean value
        ic(len(summarize_example_data_list), total_score_cal[opt.model_type].keys())
        for score_key in total_score_cal[opt.model_type].keys():
            total_score_cal[opt.model_type][score_key] /= (len(summarize_example_data_list)-skip_count)
        resultfile.writelines(result_list)
        json.dump(total_score_cal,resultfile)
        ic(skip_count, total_score_cal[opt.model_type])
    if config.mpp_model.task_config.plan_modality == "textual": return
    # Single Step Visual Plan Generation (text-to-image model, stable diffusion v2)
    model = load_model_from_config(config, f"{config.model.ckpt}")
    # model = hydra.utils.instantiate(cfg.model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

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

    sample_path = os.path.join(outpath, opt.data_type, opt.task) # os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    task_count = 0
    step_count = 0
    all_step_count = 0
    base_count = len(os.listdir(sample_path))
    # grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad(), \
        precision_scope("cuda"), \
        model.ema_scope():
            # all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
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
                        sample_path = os.path.join(outpath, opt.data_type, opt.task, "task_{}".format(task_count))
                        os.makedirs(sample_path, exist_ok=True)
                        if all_step_count in task_start_idx_list:
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

                    # all_samples.append(x_samples)

            # # additionally, save as grid
            # grid = torch.stack(all_samples, 0)
            # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            # grid = make_grid(grid, nrow=n_rows)

            # # to image
            # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            # grid = Image.fromarray(grid.astype(np.uint8))
            # grid = put_watermark(grid, wm_encoder)
            # grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
            # grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")

    # Temporal-extended multimodal procedural planning

if __name__ == "__main__":
    import datetime
    opt = parse_args()
    method_type=opt.model_type
    total_score = {}
    all_model_type_list = ['{}'.format(method_type)]
    for model_type_item in all_model_type_list:
        total_score[model_type_item] = {"sentence-bleu": 0, "bert-score-f": 0, "LCS": 0, "CLIP": 0}
    main(opt)
