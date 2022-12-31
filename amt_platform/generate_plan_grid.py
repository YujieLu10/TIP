import os
import torch
from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image
import cv2
import numpy as np
from imwatermark import WatermarkEncoder
from numpy import asarray
from torchvision import transforms
from tqdm import tqdm
import glob
import argparse
from icecream import ic

def parse_args():
    parser = argparse.ArgumentParser()
    # LLM argument
    parser.add_argument('--image_root', type=str, default="/share/edc/home/yujielu/MPP_data", help='image root')
    parser.add_argument('--source', type=str, default="groundtruth_input", help='source')
    parser.add_argument('--eval_task', type=str, default="all", help='eval task')
    opt = parser.parse_args()
    return opt    
    

def generate_plan_grid(image_path, bridge_list):
    task_num = len(os.listdir(image_path))
    for task_idx in tqdm(range(task_num)):
        all_samples = list()
        sample_path = os.path.join(image_path, f"task_{task_idx}")
        if not os.path.exists(sample_path): continue
        # step_num = len(os.listdir(sample_path))
        # jpg for groundtruth input
        for postfix in bridge_list:
            step_num = len(glob.glob1(sample_path,"step_[0-9]_bridge.png" if "wikihow" in image_path else "step_[0-9]_bridge.png")) or len(glob.glob1(sample_path,"step_[0-9].png" if "wikihow" in image_path else "step_[0-9].png"))
            for step_idx in range(1, step_num+1):            
                img = Image.open(os.path.join(sample_path, f"step_{step_idx}{postfix}.png" if "wikihow" in image_path else f"step_{step_idx}{postfix}.jpg")).resize((512, 512))
                # .thumbnail((400, 400))
                # img = Image.open("/share/edc/home/yujielu/MPP_data/experiment_output/resolution_512/wikihow/tgt-u-plan/task_0/step_1.png")
                convert_tensor = transforms.ToTensor()
                all_samples.append(convert_tensor(img))
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n c h w -> (n) c h w')
        # grid = rearrange(grid, 'n b h w c-> (n b) h w c')
        grid = make_grid(grid, nrow=8)
        # int((len(all_samples)-1)/3)+1

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))
        # grid.save(os.path.join(sample_path, f'task-grid-{task_count}.png'))
        grid.save(f"test_grid_{task_idx}.png")

if __name__ == "__main__":
    opt = parse_args()
    for data_type in ["wikihow", "recipeqa"]:
        bridge_list = [""]
        if opt.source == "groundtruth_input":
            image_path = os.path.join(opt.image_root, opt.source, data_type)
            generate_plan_grid(image_path, bridge_list)
        else:
            exp_path = os.path.join(opt.image_root, opt.source, "resolution_512", data_type)
            if opt.eval_task == "all":
                # ic(os.listdir(exp_path))
                for task_name in os.listdir(exp_path):
                    if task_name in ["tgt-u-plan", "u-plan", "c-plan"]:
                        bridge_list.append("_bridge")
                    if task_name == "vgt-u-plan": continue # temporarily skip not ready u-plan result
                    task_path = os.path.join(exp_path, task_name)
                generate_plan_grid(task_path, bridge_list)
            else:
                task_path = os.path.join(exp_path, opt.eval_task)
                ic(task_path)
                generate_plan_grid(task_path, bridge_list)
    
