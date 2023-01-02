import os
import torch
from einops import rearrange
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from imwatermark import WatermarkEncoder
from numpy import asarray
from torchvision import transforms
from tqdm import tqdm
import glob
import argparse
from icecream import ic
import math

def parse_args():
    parser = argparse.ArgumentParser()
    # LLM argument
    parser.add_argument('--image_root', type=str, default="/share/edc/home/yujielu/MPP_data", help='image root')
    parser.add_argument('--source', type=str, default="groundtruth_input", help='source')
    parser.add_argument('--eval_task', type=str, default="all", help='eval task')
    opt = parser.parse_args()
    return opt    
    

def generate_plan_grid(image_path, bridge_list, output_plan_grid_path, exp_name, data_type):
    task_num = len(os.listdir(image_path))
    # output_plan_grid_path = os.path.join(image_path, "all_plan_grid")    
    os.makedirs(output_plan_grid_path, exist_ok=True)
    for task_idx in tqdm(range(task_num)):
        sample_path = os.path.join(image_path, f"task_{task_idx}")
        if not os.path.exists(sample_path): continue
        with open(os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input", data_type, f"task_{task_idx}", "task.txt"), "r") as ftask:
            taskname = ftask.readline()
        # step_num = len(os.listdir(sample_path))
        for postfix in bridge_list:
            all_samples = list()
            if exp_name in ["vgt-u-plan", "tgt-u-plan", "tgt-u-plan-dalle"]:
                glob_path = os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input", data_type, f"task_{task_idx}")
                step_num = len(glob.glob1(glob_path,"step_[0-9]*_caption.txt")) or len(glob.glob1(glob_path,"step_[0-9]*.txt"))
            else:
                # jpg for groundtruth input
                step_num = len(glob.glob1(sample_path,"step_[0-9]*_bridge.png" if "wikihow" in image_path else "step_[0-9]*_bridge.png")) or len(glob.glob1(sample_path,"step_[0-9]*.png" if "wikihow" in image_path else "step_[0-9]*.png"))
            for step_idx in range(1, step_num+1):
                if exp_name in ["vgt-u-plan"]:
                    img = Image.open(os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input", data_type, f"task_{task_idx}", f"step_{step_idx}{postfix}.png" if "wikihow" in image_path else f"step_{step_idx}{postfix}.jpg")).resize((512, 512))
                else:
                    img = Image.open(os.path.join(sample_path, f"step_{step_idx}{postfix}.png" if "wikihow" in image_path else f"step_{step_idx}{postfix}.png")).resize((512, 512))
                # .thumbnail((400, 400))
                # img = Image.open("/share/edc/home/yujielu/MPP_data/experiment_output/resolution_512/wikihow/tgt-u-plan/task_0/step_1.png")
                convert_tensor = transforms.ToTensor()
                # all_samples.append(convert_tensor(img))
                
                # step text and image
                image = Image.new("RGB", (512, 120), "white")
                font = ImageFont.truetype("/local/home/yujielu/project/MPP/amt_platform/arial.ttf", 18)
                # font = ImageFont.load_default()
                draw = ImageDraw.Draw(image)
                # position = (10, 10)
                if exp_name in ["vgt-u-plan"]:
                    text_path = os.path.join(sample_path, f"step_{step_idx}{postfix}_caption.txt")
                elif exp_name in ["tgt-u-plan", "tgt-u-plan-dalle"]:
                    text_path = os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input", data_type, f"task_{task_idx}", f"step_{step_idx}.txt")
                else:
                    # TODO: bridge text revision
                    # text_path = os.path.join(sample_path, f"step_{step_idx}{postfix}.txt")
                    text_path = os.path.join(sample_path, f"step_{step_idx}{postfix}.txt")
                with open(text_path, 'r') as f:
                    text = f.readline()
                    if not "Step" in text:
                        text = f"Step {step_idx}: {text}"
                import textwrap
                lines = textwrap.wrap(text, width=60)
                y_text = 10
                w = 512
                for line in lines:
                    width, height = font.getsize(line)
                    draw.text(((w - width) / 2, y_text), line, font=font, fill="black")
                    y_text += height
                # draw.text(position, text, fill="black")
                # image.save(os.path.join(output_plan_grid_path, f"step_grid_{step_idx}.png"))
                new_im = Image.new('RGB', (512, 512+120))
                new_im.paste(image, (0,0))
                new_im.paste(img, (0,120))
                # new_im.save(os.path.join(output_plan_grid_path, f"step_grid_{step_idx}.png"))
                all_samples.append(convert_tensor(new_im))
            
            # SAVE TASK GRID
            grid = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n c h w -> (n) c h w')
            # grid = rearrange(grid, 'n b h w c-> (n b) h w c')
            grid = make_grid(grid, nrow=8)
            # int((len(all_samples)-1)/3)+1

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid = Image.fromarray(grid.astype(np.uint8))
            # add task name            
            column = min(step_num, 8)
            row = math.ceil(step_num/8)
            grid_task = Image.new("RGB", (512 * column, 60+(512+120)*row), "white")
            taskname_img = Image.new("RGB", (512 * column, 60), "white")
            font = ImageFont.truetype("/local/home/yujielu/project/MPP/amt_platform/arial.ttf", 22)
            draw = ImageDraw.Draw(taskname_img)
            position = (10, 10)
            draw.text(position, taskname, font=font, fill="black")
                
            grid_task.paste(taskname_img, (0,0))
            grid_task.paste(grid, (0,60))
            grid_task.save(os.path.join(output_plan_grid_path, f"plan_grid_task_{task_idx}{postfix}.png"))
            # grid.save(os.path.join(output_plan_grid_path, f"plan_grid_task_{task_idx}{postfix}.png"))

def get_bridge_list(exp_name):
    if exp_name in ["tgt-u-plan", "tgt-u-plan-dalle", "u-plan", "c-plan"]:
        bridge_list = ["_bridge", ""]
    else:
        bridge_list = [""]
    return bridge_list

if __name__ == "__main__":
    opt = parse_args()
    for data_type in ["wikihow", "recipeqa"]:
        if opt.source == "groundtruth_input":
            image_path = os.path.join(opt.image_root, opt.source, data_type)
            output_plan_grid_path = os.path.join(opt.image_root, "all_plan_grid", opt.source, data_type)
            bridge_list = get_bridge_list(opt.eval_task)
            generate_plan_grid(task_path, bridge_list, output_plan_grid_path, exp_name, data_type)
        else:
            exp_path = os.path.join(opt.image_root, opt.source, "resolution_512", data_type)
            if opt.eval_task == "all":
                for exp_name in os.listdir(exp_path):
                    if exp_name == "all_metric.csv": continue
                    bridge_list = get_bridge_list(exp_name)
                    task_path = os.path.join(exp_path, exp_name)
                    # default: resolution 512
                    output_plan_grid_path = os.path.join(opt.image_root, "all_plan_grid", opt.source, data_type, exp_name)
                    generate_plan_grid(task_path, bridge_list, output_plan_grid_path, exp_name, data_type)
            else:
                task_path = os.path.join(exp_path, opt.eval_task)
                bridge_list = get_bridge_list(opt.eval_task)
                exp_name = opt.eval_task
                output_plan_grid_path = os.path.join(opt.image_root, "all_plan_grid", opt.source, data_type, exp_name)
                generate_plan_grid(task_path, bridge_list, output_plan_grid_path, exp_name, data_type)
    
