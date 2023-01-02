import sys
sys.path.append("../submodules")
sys.path.append("../submodules/OFA")

import os
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator
from icecream import ic
import glob
from tqdm import tqdm
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # LLM argument
    parser.add_argument('--image_root', type=str, default="/share/edc/home/yujielu/MPP_data", help='image root')
    parser.add_argument('--source', type=str, default="groundtruth_input", help='source')
    parser.add_argument('--eval_task', type=str, default="all", help='eval task')
    opt = parser.parse_args()
    return opt    
    
def generate_caption(image_path):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    resolution = 256
    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # ckpt_dir='submodules/OFA-tiny'
    ckpt_dir='submodules/OFA-base'
    tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

    txt = " what does the image describe?"
    inputs = tokenizer([txt], return_tensors="pt").input_ids

    task_num = len(os.listdir(image_path))

    model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
    model = model.to(device)
    generator = sequence_generator.SequenceGenerator(
        tokenizer=tokenizer,
        beam_size=10,
        max_len_b=160,
        min_len=20,
        no_repeat_ngram_size=3,
    ).to(device)
    for task_idx in tqdm(range(task_num)):
        sample_path = os.path.join(image_path, f"task_{task_idx}")
        if not os.path.exists(sample_path): continue
        # step_num = len(os.listdir(sample_path))
        # jpg for groundtruth input
        for postfix in ["_bridge", ""]:
            step_num = len(glob.glob1(sample_path,"step_[0-9]*_bridge.png" if "wikihow" in image_path else "step_[0-9]*_bridge.png")) or len(glob.glob1(sample_path,"step_[0-9]*.png" if "wikihow" in image_path else "step_[0-9]*.png"))
            
            for step_idx in range(1, step_num+1):            
                img = Image.open(os.path.join(sample_path, f"step_{step_idx}{postfix}.png" if "wikihow" in image_path else f"step_{step_idx}{postfix}.png"))
                patch_img = patch_resize_transform(img).unsqueeze(0)

                data = {}
                data["net_input"] = {"input_ids": inputs.to(device), 'patch_images': patch_img.to(device), 'patch_masks':torch.tensor([True]).to(device)}
                # using the generator of fairseq version
                gen_output = generator.generate([model], data)
                gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

                caption = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

                with open(os.path.join(sample_path, f"step_{step_idx}{postfix}_caption.txt"), 'w') as f:
                    f.write(f"{caption}")
                # # using the generator of huggingface version
                # model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
                # gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
                # ic(tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip())

if __name__ == "__main__":
    opt = parse_args()
    for data_type in ["wikihow", "recipeqa"]:
        if opt.source == "groundtruth_input":
            image_path = os.path.join(opt.image_root, opt.source, data_type)
            generate_caption(image_path)
        else:
            exp_path = os.path.join(opt.image_root, opt.source, "resolution_512", data_type)
            if opt.eval_task == "all":
                # ic(os.listdir(exp_path))
                for task_name in os.listdir(exp_path):
                    if task_name == "vgt-u-plan": continue # temporarily skip not ready u-plan result
                    task_path = os.path.join(exp_path, task_name)
                generate_caption(task_path)
            else:
                task_path = os.path.join(exp_path, opt.eval_task)
                ic(task_path)
                generate_caption(task_path)
    
