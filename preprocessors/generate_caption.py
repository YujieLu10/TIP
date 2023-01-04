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

from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
# import cog
import sys
sys.path.append("submodules")
sys.path.append("submodules/BLIP")
from models.blip import blip_decoder


class Predictor(object):
    def __init__(self) -> None:
        self.device = "cuda:0"

        self.models = {
            'image_captioning': blip_decoder(pretrained='/share/edc/home/yujielu/MPP_data/model_base_caption_capfilt_large.pth',
                                             image_size=384, vit='base'),
        }

    def predict(self, image, task):
        im = load_image(image, image_size=384, device=self.device)
        model = self.models[task]
        model.eval()
        model = model.to(self.device)

        if task == 'image_captioning':
            with torch.no_grad():
                caption = model.generate(im, sample=False, num_beams=3, max_length=20, min_length=5)
                # return 'Caption: ' + caption[0]
                return caption[0]

def load_image(image, image_size, device):
    raw_image = Image.open(image).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def parse_args():
    parser = argparse.ArgumentParser()
    # LLM argument
    parser.add_argument('--image_root', type=str, default="/share/edc/home/yujielu/MPP_data", help='image root')
    parser.add_argument('--source', type=str, default="groundtruth_input", help='source')
    parser.add_argument('--eval_task', type=str, default="all", help='eval task')
    opt = parser.parse_args()
    return opt    
    
def generate_caption(image_path, use_blip=False, bridge_list=["_bridge", ""]):
    task_num = len(os.listdir(image_path))
    if use_blip:
        predictor = Predictor()
    else:
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
        tokenizer = OFATokenizer.from_pretrained(ckpt_dir, use_auth_token="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")

        txt = " what does the image describe?"
        inputs = tokenizer([txt], return_tensors="pt").input_ids

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
        for postfix in bridge_list:
            step_num = len(glob.glob1(sample_path,"step_[0-9]*_bridge.png" if "wikihow" in image_path else "step_[0-9]*_bridge.png")) or len(glob.glob1(sample_path,"step_[0-9]*.png" if "wikihow" in image_path else "step_[0-9]*.png"))
            
            for step_idx in range(1, step_num+1):
                img_path = os.path.join(sample_path, f"step_{step_idx}{postfix}.png" if "wikihow" in image_path else f"step_{step_idx}{postfix}.png")
                if use_blip:
                    caption = predictor.predict(image=img_path, task="image_captioning")
                else:
                    img = Image.open(img_path)
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
        elif opt.source == "template_check":    
            exp_path = os.path.join(opt.image_root, "template_eval_output", data_type, "tgt-u-plan")
            if opt.eval_task == "all":
                for task_name in os.listdir(exp_path):
                    if "t2i" in task_name or task_name == "all_template_metric.csv": continue
                    task_path = os.path.join(exp_path, task_name)
                    ic(task_path)
                    generate_caption(task_path, bridge_list=[""])
        else:
            exp_path = os.path.join(opt.image_root, opt.source, "resolution_512", data_type)
            if opt.eval_task == "all":
                # ic(os.listdir(exp_path))
                for task_name in os.listdir(exp_path):
                    if task_name in ["vgt-u-plan", "vgt-u-plan-blip"]: continue # temporarily skip not ready u-plan result
                    task_path = os.path.join(exp_path, task_name)
                generate_caption(task_path)
            else:
                task_path = os.path.join(exp_path, opt.eval_task)
                ic(task_path)
                generate_caption(task_path)
    
