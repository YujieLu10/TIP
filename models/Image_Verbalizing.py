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

from LLM_Reasoning import *

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


class Image_Verbalizing(object):
    def __init__(self, opt, outpath) -> None:
        super().__init__()
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        self.outpath = outpath # "/share/edc/home/yujielu/MPP_data/test_config/wikihow/u-plan/"
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.opt = opt
        if self.opt.caption_model_type == "blip":
            self.predictor = Predictor()
        else:
            ckpt_dir='submodules/OFA-base'
            self.tokenizer = OFATokenizer.from_pretrained(ckpt_dir, use_auth_token="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")

            txt = " what does the image describe?"
            self.inputs = self.tokenizer([txt], return_tensors="pt").input_ids
            
            self.model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
            self.model = self.model.to(self.device)
            
            self.generator = sequence_generator.SequenceGenerator(
                tokenizer=self.tokenizer,
                beam_size=10,
                max_len_b=160,
                min_len=20,
                no_repeat_ngram_size=3,
            ).to(self.device)

    def get_caption(self, img_path):
        if self.opt.caption_model_type == "blip":
            caption = self.get_caption_blip(img_path)
        else:
            caption = self.get_caption_ofa(img_path)
        return caption
    
    def get_caption_ofa(self, img_path):
        img = Image.open(img_path)
        patch_img = self.patch_resize_transform(img).unsqueeze(0)

        data = {}
        data["net_input"] = {"input_ids": self.inputs.to(self.device), 'patch_images': patch_img.to(self.device), 'patch_masks':torch.tensor([True]).to(self.device)}
        # using the generator of fairseq version
        gen_output = self.generator.generate([self.model], data)
        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

        caption = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()    
        return caption
        
    def get_caption_blip(self, img_path):
        caption = self.predictor.predict(image=img_path, task="image_captioning")
        return caption

    def write_verbalization(self, sample_path, postfix=""):
        # step_num = len(os.listdir(sample_path))
        step_num = len(glob.glob1(sample_path,"step_[0-9]*_bridget2i-0.png")) or len(glob.glob1(sample_path,"step_[0-9]*_bridge.png")) or len(glob.glob1(sample_path,"step_[0-9]*.png"))
        for step_idx in range(1, step_num+1):   
            img_path = os.path.join(sample_path, f"step_{step_idx}{postfix}.png")         
            caption = self.get_caption(img_path)
            with open(os.path.join(sample_path, f"step_{step_idx}{postfix}_caption.txt"), 'w') as f:
                f.write(f"{caption}")
            # # using the generator of huggingface version
            # model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
            # gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
            # ic(tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip())

    def start_verbalizing(self, single_caption=True, img_path=None):
        if single_caption:
            caption = self.get_caption(img_path)
        else:                
            task_num = len(os.listdir(self.outpath))
            for task_idx in tqdm(range(task_num)):
                sample_path = os.path.join(self.outpath, f"task_{task_idx}")
                ic(sample_path)
                if not os.path.exists(sample_path): continue
                if (self.opt.task in ["m-plan"] and not self.opt.t2i_template_check and not self.opt.i2t_template_check):
                    for postfix in ["", "_bridge"]:
                        self.write_verbalization(sample_path, postfix)
                elif self.opt.t2i_template_check:
                    postfix_list = [""]
                    for bridgename in t2i_template_dict: postfix_list.append(f"_bridge{bridgename}")
                    for postfix in postfix_list:
                        self.write_verbalization(sample_path, postfix)
                elif self.opt.i2t_template_check:
                    # postfix_list = [""]
                    # for postfix in postfix_list:
                    #     self.write_verbalization(sample_path, postfix)
                    pass