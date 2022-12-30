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

class Image_Verbalizing(object):
    def __init__(self, args) -> None:
        super().__init__()
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution = 256
        self.outpath = args.outpath # "/share/edc/home/yujielu/MPP_data/test_config/wikihow/u-plan/"
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std)
        ])
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ckpt_dir='../submodules/OFA-tiny'
        tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

        txt = " what does the image describe?"
        self.inputs = tokenizer([txt], return_tensors="pt").input_ids
        
        self.model = OFAModel.from_pretrained(self.ckpt_dir, use_cache=False)
        # model = model.to(device)

    def get_caption(self, img_path):
        img = Image.open(img_path)
        patch_img = self.patch_resize_transform(img).unsqueeze(0)
        generator = sequence_generator.SequenceGenerator(
            tokenizer=self.tokenizer,
            beam_size=10,
            max_len_b=160,
            min_len=20,
            no_repeat_ngram_size=3,
        )

        data = {}
        data["net_input"] = {"input_ids": self.inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}
        # using the generator of fairseq version
        gen_output = generator.generate([self.model], data)
        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

        caption = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
        return caption


    def start_verbalizing(self, single_caption=True, img_path=None):
        if single_caption:
            caption = self.get_caption(img_path)
        else:                
            task_num = len(os.listdir(self.outpath))
            for task_idx in tqdm(range(task_num)):
                sample_path = os.path.join(self.outpath, f"task_{task_idx}")
                if not os.path.exists(sample_path): continue
                # step_num = len(os.listdir(sample_path))
                step_num = len(glob.glob1(sample_path,"step_*.png"))
                for step_idx in range(1, step_num+1):   
                    img_path = os.path.join(sample_path, f"step_{step_idx}.png")         
                    caption = self.get_caption(img_path)
                    with open(os.path.join(sample_path, f"step_{step_idx}_caption.txt"), 'w') as f:
                        f.write(f"{caption}")
                    # # using the generator of huggingface version
                    # model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
                    # gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
                    # ic(tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip())