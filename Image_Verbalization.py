import sys
sys.path.append("submodules")
sys.path.append("submodules/OFA")
PLAN_ROOT = "/share/edc/home/yujielu/MPP_data/test_config/wikihow/u-plan/"

import os
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator
from icecream import ic
import glob
from tqdm import tqdm
import torch

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ckpt_dir='submodules/OFA-tiny'
tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

txt = " what does the image describe?"
inputs = tokenizer([txt], return_tensors="pt").input_ids

task_num = len(os.listdir(PLAN_ROOT))

model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
# model = model.to(device)
        
for task_idx in tqdm(range(task_num)):
    sample_path = os.path.join(PLAN_ROOT, f"task_{task_idx}")
    if not os.path.exists(sample_path): continue
    # step_num = len(os.listdir(sample_path))
    step_num = len(glob.glob1(sample_path,"step_*.png"))
    for step_idx in range(step_num):            
        img = Image.open(os.path.join(sample_path, f"step_{step_idx}.png"))
        patch_img = patch_resize_transform(img).unsqueeze(0)
        generator = sequence_generator.SequenceGenerator(
            tokenizer=tokenizer,
            beam_size=10,
            max_len_b=160,
            min_len=20,
            no_repeat_ngram_size=3,
        )

        data = {}
        data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}
        # using the generator of fairseq version
        gen_output = generator.generate([model], data)
        gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

        caption = tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()

        with open(os.path.join(sample_path, f"step_{step_idx}_caption.txt"), 'w') as f:
            f.write(f"{caption}")
        # # using the generator of huggingface version
        # model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
        # gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
        # ic(tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip())
