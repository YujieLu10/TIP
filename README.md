# MPP
Multimodal-Procedural-Planning

## Installation

```
git clone --recursive git@github.com:YujieLu10/MPP.git
cd MPP
conda create -n mpp
conda activate mpp
<!-- conda install pytorch==1.12.1 torchvision==0.13.1 -c pytorch -->
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install transformers==4.19.2 diffusers invisible-watermark
pip install -e .
pip install -r requirements.tx
```

## Script

### WikiHow
- m-plan: multimodal procedural planning 

- u-plan: unimodal procedural planning that seperately plan in textual and visual space

- t(v)gt-v(t)-plan: visual procedural planning with ground truth textual procedural plans, aka. generating visual plans directly using ground truth textual plan (textual procedural planning with ground truth visual procedural plans, aka. generating textual plans directly using ground truth visual plan)

- t(v)gt-m-plan: more like text to image generation with temporal dimension (more like image captioning with temporal dimension)

```
# with 512x512 resolution
python scripts/txt2img.py --ckpt /share/edc/home/yujielu/MPP_data/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 512 --W 512 --outdir /share/edc/home/yujielu/MPP_data/wikihow/wikicausal_demo/tgt-v-plan --from_file /local/home/yujielu/project/MPP/GoalAgent/data/wikihow/wikicausal_demo.csv

# with 768x768 resolution
python scripts/txt2img.py --ckpt /share/edc/home/yujielu/MPP_data/v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --outdir /share/edc/home/yujielu/MPP_data/wikihow/wikicausal_demo/tgt-v-plan_768 --from_file /local/home/yujielu/project/MPP/GoalAgent/data/wikihow/wikicausal_demo.csv

# seperate task
python scripts/txt2img.py --ckpt /share/edc/home/yujielu/MPP_data/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 512 --W 512 --outdir /share/edc/home/yujielu/MPP_data/wikihow/wikicausal_demo/tgt-v-plan --from_file /local/home/yujielu/project/MPP/GoalAgent/data/wikihow/wikicausal_demo.csv
python scripts/txt2img.py --ckpt /share/edc/home/yujielu/MPP_data/v2-1_768-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768 --outdir /share/edc/home/yujielu/MPP_data/wikihow/wikicausal_demo/tgt-v-plan_768 --from_file /local/home/yujielu/project/MPP/GoalAgent/data/wikihow/wikicausal_demo.csv
```

### RecipeQA: A Challenge Dataset for Multimodal Comprehension of Cooking Recipes
https://aclanthology.org/D18-1166.pdf

### WinoGround (maybe a little weird to use this)

```
CUDA_VISIBLE_DEVICES=6 python scripts/txt2img.py --ckpt /share/edc/home/yujielu/MPP_data/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 512 --W 512 --outdir /share/edc/home/yujielu/MPP_data/winoground/txt2img-samples --from_file /local/home/yujielu/project/MPP/GoalAgent/data/wikihow/wikicausal_demo.csv

CUDA_VISIBLE_DEVICES=7 python scripts/txt2img.py --ckpt /share/edc/home/yujielu/MPP_data/v2-1_512-ema-pruned.ckpt --config configs/stable-diffusion/v2-inference.yaml --H 512 --W 512 --outdir /share/edc/home/yujielu/MPP_data/diffusiondb/txt2img-samples --from_file /local/home/yujielu/project/MPP/GoalAgent/data/wikihow/wikicausal_demo.csv
```

