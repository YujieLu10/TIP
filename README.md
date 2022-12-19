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
