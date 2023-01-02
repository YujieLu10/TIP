cd submodules
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
pip install OFA/transformers/
git clone https://huggingface.co/OFA-Sys/OFA-tiny 

pip install git+https://github.com/openai/CLIP.git

python -m spacy download en_core_web_md
conda install -n mpp ipykernel --update-deps --force-reinstall