import os
from sentence_transformers import SentenceTransformer, util as st_utils
import clip
import torch
import evaluate
import spacy
# from cider import Cider
from tqdm import tqdm
import glob
from icecream import ic
import json
from PIL import Image
import argparse
import csv

class Template_Checker(object):
    def __init__(self, opt, task_name) -> None:
        self.opt = opt
        self.task_name = task_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def get_similarity_tgt(self, total_score_cal, tplan, vplan, tgt, step_num):
        with torch.no_grad():
            pre_features = self.clip_model.encode_text(tplan)
            gt_features = self.clip_model.encode_text(tgt)
            pre_features /= pre_features.norm(dim=-1, keepdim=True)
            gt_features /= gt_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * pre_features @ gt_features.T)#.softmax(dim=-1)
            # ic(similarity.cpu().item())
            # # TODO: debug
            # ic(similarity.detach().cpu().numpy())
            total_score_cal["tplan-tgt-clip-score"] += (similarity.cpu().item() / step_num)
            
            pre_features = self.clip_model.encode_image(vplan)
            gt_features = self.clip_model.encode_text(tgt)
            pre_features /= pre_features.norm(dim=-1, keepdim=True)
            gt_features /= gt_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * pre_features @ gt_features.T)#.softmax(dim=-1)
            # ic(similarity.cpu().item())
            # # TODO: debug
            # ic(similarity.detach().cpu().numpy())
            total_score_cal["vplan-tgt-clip-score"] += (similarity.cpu().item() / step_num)
        return total_score_cal

    def get_similarity_vgt(self, total_score_cal, tplan, vplan, vgt, step_num):
        with torch.no_grad():
            pre_features = self.clip_model.encode_image(vplan)
            gt_features = self.clip_model.encode_image(vgt)
            pre_features /= pre_features.norm(dim=-1, keepdim=True)
            gt_features /= gt_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * pre_features @ gt_features.T)#.softmax(dim=-1)
            # ic(similarity.cpu().item())
            # ic(similarity)
            # ic(similarity.cpu().item())
            # # TODO: debug
            # ic(similarity.detach().cpu().numpy())
            total_score_cal["vplan-vgt-clip-score"] += (similarity.cpu().item() / step_num)
            
            pre_features = self.clip_model.encode_text(tplan)
            gt_features = self.clip_model.encode_image(vgt)
            pre_features /= pre_features.norm(dim=-1, keepdim=True)
            gt_features /= gt_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * pre_features @ gt_features.T)#.softmax(dim=-1)
            # ic(similarity.cpu().item())
            # ic(similarity)
            # ic(similarity.cpu().item())
            # # TODO: debug
            # ic(similarity.detach().cpu().numpy())
            total_score_cal["tplan-vgt-clip-score"] += (similarity.cpu().item() / step_num)
        return total_score_cal
    
    def get_plan_clip_score(self, total_score_cal, tplan, vplan, step_num):
        with torch.no_grad():
            pre_features = self.clip_model.encode_text(tplan)
            gt_features = self.clip_model.encode_image(vplan)
            pre_features /= pre_features.norm(dim=-1, keepdim=True)
            gt_features /= gt_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * pre_features @ gt_features.T)#.softmax(dim=-1)
            # ic(similarity.cpu().item())
            
            total_score_cal["tplan-vplan-clip-score"] += (similarity.cpu().item() / step_num)
        return total_score_cal

    def calculate_sample_step_score(self, predict_sample_path, gt_sample_path, step_idx, total_score_cal, tep, tgt, step_num, data_type, module, template):
        # Table 3, consider to merge with Table 2
        # ic(tep, tgt)
        text_input = torch.cat([clip.tokenize(tep)]).to(self.device)
        if module == "t2i-bridge":
            image = Image.open(os.path.join(predict_sample_path, f"step_{step_idx}_bridge{template}.png"))
        else:
            # ic(os.path.join(predict_sample_path, f"step_{step_idx}.png"))
            image = Image.open(os.path.join(predict_sample_path, f"step_{step_idx}.png"))
        image_input = self.preprocess(image).unsqueeze(0).to(self.device).to(self.device)
        
        tgt = torch.cat([clip.tokenize(tgt)]).to(self.device)
        gt_image = Image.open(os.path.join(gt_sample_path, f"step_{step_idx}.jpg" if data_type == "recipeqa" else f"step_{step_idx}.png"))
        vgt = self.preprocess(gt_image).unsqueeze(0).to(self.device).to(self.device)
        
        # if module == "t2i-bridge":
        #     total_score_cal = self.get_similarity_vgt(total_score_cal, text_input, image_input, vgt, step_num)
        # else:
        #     # ic(text_input, tgt)
        #     total_score_cal = self.get_similarity_tgt(total_score_cal, text_input, image_input, tgt, step_num)
        total_score_cal = self.get_plan_clip_score(total_score_cal, text_input, image_input, step_num)
        
        return total_score_cal
    
    
    def get_content(self, path, step_idx, postfix):
        content = ""
        with open(os.path.join(path, f"step_{step_idx}{postfix}.txt"), 'r') as f:
            current_text = f.readline()
            content += f"{current_text} "
        return content
        

    def eval_template(self, taskpath, template, data_type, module):        
        total_score_cal = {"vplan-vgt-clip-score": 0, "tplan-tgt-clip-score": 0, "tplan-vgt-clip-score": 0, "vplan-tgt-clip-score": 0, "tplan-vplan-clip-score": 0}
        predict_data_path = taskpath
        # gt_path = os.path.join(opt.image_root, opt.source, data_type)
        gt_data_path = os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input/{}".format(data_type))
        task_num = self.opt.task_num
        for task_idx in tqdm(range(task_num)):
            predict_sample_path = os.path.join(predict_data_path, f"task_{task_idx}")
            gt_sample_path = os.path.join(gt_data_path, f"task_{task_idx}")
            # ic(predict_sample_path)
            if not os.path.exists(predict_sample_path): continue
            # skip task evaluate
            step_num = len(glob.glob1(gt_sample_path,f"step_[0-9]*_bridge{template}_caption.txt")) or len(glob.glob1(gt_sample_path,"step_[0-9]*_caption.txt")) or len(glob.glob1(gt_sample_path,"step_[0-9]*.txt"))
            for step_idx in range(1, step_num+1):
                # _bridge_caption => _bridge_tplan after generate bridge tplan
                tep = ""
                if module == "i2t-bridge":
                    tep = self.get_content(predict_sample_path, step_idx, f"_bridge{template}_tplan")
                    # tep = f"Step {step_idx}: {tep}"
                else:
                    tep = self.get_content(predict_sample_path, step_idx, "")
                    # tep = f"Step {step_idx}: {tep}"
                tgt = self.get_content(gt_sample_path, step_idx, "")
                total_score_cal = self.calculate_sample_step_score(predict_sample_path, gt_sample_path, step_idx, total_score_cal, tep, tgt, step_num, data_type, module, template)
        for score_key in total_score_cal.keys():
            total_score_cal[score_key] /= task_num
            total_score_cal[score_key] = round(total_score_cal[score_key], 4)
        csv_line = [self.task_name] + [module] + [template] + [total_score_cal["vplan-vgt-clip-score"]] + [total_score_cal["tplan-tgt-clip-score"]] + [total_score_cal["vplan-tgt-clip-score"]] + [total_score_cal["tplan-vgt-clip-score"]] + [total_score_cal["tplan-vplan-clip-score"]]
        return csv_line

def parse_args():
    parser = argparse.ArgumentParser()
    # LLM argument
    parser.add_argument('--image_root', type=str, default="/share/edc/home/yujielu/MPP_data", help='image root')
    parser.add_argument('--source', type=str, default="experiment_output", help='source')
    parser.add_argument('--task_num', type=int, default=5)
    opt = parser.parse_args()
    return opt    
    
if __name__ == "__main__":
    opt = parse_args()
    for data_type in ["wikihow", "recipeqa"]:
        exp_path = os.path.join(opt.image_root, "template_eval_output", data_type)
        os.makedirs(exp_path, exist_ok=True)
        with open(os.path.join(exp_path, "all_template_metric.csv"), 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            head_line = ['task_name', 'module', 'template', 'vplan-vgt-clip-score', 'tplan-tgt-clip-score', 'vplan-tgt-clip-score', 'tplan-vgt-clip-score', 'tplan-vplan-clip-score']
            writer.writerow(head_line)
            # for task_name in ["tgt-u-plan"]:
            for task_name in ["m-plan"]:
                template_checker = Template_Checker(opt, task_name)
                task_path = os.path.join(exp_path, task_name)
                for module in ["t2i-bridge", "i2t-bridge", ""]:
                    if not module == "":
                        template_list = [f"t2i-{i}" for i in range(6)] if module in ["t2i-bridge"] else [f"i2t-{i}" for i in range(6)]
                    else:
                        template_list = [""]
                    # template_list = ["test_template"]
                    for template in template_list:
                        metric_csv_line = template_checker.eval_template(task_path, template, data_type, module)
                        writer.writerow(metric_csv_line)
