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

class Automatic_Evaluator(object):
    def __init__(self, opt, task_name) -> None:
        self.opt = opt
        self.task_name = task_name
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.meteor = evaluate.load('meteor')
        # self.cider = Cider()
        self.nlp_encoder = spacy.load('en_core_web_md')
        
    def lcs(self, X, Y, m, n):
        if m == 0 or n == 0:
            return 0
        elif X[m-1] == Y[n-1]:
            return 1 + self.lcs(X, Y, m-1, n-1)
        else:
            return max(self.lcs(X, Y, m, n-1), self.lcs(X, Y, m-1, n))
   
    def similariry_score(self, str1, str2):
        #Compute embedding for both lists
        embedding_1= self.model.encode(str1, convert_to_tensor=True)
        embedding_2 = self.model.encode(str2, convert_to_tensor=True)
        score = st_utils.pytorch_cos_sim(embedding_1, embedding_2).item()
        return score

    def calculate_sample_step_score(self, predict_data_path, task_idx, step_idx, total_score_cal, tep, ctep, use_bridge, step_num):
        # Table 3, consider to merge with Table 2
        text_input = torch.cat([clip.tokenize(tep)]).to(self.device)
        image = Image.open(os.path.join(predict_data_path, f"task_{task_idx}", f"step_{step_idx}{use_bridge}.jpg" if (self.task_name in ["vgt-u-plan", "vgt-u-plan-blip"] and self.opt.data_type == "recipeqa") else f"step_{step_idx}{use_bridge}.png"))
        image_input = self.preprocess(image).unsqueeze(0).to(self.device).to(self.device)
        # total_score_cal["vplan-t-clip-score"] += 0
        # total_score_cal["tplan-v-clip-score"] += 0
        # total_score_cal["vplan-v-clip-score"] += 0
        # total_score_cal["tplan-t-clip-score"] += 0

        with torch.no_grad():
            pre_features = self.clip_model.encode_text(text_input)
            vplan_features = self.clip_model.encode_image(image_input)
            pre_features /= pre_features.norm(dim=-1, keepdim=True)
            vplan_features /= vplan_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * pre_features @ vplan_features.T)#.softmax(dim=-1)
            # ic(similarity.cpu().item())
            # # TODO: debug
            # ic(similarity.detach().cpu().numpy())
            total_score_cal["vplan-tplan-clip-score"] += similarity.cpu().item() / step_num
        
        total_score_cal["vplancap-tplan-sent-bert"] += 0 / step_num
        total_score_cal["vplancap-tplan-bert-f1"] += 0 / step_num
        # total_score_cal["caption-vcap-sent-bert"] += 0
        # total_score_cal["caption-vcap-bert-f1"] += 0
        # total_score_cal["tplan-t-cider"] += self.cider.compute_score(task_eval_groundtruth, task_eval_predict)
        # total_score_cal["caption-t-cider"] += 0
        # total_score_cal["caption-vcap-cider"] += 0
        return total_score_cal

    def calculate_sample_score(self, total_score_cal, task_eval_groundtruth, task_eval_predict, visual_task_eval_groundtruth, visual_task_eval_predict, caption_task_eval_groundtruth, caption_task_eval_predict):
        import nltk
        task_eval_groundtruth = task_eval_groundtruth.replace('.', ' ')
        task_eval_predict = task_eval_predict.replace('.', ' ')
        # ic(task_eval_predict, task_eval_groundtruth)
        total_score_cal["sentence-bleu"] += nltk.translate.bleu_score.sentence_bleu([task_eval_groundtruth.split()], task_eval_predict.split())
        total_score_cal["wmd"] += self.nlp_encoder(task_eval_groundtruth).similarity(self.nlp_encoder(task_eval_predict))
        from rouge import Rouge
        rouge = Rouge()
        scores = rouge.get_scores(task_eval_predict, task_eval_groundtruth)
        total_score_cal["rouge-1-f"] += scores[0]["rouge-1"]["f"]
        total_score_cal["rouge-1-p"] += scores[0]["rouge-1"]["p"]
        total_score_cal["rouge-1-r"] += scores[0]["rouge-1"]["r"]
        from datasets import load_metric
        bertscore = load_metric("bertscore")
        try:
            bert_results = bertscore.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth], model_type="distilbert-base-uncased")
            total_score_cal["bert-score-f"] += bert_results["f1"][0]
            total_score_cal["bert-score-p"] += bert_results["precision"][0]
            total_score_cal["bert-score-r"] += bert_results["recall"][0]
        except:
            ic()
        total_score_cal["meteor"] += self.meteor.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth])["meteor"]
        
        # Table 2
        total_score_cal["sentence-bert-score"] += self.similariry_score(task_eval_predict, task_eval_groundtruth)
        
        total_score_cal["caption-t-bleu"] += 0
        total_score_cal["caption-vcap-bleu"] += 0
        # total_score_cal["lcs"] += self.lcs(task_eval_predict, task_eval_groundtruth, len(task_eval_predict), len(task_eval_groundtruth))
        # total_score_cal["caption-t-lcs"] += 0
        # total_score_cal["caption-vcap-lcs"] += 0
        
        total_score_cal["gpt3-plan-accuracy"] += 0
        total_score_cal["caption-gpt3-plan-accuracy"] += 0
        return total_score_cal
    
    
    def get_content(self, path, step_idx, postfix):
        content = ""
        with open(os.path.join(path, f"step_{step_idx}{postfix}.txt"), 'r') as f:
            current_text = f.readline()
            content += f"{current_text} "
        return content
        
    def calculate_total_score(self, total_score_cal, task_eval_groundtruth=None, task_eval_predict=None, visual_task_eval_groundtruth=None, visual_task_eval_predict=None, caption_task_eval_groundtruth=None, caption_task_eval_predict=None, from_task_path=""):
        if len(from_task_path):
            task_num = len(os.listdir(from_task_path))
            gt_data_path = os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input/{}".format(self.opt.data_type))
            for task_idx in tqdm(range(task_num)):
                sample_path = os.path.join(from_task_path, f"task_{task_idx}")
                # TODO: incorporate with Pan's crawled data and RecipeQA
                gt_sample_path = os.path.join(gt_data_path, f"task_{task_idx}")
                if not os.path.exists(sample_path): continue
                step_num = len(glob.glob1(gt_sample_path,"step_[0-9]*_bridge_caption.txt")) or len(glob.glob1(gt_sample_path,"step_[0-9]*_caption.txt")) or len(glob.glob1(gt_sample_path,"step_[0-9]*.txt"))
                # TODO: load image array
                visual_task_eval_predict = ""
                visual_task_eval_groundtruth = ""
                task_eval_predict = ""
                task_eval_groundtruth = ""
                caption_task_eval_predict = ""
                caption_task_eval_groundtruth = ""
                # skip task evaluate
                for step_idx in range(1, step_num+1):
                    task_eval_predict += self.get_content(gt_sample_path if self.opt.task in ["tgt-u-plan", "tgt-u-plan-dalle"] else sample_path, step_idx, "")
                    task_eval_groundtruth += self.get_content(gt_sample_path, step_idx, "")
                    caption_task_eval_predict += self.get_content(sample_path, step_idx, "_caption")
                    caption_task_eval_groundtruth += self.get_content(gt_sample_path, step_idx, "_caption")               
        else:
            return self.calculate_sample_score(total_score_cal, task_eval_groundtruth, task_eval_predict, visual_task_eval_groundtruth, visual_task_eval_predict, caption_task_eval_groundtruth, caption_task_eval_predict)
        
    
    def eval_all(self, outpath, use_bridge):        
        total_score_cal = {"sentence-bleu": 0, "wmd": 0, "rouge-1-f": 0, "rouge-1-p": 0, "rouge-1-r": 0, "bert-score-f": 0, "bert-score-p": 0, "bert-score-r": 0, "meteor": 0, "sentence-bert-score": 0, "caption-t-bleu": 0, "caption-vcap-bleu": 0, "lcs": 0, "caption-t-lcs": 0, "caption-vcap-lcs": 0, "gpt3-plan-accuracy": 0, "caption-gpt3-plan-accuracy": 0, "vplan-t-clip-score": 0, "tplan-v-clip-score": 0, "vplan-v-clip-score": 0, "tplan-t-clip-score": 0, "vplan-tplan-clip-score": 0, "caption-t-sent-bert": 0, "caption-t-bert-f1": 0, "caption-vcap-sent-bert": 0, "caption-vcap-bert-f1": 0, "vplancap-tplan-sent-bert": 0, "vplancap-tplan-bert-f1": 0, "avg_length": 0, "gt_avg_length": 0}
        predict_data_path = outpath
        gt_data_path = os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input/{}".format(self.opt.data_type))
        task_num = self.opt.task_num if self.opt.task_num > 0 else len(os.listdir(predict_data_path))
        for task_idx in tqdm(range(task_num)):
            predict_sample_path = os.path.join(predict_data_path, f"task_{task_idx}")
            gt_sample_path = os.path.join(gt_data_path, f"task_{task_idx}")
            if not os.path.exists(predict_sample_path): continue
            visual_task_eval_predict = ""
            visual_task_eval_groundtruth = ""
            task_eval_predict = ""
            task_eval_groundtruth = ""
            caption_task_eval_predict = ""
            caption_task_eval_groundtruth = ""
            # skip task evaluate
            # TODO: fix bug, step num of gt and predict are different, should load seperately
            step_num = len(glob.glob1(predict_sample_path,"step_[0-9]*_bridge_caption.txt")) or len(glob.glob1(predict_sample_path,"step_[0-9]*_caption.txt")) or len(glob.glob1(predict_sample_path,"step_[0-9]*.txt"))
            total_score_cal["avg_length"] += step_num
            for step_idx in range(1, step_num+1):
                tep = self.get_content(gt_sample_path if (self.task_name in ["tgt-u-plan", "tgt-u-plan-dalle"] and use_bridge == "") else predict_sample_path, step_idx, f"{use_bridge}_caption" if self.task_name in ["vgt-u-plan", "vgt-u-plan-blip"] else f"{use_bridge}")
                if len(use_bridge): tep = f"Step {step_idx}: {tep}"
                task_eval_predict = task_eval_predict + " " + tep
                ctep = self.get_content(predict_sample_path, step_idx, f"{use_bridge}_caption")
                ctep = f"Step {step_idx}: {ctep}"
                caption_task_eval_predict = caption_task_eval_predict + " " + ctep
                total_score_cal = self.calculate_sample_step_score(predict_data_path, task_idx, step_idx, total_score_cal, tep, ctep, use_bridge, step_num)
            step_num = len(glob.glob1(gt_sample_path,"step_[0-9]*_bridge_caption.txt")) or len(glob.glob1(gt_sample_path,"step_[0-9]*_caption.txt")) or len(glob.glob1(gt_sample_path,"step_[0-9]*.txt"))
            total_score_cal["gt_avg_length"] += step_num
            for step_idx in range(1, step_num+1):
                teg = self.get_content(gt_sample_path, step_idx, "")
                task_eval_groundtruth += teg
                cteg = self.get_content(gt_sample_path, step_idx, "_caption")
                caption_task_eval_groundtruth += cteg
            total_score_cal = self.calculate_sample_score(total_score_cal, task_eval_groundtruth, task_eval_predict, visual_task_eval_groundtruth, visual_task_eval_predict, caption_task_eval_groundtruth, caption_task_eval_predict)
            # ic(total_score_cal)
        # mean value
        for score_key in total_score_cal.keys():
            total_score_cal[score_key] /= task_num
            total_score_cal[score_key] = round(total_score_cal[score_key], 4)
        # resultfile.writelines(result_list)
        # json.dump(total_score_cal,resultfile)
        # with open(os.path.join(predict_data_path, f"all_total_score_cal{use_bridge}.txt"), 'w') as resultfile:
        #     json.dump(total_score_cal, resultfile)
        csv_line = [self.task_name] + [use_bridge] + [total_score_cal["sentence-bleu"]] + [total_score_cal["wmd"]] + [total_score_cal["rouge-1-f"]] + [total_score_cal["bert-score-f"]] + [total_score_cal["meteor"]] + [total_score_cal["sentence-bert-score"]] + [total_score_cal["vplan-tplan-clip-score"]] + [total_score_cal["avg_length"]] + [total_score_cal["gt_avg_length"]]
        return csv_line