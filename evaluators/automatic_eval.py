import os
from sentence_transformers import SentenceTransformer, util as st_utils
import clip
import torch
import evaluate
import spacy
# from cider import Cider
import tqdm
import glob

class Automatic_Evaluator(object):
    def __init__(self, opt) -> None:
        self.opt = opt
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

    def calculate_sample_score(self, total_score_cal, task_eval_groundtruth, task_eval_predict, visual_task_eval_groundtruth, visual_task_eval_predict, caption_task_eval_groundtruth, caption_task_eval_predict):
        import nltk
        task_eval_groundtruth = task_eval_groundtruth.replace('.', ' ')
        task_eval_predict = task_eval_predict.replace('.', ' ')
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
            pass
        
        total_score_cal["meteor"] += self.meteor.compute(predictions=[task_eval_predict], references=[task_eval_groundtruth])
        
        # Table 2
        total_score_cal["sentence-bert-score"] += self.similariry_score(task_eval_predict, task_eval_groundtruth)
        
        total_score_cal["caption-t-bleu"] += 0
        total_score_cal["caption-vcap-bleu"] += 0
        
        total_score_cal["lcs"] += self.lcs(task_eval_predict, task_eval_groundtruth, len(task_eval_predict), len(task_eval_groundtruth))
        total_score_cal["caption-t-lcs"] += 0
        total_score_cal["caption-vcap-lcs"] += 0
        
        total_score_cal["gpt3-plan-accuracy"] += 0
        total_score_cal["caption-gpt3-plan-accuracy"] += 0
        
        # Table 3
        # image_input = preprocess(image).unsqueeze(0).to(device)
        total_score_cal["vplan-t-clip-score"] += 0
        total_score_cal["tplan-v-clip-score"] += 0
        total_score_cal["vplan-v-clip-score"] += 0
        with torch.no_grad():
            pre_features = self.clip_model(torch.cat([clip.tokenize(task_eval_predict)]).to(self.device))
            gt_features = self.clip_model(torch.cat([clip.tokenize(task_eval_groundtruth)]).to(self.device))
            pre_features /= pre_features.norm(dim=-1, keepdim=True)
            gt_features /= gt_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * pre_features @ gt_features.T).softmax(dim=-1)
            total_score_cal["tplan-t-clip-score"] += similarity
        
        # total_score_cal["caption-t-sent-bert"] += 0
        # total_score_cal["caption-t-bert-f1"] += 0
        # total_score_cal["caption-vcap-sent-bert"] += 0
        # total_score_cal["caption-vcap-bert-f1"] += 0
        # total_score_cal["tplan-t-cider"] += self.cider.compute_score(task_eval_groundtruth, task_eval_predict)
        # total_score_cal["caption-t-cider"] += 0
        # total_score_cal["caption-vcap-cider"] += 0
        
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
            gt_data_path = os.path.join("/share/edc/home/yujielu/MPP_data/dataset/{}".format(self.opt.data_type))
            for task_idx in tqdm(range(task_num)):
                sample_path = os.path.join(from_task_path, f"task_{task_idx}")
                # TODO: incorporate with Pan's crawled data and RecipeQA
                gt_sample_path = os.path.join(gt_data_path, f"task_{task_idx}")
                if not os.path.exists(sample_path): continue
                step_num = len(glob.glob1(sample_path,"step_*.txt"))
                # TODO: load image array
                visual_task_eval_predict = ""
                visual_task_eval_groundtruth = ""

                for step_idx in range(step_num):
                    task_eval_predict = self.get_content(sample_path, step_idx, "")
                    task_eval_groundtruth = self.get_content(gt_sample_path, step_idx, "")
                    caption_task_eval_predict = self.get_content(sample_path, step_idx, "_caption")
                    caption_task_eval_groundtruth = self.get_content(gt_sample_path, step_idx, "_caption")

                
        else:
            return self.calculate_sample_score(total_score_cal, task_eval_groundtruth, task_eval_predict, visual_task_eval_groundtruth, visual_task_eval_predict, caption_task_eval_groundtruth, caption_task_eval_predict)