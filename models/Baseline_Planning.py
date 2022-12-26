import os
import torch
from stablediffusion.scripts.mpp_utils.prompt_process import load_prompt
from icecream import ic
import json

from LLM_Reasoning import LLM_Reasoning
from preprocess import generate_caption
from Image_Generation import Image_Generation

class Baseline_Planner(object):
    def __init__(self, opt, config, outpath, data, task_start_idx_list, summarize_example_data_list) -> None:
        self.outpath = outpath
        self.summarize_example_data_list = summarize_example_data_list
        
        method_type=opt.model_type
        self.total_score = {}
        all_model_type_list = ['{}'.format(method_type)]
        for model_type_item in all_model_type_list:
            self.total_score[model_type_item] = {"sentence-bleu": 0, "bert-score-f": 0, "LCS": 0, "CLIP": 0}
            
        self.data, self.task_start_idx_list, self.summarize_example_data_list = load_prompt(opt, config)

    def open_loop_visual_plan_generation(self, opt, data, task_start_idx_list):
        image_generator = Image_Generation(opt)
        image_generator.generate_image(opt, data, task_start_idx_list)
    
    def open_loop_textual_plan_generation(self, opt, outpath, summarize_example_data_list):
        task_result_dir = os.path.join(outpath, opt.data_type, opt.task)
        if self.config.mpp_model.task_config.ground_truth_modality == "visual": # image caption
            generate_caption(task_result_dir)
        else:
            llm_reasoning_engine = LLM_Reasoning(opt)
            # task_path = opt.data_type + "/" + method_type + "/" + datetime.now().strftime("%Y_%m_%d") + "/" + "demo_{}_{}_inter{}_var{}_heldout{}_oloop{}_".format(opt.language_model_type, opt.model_type, opt.open_loop) + datetime.now().strftime("%H%M%S")
            # task_result_dir = os.path.join("../result", task_path)
            if not os.path.isdir(task_result_dir): os.makedirs(task_result_dir)
            skip_count = 0
            with open(os.path.join(task_result_dir, "{}_task_result.txt".format(opt.language_model_type)), 'w') as resultfile:
                total_score_cal = self.total_score.copy()
                for data_example in summarize_example_data_list[1:]:
                    
                    total_score_cal, result_list = llm_reasoning_engine.language_planning(total_score_cal, data_example)

                # mean value
                ic(len(summarize_example_data_list), total_score_cal[opt.model_type].keys())
                for score_key in total_score_cal[opt.model_type].keys():
                    total_score_cal[opt.model_type][score_key] /= (len(summarize_example_data_list)-skip_count)
                resultfile.writelines(result_list)
                json.dump(total_score_cal,resultfile)
                ic(skip_count, total_score_cal[opt.model_type])
                
    def start_planning(self):
        data, task_start_idx_list, summarize_example_data_list = load_prompt(self.opt, self.config)
        if self.opt.task == "u-plan":
            self.open_loop_textual_plan_generation(self.opt, self.outpath, summarize_example_data_list)
            self.open_loop_visual_plan_generation(self.opt, data, task_start_idx_list)
        elif self.opt.task == "tgt-u-plan": # text to image generation
            self.open_loop_visual_plan_generation(self.opt, data, task_start_idx_list)
        elif self.opt.task == "vgt-u-plan": # image caption
            self.open_loop_textual_plan_generation(self.opt, self.outpath, summarize_example_data_list)