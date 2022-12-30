import os
from mpp_utils.data_loader import Data_Loader
from icecream import ic
import json
import glob
import shutil

from LLM_Reasoning import LLM_Reasoning
from preprocessors.generate_caption import generate_caption
from Image_Generation import Image_Generation
from evaluators.automatic_eval import Automatic_Evaluator
from Base_Planning import Base_Planner

class Baseline_Planner(Base_Planner):
    def __init__(self, opt, config, outpath) -> None:
        super().__init__(opt)
        self.outpath = outpath
        self.opt = opt
        self.config = config
        self.automatic_evaluator = Automatic_Evaluator(self.opt)
        self.data_loader = Data_Loader(opt)
                
    def open_loop_visual_plan_generation(self):
        data, task_start_idx_list, _ = self.data_loader.load_sample(self.opt, self.config, load_task=False, out_path=self.outpath)
        ic(len(task_start_idx_list))
        opt, config, outpath = self.opt, self.config, self.outpath
        # ic(data, task_start_idx_list)
        image_generator = Image_Generation(opt, config, outpath)
        image_generator.generate_image(data, task_start_idx_list)
    
    def open_loop_textual_plan_generation(self, summarize_example_data_list):
        opt, config, outpath = self.opt, self.config, self.outpath
        task_result_dir = outpath
        if self.opt.task == "vgt-u-plan": #self.config.mpp_model.task_config.ground_truth_modality == "visual": # image caption
            exist_task_num = len(os.listdir(task_result_dir))
            if exist_task_num == 0 or (opt.resume and exist_task_num < opt.task_num):
                for task_idx in range(0 if not opt.resume else exist_task_num-1, opt.task_num):
                    task_path = os.path.join(task_result_dir, "task_{}".format(task_idx))
                    os.makedirs(task_path, exist_ok=True)
                    gt_task_path = os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input", opt.data_type, "task_{}".format(task_idx))
                    step_num = len(glob.glob1(gt_task_path,"step_*.txt"))
                    shutil.copyfile(os.path.join(gt_task_path, "task.txt"), os.path.join(task_path, "task.txt"))
                    for step_idx in range(1, step_num+1):
                        if opt.data_type == "wikihow":
                            img_name = f"step_{step_idx}.png"
                            shutil.copyfile(os.path.join(gt_task_path, img_name), os.path.join(task_path, img_name))
                        else:
                            img_name = f"step_{step_idx}.jpg"
                            shutil.copyfile(os.path.join(gt_task_path, img_name), os.path.join(task_path, img_name))
            generate_caption(task_result_dir)
        else:
            llm_reasoning_engine = LLM_Reasoning(opt)
            llm_reasoning_engine.generate_language_plan(opt, task_result_dir, summarize_example_data_list)

                
    def start_planning(self):
        _, _, summarize_example_data_list = self.data_loader.load_sample(self.opt, self.config, load_task=True, out_path=self.outpath)
        # if self.opt.task_num > 0: summarize_example_data_list = summarize_example_data_list[:self.opt.task_num]
        if self.opt.task == "u-plan":
            ic(summarize_example_data_list)
            self.open_loop_textual_plan_generation(summarize_example_data_list)
            self.open_loop_visual_plan_generation()
        elif self.opt.task == "tgt-u-plan": # text to image generation
            self.open_loop_visual_plan_generation()
        elif self.opt.task == "vgt-u-plan": # image caption
            self.open_loop_textual_plan_generation(None)
        # eval_path = self.outpath # "/share/edc/home/yujielu/MPP_data/test_config/wikihow/u-plan/"
        # self.automatic_evaluator.calculate_total_score(total_score_cal=self.total_score_cal, from_task_path=eval_path)