import os
from mpp_utils.data_loader import load_sample
from icecream import ic
import json

from LLM_Reasoning import LLM_Reasoning
from preprocessors import generate_caption
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

    def open_loop_visual_plan_generation(self):
        data, task_start_idx_list, _ = load_sample(self.opt, self.config)
        opt, config, outpath = self.opt, self.config, self.outpath
        image_generator = Image_Generation(opt, config, outpath)
        image_generator.generate_image(opt, data, task_start_idx_list)
    
    def open_loop_textual_plan_generation(self, summarize_example_data_list):
        opt, config, outpath = self.opt, self.config, self.outpath
        task_result_dir = os.path.join(outpath, opt.data_type, "bridge" if opt.use_bridge else "origin", opt.task+("_w_task_hint" if opt.use_task_hint else ""))
        if self.opt.task == "vgt-u-plan": #self.config.mpp_model.task_config.ground_truth_modality == "visual": # image caption
            generate_caption(task_result_dir)
        else:
            llm_reasoning_engine = LLM_Reasoning(opt)
            llm_reasoning_engine.generate_language_plan(opt, task_result_dir, summarize_example_data_list)

                
    def start_planning(self):
        _, _, summarize_example_data_list = load_sample(self.opt, self.config, load_task=True)
        if self.opt.task == "u-plan":
            # self.open_loop_textual_plan_generation(summarize_example_data_list)
            self.open_loop_visual_plan_generation()
        elif self.opt.task == "tgt-u-plan": # text to image generation
            self.open_loop_visual_plan_generation()
        elif self.opt.task == "vgt-u-plan": # image caption
            self.open_loop_textual_plan_generation(None)
        eval_path = self.outpath # "/share/edc/home/yujielu/MPP_data/test_config/wikihow/u-plan/"
        self.automatic_evaluator.calculate_total_score(total_score_cal=self.total_score_cal, from_task_path=eval_path)