import os
import openai
from mpp_utils.data_loader import Data_Loader
from LLM_Reasoning import LLM_Reasoning
from Image_Generation import Image_Generation
from evaluators.automatic_eval import Automatic_Evaluator
from Base_Planning import Base_Planner
from icecream import ic

class MPP_Planner(Base_Planner):
    """Option1: Closed-loop Procedural Planning Option2: first generate textual plan candidate pool, and then use captions of text-to-image visual plan and prompt GPT3 whether it can complete the task. and then rerank textual plan according visual plan correctness"""
    def __init__(self, opt, config, outpath) -> None:
        super().__init__(opt)
        self.outpath = outpath
        self.data_loader = Data_Loader(opt)
        self.data, self.task_start_idx_list, self.summarize_example_data_list = self.data_loader.load_sample(opt, config, load_task=False, out_path=self.outpath)
        self.completion = openai.Completion()

        self.curr_action = ""
        
        self.opt = opt
        self.config = config
        self.llm_reasoning_engine = LLM_Reasoning(self.opt)
        self.image_generator = Image_Generation(self.opt, self.config, self.outpath)
        # self.automatic_evaluator = Automatic_Evaluator(self.opt)
    
    def closed_loop_textual_plan_generation(self, task_result_dir, sample, step_idx):
        # Closed-loop Single Step Textual Plan Generation (LLM, GPT3)
        plan_termination, self.curr_action = self.llm_reasoning_engine.generate_language_plan(self.opt, task_result_dir, None, sample=sample, step_idx=step_idx)
        return plan_termination


    def closed_loop_visual_plan_generation(self, step_idx, task_idx):
        # Closed-loop Single Step Visual Plan Generation (text-to-image model, stable diffusion v2)
        task_start_idx_list = []
        batch_size = self.opt.n_samples
        prompt = self.curr_action
        assert prompt is not None
        data = [batch_size * [prompt]]
        
        self.image_generator.generate_image(data, task_start_idx_list=task_start_idx_list, step_idx=step_idx, task_idx=task_idx)

 
    def temporal_extended_mpp(self, sample, task_idx):
        # Temporal-extended multimodal procedural planning
        task_result_dir = os.path.join(self.outpath, "task_{}".format(task_idx))
        step_idx = 0
        while True:
            step_idx += 1
            plan_termination = self.closed_loop_textual_plan_generation(task_result_dir, sample, step_idx)
            if plan_termination: break
            self.closed_loop_visual_plan_generation(step_idx, task_idx)
            

    def start_planning(self, open_loop=False):
        # load task list
        if open_loop:
            # text plan revise
            pass
        else: # c-plan
            # close loop condition on multimodal generated plan
            # if self.opt.task_num > 0: self.summarize_example_data_list = self.summarize_example_data_list[:self.opt.task_num]
            exist_task_num = len(os.listdir(self.outpath))
            for task_idx, sample in enumerate(self.summarize_example_data_list):
                start_task_idx = (task_idx + exist_task_num - 1) if self.opt.resume else task_idx
                ic(start_task_idx)
                self.temporal_extended_mpp(sample, start_task_idx)
        
        # eval_path = self.outpath # "/share/edc/home/yujielu/MPP_data/test_config/wikihow/u-plan/"
        # self.automatic_evaluator.calculate_total_score(total_score_cal=self.total_score_cal, from_task_path=eval_path)