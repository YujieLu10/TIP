import os
import openai
from mpp_utils.data_loader import load_sample
from LLM_Reasoning import LLM_Reasoning
from Image_Generation import Image_Generation
from evaluators.automatic_eval import Automatic_Evaluator
from Base_Planning import Base_Planner
from icecream import ic

MAX_STEPS = 5
class MPP_Planner(Base_Planner):
    """Option1: Closed-loop Procedural Planning Option2: first generate textual plan candidate pool, and then use captions of text-to-image visual plan and prompt GPT3 whether it can complete the task. and then rerank textual plan according visual plan correctness"""
    def __init__(self, opt, config, outpath) -> None:
        super().__init__(opt)
        self.outpath = outpath
        self.data, self.task_start_idx_list, self.summarize_example_data_list = load_sample(opt, config)
        self.completion = openai.Completion()

        self.task_prediced_steps = []
        self.curr_prompt = ""
        self.curr_action = ""
        self.result_list = []
        self.step_sequence = []
        self.task_eval_predict = ""
        
        self.opt = opt
        self.config = config
        self.llm_reasoning_engine = LLM_Reasoning(self.opt)
        self.image_generator = Image_Generation(self.opt, self.config, self.outpath)
        self.automatic_evaluator = Automatic_Evaluator(self.opt)
    
    def closed_loop_textual_plan_generation(self, step_idx):
        # Closed-loop Single Step Textual Plan Generation (LLM, GPT3)
        best_action = self.llm_reasoning_engine.ask(self.curr_prompt)
        plan_termination = step_idx > MAX_STEPS # GPT3 reach end token EOS
        if plan_termination: return plan_termination

        self.task_prediced_steps.append(best_action)
        self.curr_prompt += f'\nStep {step_idx}: {best_action}.'
        self.curr_action = f'\nStep {step_idx}: {best_action}.'
        self.result_list.append(f'Step {step_idx}: {best_action}.')
        self.step_sequence.append(best_action)
        self.task_eval_predict += f'Step {step_idx}: {best_action}.'        
        return plan_termination


    def closed_loop_visual_plan_generation(self, step_idx):
        # Closed-loop Single Step Visual Plan Generation (text-to-image model, stable diffusion v2)
        task_start_idx_list = []
        batch_size = self.opt.n_samples
        prompt = self.curr_action
        assert prompt is not None
        data = [batch_size * [prompt]]
        
        self.image_generator.generate_image(self.opt, data, task_start_idx_list=task_start_idx_list, step_idx=step_idx)

 
    def temporal_extended_mpp(self, sample):
        # Temporal-extended multimodal procedural planning
        self.curr_prompt = sample["tasks"]
        step_idx = 0
        while True:
            step_idx += 1
            plan_termination = self.closed_loop_textual_plan_generation(step_idx)
            if plan_termination: break
            self.closed_loop_visual_plan_generation(step_idx)
            

    def start_planning(self):
        # load task list
        ic(self.summarize_example_data_list)
        if self.opt.task_num > 0: self.summarize_example_data_list = self.summarize_example_data_list[:self.opt.task_num]
        for sample in self.summarize_example_data_list:
            self.temporal_extended_mpp(sample)
        
        eval_path = self.outpath # "/share/edc/home/yujielu/MPP_data/test_config/wikihow/u-plan/"
        self.automatic_evaluator.calculate_total_score(total_score_cal=self.total_score_cal, from_task_path=eval_path)