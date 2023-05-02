from itertools import islice
from icecream import ic
import os
from tqdm import tqdm
import glob
from LLM_Reasoning import LLM_Reasoning

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

class Data_Loader(object):
    def __init__(self, opt) -> None:
        self.llm_reasoning_engine = LLM_Reasoning(opt)

    def load_sample(self, opt, config, load_task=False, out_path="", load_caption=False):
        batch_size = opt.n_samples
        data = []
        task_start_idx_list = []
        summarize_example_data_list = []
        before_revision_example_list = []
        caption_list = []
        all_step_count = 0
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]
        else:
            exist_task_num = len(os.listdir(out_path))
            if opt.task in ["tgt-u-plan", "tgt-u-plan-dalle", "c-plan"] or load_task:
                gt_data_path = os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input/{}".format(opt.data_type))
                task_num = opt.task_num if opt.task_num > 0 else len(os.listdir(gt_data_path))
                for task_idx in tqdm(range(0 if not opt.resume else exist_task_num-1, task_num)):
                    step_list = []
                    caption_list = []
                    task_start_idx_list.append(all_step_count)
                    all_step_count += 1
                    gt_sample_path = os.path.join(gt_data_path, f"task_{task_idx}")
                    step_num = len(glob.glob1(gt_sample_path,"step_[0-9]*_bridge_caption.txt")) or len(glob.glob1(gt_sample_path,"step_[0-9]*_caption.txt")) or len(glob.glob1(gt_sample_path,"step_[0-9]*.txt"))
                    with open(os.path.join(gt_sample_path, f"task.txt"), 'r') as ft:
                        task = ft.readline()
                        data.append(task)
                    for step_idx in range(1, step_num+1):
                        with open(os.path.join(gt_sample_path, f"step_{step_idx}.txt"), 'r') as fs:
                            step = fs.readline()
                            data.append(step)
                            step_list.append(step)
                        if load_caption:
                            with open(os.path.join(gt_sample_path, f"step_{step_idx}_caption.txt"), 'r') as fc:
                                caption = fc.readline()
                                caption_list.append(f"Caption {step_idx}: {caption}")
                        all_step_count += 1
                    summarize_example_data_list.append({"tasks": task, "steps": step_list, "captions": caption_list})
            elif opt.task in ["u-plan", "m-plan"]:
                task_num = opt.task_num if opt.task_num > 0 else len(os.listdir(gt_data_path))
                all_step_count = 0

                for task_idx in tqdm(range(0 if not opt.resume else exist_task_num-1, task_num)):
                    step_list = []
                    caption_list = []
                    sample_path = os.path.join(out_path, f"task_{str(task_idx)}")
                    if not os.path.exists(sample_path):
                        continue
                    task_start_idx_list.append(all_step_count)
                    all_step_count += 1
                    with open(os.path.join(sample_path, f"task.txt"), 'r') as f:
                        task = f.readline()
                        data.append(task)
                    step_num = len(glob.glob1(sample_path,"step_[0-9]*_bridget2i-0_caption.txt")) or len(glob.glob1(sample_path,"step_[0-9]*_bridge_caption.txt")) or len(glob.glob1(sample_path,"step_[0-9]*_caption.txt")) or len(glob.glob1(sample_path,"step_[0-9]*_bridge.txt")) or len(glob.glob1(sample_path,"step_[0-9]*.txt"))
                    for step_idx in range(1, step_num+1):
                        all_step_count += 1
                        with open(os.path.join(sample_path, f"step_{step_idx}.txt"), 'r') as f:
                            current_text = f.readline()
                            if opt.use_task_hint:
                                current_text = "The task is {} {}".format(task, current_text)
                            step_list.append(current_text)
                            data.append(current_text)
                        if load_caption:
                            with open(os.path.join(sample_path, f"step_{step_idx}_caption.txt"), 'r') as fc:
                                caption = fc.readline()
                                caption_list.append(f"Caption {step_idx}: {caption}")
                    if load_caption:
                        summarize_example_data_list.append({"tasks": task, "steps": step_list, "captions": caption_list})
            data = list(chunk(data, batch_size))
        return data, task_start_idx_list, summarize_example_data_list