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

    def load_sample(self, opt, config, load_task=False, out_path=""):
        batch_size = opt.n_samples
        data = []
        task_start_idx_list = []
        summarize_example_data_list = []
        all_step_count = 0
        if not opt.from_file:
            prompt = opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]
        else:
            exist_task_num = len(os.listdir(out_path))
            if opt.task in ["tgt-u-plan", "c-plan", "m-plan"] or load_task:
                gt_data_path = os.path.join("/share/edc/home/yujielu/MPP_data/groundtruth_input/{}".format(opt.data_type))
                task_num = opt.task_num if opt.task_num > 0 else len(os.listdir(gt_data_path))
                # if exist_task_num == 0 or (opt.resume and exist_task_num < opt.task_num):
                # for task_idx in range(0 if not opt.resume else exist_task_num-1, opt.task_num):
                for task_idx in tqdm(range(0 if not opt.resume else exist_task_num-1, task_num)):
                    step_list = []
                    task_start_idx_list.append(all_step_count)
                    all_step_count += 1
                    gt_sample_path = os.path.join(gt_data_path, f"task_{task_idx}")
                    step_num = len(glob.glob1(gt_sample_path,"step_[0-9]_caption.txt")) or len(glob.glob1(gt_sample_path,"step_[0-9].txt"))
                    with open(os.path.join(gt_sample_path, f"task.txt"), 'r') as ft:
                        task = ft.readline()
                        data.append(task)
                    ic(gt_sample_path, step_num)
                    for step_idx in range(1, step_num+1):
                        with open(os.path.join(gt_sample_path, f"step_{step_idx}.txt"), 'r') as fs:
                            step = fs.readline()
                            data.append(step)
                            step_list.append(step)
                        all_step_count += 1
                    summarize_example_data_list.append({"tasks": task, "steps": step_list})
            elif opt.task in ["u-plan"]:
                # "bridge" if opt.use_bridge else "origin"
                task_num = opt.task_num if opt.task_num > 0 else len(os.listdir(gt_data_path))
                all_step_count = 0

                for task_idx in tqdm(range(0 if not opt.resume else exist_task_num-1, task_num)):
                    # data = []
                    step_list = []
                    sample_path = os.path.join(out_path, f"task_{str(task_idx)}")
                    if not os.path.exists(sample_path):
                        continue
                    task_start_idx_list.append(all_step_count)
                    all_step_count += 1
                    # step_num = len(os.listdir(sample_path))
                    with open(os.path.join(sample_path, f"task.txt"), 'r') as f:
                        task = f.readline()
                        data.append(task)
                    step_num = len(glob.glob1(sample_path,"step_[0-9]_caption.txt")) or len(glob.glob1(sample_path,"step_[0-9].txt"))
                    for step_idx in range(1, step_num+1):
                        all_step_count += 1
                        with open(os.path.join(sample_path, f"step_{step_idx}.txt"), 'r') as f:
                            current_text = f.readline()
                            if opt.use_task_hint:
                                current_text = "The task is {} {}".format(task, current_text)
                            # if opt.use_bridge:
                            #     current_text = self.llm_reasoning_engine.ask_prompt(current_text)
                            # step_list.append(current_text)
                            data.append(current_text)
                    # summarize_example_data_list.append({"tasks": task, "steps": step_list})
            data = list(chunk(data, batch_size))
        return data, task_start_idx_list, summarize_example_data_list



    # if opt.task in ["tgt-u-plan", "m-plan"] or load_task:
    #     print(f"reading prompts from {opt.from_file}")
    #     if opt.file_type == "json":
    #         import json
    #         with open('/local/home/yujielu/project/MPP/submodules/GoalAgent/data/{}/{}_available_examples.json'.format(opt.data_type, opt.data_type), 'r') as f:
    #             available_examples = json.load(f)
    #             if opt.task_num > 0: available_examples = available_examples[:opt.task_num]
    #         # example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
    #         # example_task_embedding = translation_lm.encode(example_task_list, batch_size=args.encode_batch_size, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory
    #         data = []
    #         idx = 0
    #         for example in available_examples:
    #             task_start_idx_list.append(idx)
    #             for step in example.split('\n'):
    #                 if opt.use_task_hint and idx > 0:
    #                     step = "The task is {} {}".format(data[0][6:], step)
    #                 data.append(step)
    #                 idx += 1
    #             summarize_example_data_list.append({"tasks": example.split('\n')[0], "steps": '.'.join(example.split('\n')[1:])})
    #     else:
    #         with open(opt.from_file, "r") as f:
    #             data = f.read().splitlines()
    #             data = [p for p in data for i in range(opt.repeat)]
    #             old_data = data.copy()
    #             data = []
    #             idx = 0
    #             for p in old_data:
    #                 new_p = p.split("##STEP##")
    #                 task_start_idx_list.append(idx)
    #                 idx += 1
    #                 for item in new_p:
    #                     idx += 1
    #                     if len(item) > 1:
    #                         data.append(item)
    # elif opt.task in ["u-plan"]:
    #     resolution_config = "resolution_{}".format(opt.resolution)
    #     text_path = os.path.join(opt.outdir, "debug_output" if opt.debug else "experiment_output", resolution_config, opt.data_type, "bridge" if opt.use_bridge else "origin", opt.task+("_w_task_hint" if opt.use_task_hint else "")) # "/share/edc/home/yujielu/MPP_data/test_config/wikihow/u-plan/"
    #     task_num = len(os.listdir(text_path))
    #     all_step_count = 0
    #     for task_idx in tqdm(range(task_num)):
    #         data = []
    #         sample_path = os.path.join(text_path, f"task_{str(task_idx)}")
    #         if not os.path.exists(sample_path):
    #             continue
    #         task_start_idx_list.append(all_step_count)
    #         # step_num = len(os.listdir(sample_path))
    #         with open(os.path.join(sample_path, f"task.txt"), 'r') as f:
    #             task = f.readline()
    #         step_num = len(glob.glob1(sample_path,"step_*.txt"))
    #         for step_idx in range(1, step_num+1):
    #             all_step_count += 1
    #             with open(os.path.join(sample_path, f"step_{step_idx}.txt"), 'r') as f:
    #                 current_text = f.readline()
    #                 if opt.use_task_hint:
    #                     current_text = "The task is {} {}".format(task, current_text)
    #                 data.append(current_text)
    #         summarize_example_data_list.append({"tasks": task, "steps": data})
    # data = list(chunk(data, batch_size))